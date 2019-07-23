#include "..\include\primitives.h"


#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/Point_with_normal_3.h>
#include <CGAL/property_map.h>
#include <CGAL/Shape_detection_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel  Kernel;
typedef Kernel::FT                                           FT;
typedef std::pair<Kernel::Point_3, Kernel::Vector_3>         Point_with_normal;
typedef std::vector<Point_with_normal>                       Pwn_vector;
typedef CGAL::First_of_pair_property_map<Point_with_normal>  Point_map;
typedef CGAL::Second_of_pair_property_map<Point_with_normal> Normal_map;

// In Efficient_RANSAC_traits the basic types, i.e., Point and Vector types
// as well as iterator type and property maps, are defined.
typedef CGAL::Shape_detection_3::Shape_detection_traits<Kernel,
	Pwn_vector, Point_map, Normal_map>            Traits;
typedef CGAL::Shape_detection_3::Efficient_RANSAC<Traits> Efficient_ransac;
typedef CGAL::Shape_detection_3::Cone<Traits>             CGALCone;
typedef CGAL::Shape_detection_3::Cylinder<Traits>         CGALCylinder;
typedef CGAL::Shape_detection_3::Plane<Traits>            CGALPlane;
typedef CGAL::Shape_detection_3::Sphere<Traits>           CGALSphere;
typedef CGAL::Shape_detection_3::Torus<Traits>            CGALTorus;

lmu::PointCloud getPoints(const CGAL::Shape_detection_3::Shape_base<Traits>& shape, const Pwn_vector& pointsWithNormals)
{
	lmu::PointCloud points;
	points.resize(shape.indices_of_assigned_points().size(), 6);
	int i = 0;
	for (const auto& index : shape.indices_of_assigned_points())
	{
		auto pn = pointsWithNormals[index];

		Eigen::Matrix<double, 1, 6> rowVec;
		rowVec << pn.first.x(), pn.first.y(), pn.first.z(), pn.second.x(), pn.second.y(), pn.second.z();
		points.row(i) = rowVec;
		i++;
	}

	//std::cout << "Total points: " << pointsWithNormals.size() << " Assigned points: " << points.rows() << std::endl;

	return points;
}

std::string lmu::primitiveTypeToString(PrimitiveType type)
{
	switch (type)
	{
	case PrimitiveType::None:
		return "None";
	case PrimitiveType::Cylinder:
		return "Cylinder";
	case PrimitiveType::Sphere:
		return "Sphere";
	case PrimitiveType::Cone:
		return "Cone";
	case PrimitiveType::Box:
		return "Box";
	default:
		return "Unknown Primitive Type";
	}
}

lmu::PrimitiveType lmu::primitiveTypeFromString(std::string type)
{
	std::transform(type.begin(), type.end(), type.begin(), ::tolower);

	if (type == "cylinder")
		return PrimitiveType::Cylinder;
	else if (type == "sphere")
		return PrimitiveType::Sphere;
	else if (type == "cone")
		return PrimitiveType::Cone;
	else if (type == "box")
		return PrimitiveType::Cone;
	else
		return PrimitiveType::None;
}

lmu::RansacResult lmu::mergeRansacResults(const std::vector<RansacResult>& results)
{
	RansacResult res;

	if (results.empty())
		return res;
	
	size_t pointCloudRows = 0;
	size_t pointCloudCols = 0;
	for (const auto& r : results)
	{
		pointCloudCols = r.pc.cols();
		pointCloudRows += r.pc.rows();
	}

	PointCloud newPointCloud(pointCloudRows, pointCloudCols);
	int rowIdx = 0;
	for (const auto& r : results)
	{
		res.manifolds.insert(res.manifolds.end(), r.manifolds.begin(), r.manifolds.end());

		std::cout << "Size: " << r.pc.rows() << std::endl;

		newPointCloud.block(rowIdx,0, r.pc.rows(), r.pc.cols()) << r.pc;

		rowIdx += r.pc.rows();
	}
	res.pc = newPointCloud;

	return res;
}

lmu::RansacResult lmu::extractManifoldsWithCGALRansac(const lmu::PointCloud& pc, const lmu::RansacParams& params, bool projectPointsOnSurface)
{
	//auto pcChars = getPointCloudCharacteristics(pc, 1, 0.5);
	//std::cout << pcChars.maxDistance << " " << pcChars.minDistance << " " << pcChars.meanDistance << std::endl;
	double diagLength = computeAABBLength(pc);
	//std::cout << "PC DIAG LENGTH: " << diagLength << std::endl;

	// Add points and normals to the correct structure.
	Pwn_vector pointsWithNormals;
	for (int i = 0; i < pc.rows(); i++)
	{
		Eigen::Vector3d p = pc.row(i).leftCols(3); //pc.block<1, 3>(i, 0);
		Eigen::Vector3d n = pc.row(i).rightCols(3);//pc.block<1, 3>(i, 3);

		//std::cout << "---------------------------" << std::endl;
		//std::cout << p << " " << n << std::endl;
		//std::cout << pc.row(i) << std::endl;

		Point_with_normal pn;
		pn.first = Kernel::Point_3(p.x(), p.y(), p.z());
		pn.second = Kernel::Vector_3(n.x(), n.y(), n.z());
		pointsWithNormals.push_back(pn);
	}

	Efficient_ransac ransac;
	// Provides the input data.
	ransac.set_input(pointsWithNormals);

	// Register shapes for detection
	if (params.types.empty())
	{
		ransac.add_shape_factory<CGALPlane>();
		ransac.add_shape_factory<CGALSphere>();
		ransac.add_shape_factory<CGALCylinder>();
	}
	else
	{
		if (params.types.count(ManifoldType::Cylinder))
			ransac.add_shape_factory<CGALCylinder>();
		if (params.types.count(ManifoldType::Plane))
			ransac.add_shape_factory<CGALPlane>();
		if (params.types.count(ManifoldType::Sphere))
			ransac.add_shape_factory<CGALSphere>();
	}
	
	//ransac.add_shape_factory<Cone>();
	//ransac.add_shape_factory<Torus>();
	
	// Sets parameters for shape detection.
	Efficient_ransac::Parameters parameters;
	// Sets probability to miss the largest primitive at each iteration.
	parameters.probability = params.probability;//0.1;

	// Detect shapes with at least 500 points.
	parameters.min_points = params.min_points;//(double)pc.rows() * 0.01;

	// Sets maximum Euclidean distance between a point and a shape.
	parameters.epsilon = params.epsilon * diagLength;//2.0 / diagLength; //0.02

	// Sets maximum Euclidean distance between points to be clustered.
	parameters.cluster_epsilon = params.cluster_epsilon * diagLength;//6.0 / diagLength; //0.1

	// Sets maximum normal deviation.
	// 0.9 < dot(surface_normal, point_normal); 
	parameters.normal_threshold = params.normal_threshold;//0.9;

	//ransac.preprocess();
	ransac.detect(parameters);

	std::cout << ransac.shapes().end() - ransac.shapes().begin() << " detected shapes, "
		<< ransac.number_of_unassigned_points()
		<< " unassigned points." << std::endl;

	ManifoldSet manifolds;

	for (const auto& shape : ransac.shapes())
	{
		// Get specific parameters depending on detected shape.
		if (CGALSphere* sphere = dynamic_cast<CGALSphere*>(shape.get()))
		{
			auto m = std::make_shared<lmu::Manifold>(
				lmu::ManifoldType::Sphere,
				Eigen::Vector3d(sphere->center().x(), sphere->center().y(), sphere->center().z()),
				Eigen::Vector3d(0, 0, 0),
				Eigen::Vector3d(sphere->radius(), sphere->radius(), sphere->radius())
				);

			m->pc = getPoints(*shape, pointsWithNormals);

			std::cout << "sphere detected" << std::endl;

			manifolds.push_back(m);
		}
		else if (CGALCylinder* cylinder = dynamic_cast<CGALCylinder*>(shape.get()))
		{
			auto m = std::make_shared<lmu::Manifold>(
				lmu::ManifoldType::Cylinder,
				Eigen::Vector3d(cylinder->axis().point().x(), cylinder->axis().point().y(), cylinder->axis().point().z()),
				Eigen::Vector3d(cylinder->axis().direction().vector().x(), cylinder->axis().direction().vector().y(), cylinder->axis().direction().vector().z()),
				Eigen::Vector3d(cylinder->radius(), cylinder->radius(), cylinder->radius())
				);

			m->pc = getPoints(*shape, pointsWithNormals);

			std::cout << "cylinder detected" << std::endl;

			manifolds.push_back(m);
		}
		else if (CGALPlane* plane = dynamic_cast<CGALPlane*>(shape.get()))
		{
			Eigen::Vector3d n(plane->plane_normal().x(), plane->plane_normal().y(), plane->plane_normal().z());
		
			auto m = std::make_shared<lmu::Manifold>(
				lmu::ManifoldType::Plane,
				-plane->d() * n,
				Eigen::Vector3d(plane->plane_normal().x(), plane->plane_normal().y(), plane->plane_normal().z()),
				Eigen::Vector3d(0,0,0)
				);

			m->pc = getPoints(*shape, pointsWithNormals);

			std::cout << "plane detected" << std::endl;

			manifolds.push_back(m);
		}
	}

	if (projectPointsOnSurface) {
		for (auto& m : manifolds) {
			m->projectPointsOnSurface();
		}
	}	
	RansacResult res;
	res.manifolds = manifolds;
	res.pc = pc;

	return res;
}

#include <RansacShapeDetector.h>
#include <PlanePrimitiveShapeConstructor.h>
#include <CylinderPrimitiveShapeConstructor.h>
#include <SpherePrimitiveShapeConstructor.h>
#include <PlanePrimitiveShape.h>
#include <CylinderPrimitiveShape.h>
#include <SpherePrimitiveShape.h>

// #include <ConePrimitiveShapeConstructor.h>
// #include <TorusPrimitiveShapeConstructor.h>

void compute_bbox(const PointCloud& pc, Vec3f& min_pt, Vec3f& max_pt)
{
	float fmax = std::numeric_limits<float>::max();
	min_pt[0] = fmax;
	min_pt[1] = fmax;
	min_pt[2] = fmax;

	max_pt[0] = -fmax;
	max_pt[1] = -fmax;
	max_pt[2] = -fmax;

	for (unsigned int i = 0; i < pc.size(); ++i) {
		Point p = pc[i];
		min_pt[0] = std::min(min_pt[0], p[0]);
		min_pt[1] = std::min(min_pt[1], p[1]);
		min_pt[2] = std::min(min_pt[2], p[2]);
		max_pt[0] = std::max(max_pt[0], p[0]);
		max_pt[1] = std::max(max_pt[1], p[1]);
		max_pt[2] = std::max(max_pt[2], p[2]);
	}
}

lmu::RansacResult lmu::extractManifoldsWithOrigRansac(const lmu::PointCloud& pc, const lmu::RansacParams& params, bool projectPointsOnSurface)
{
	// Convert point cloud.
	::PointCloud pcConv;
	for (int i = 0; i < pc.rows(); ++i)
	{
		::Point point(
			Vec3f(pc.coeff(i, 0), pc.coeff(i, 1), pc.coeff(i, 2)),
			Vec3f(pc.coeff(i, 3), pc.coeff(i, 4), pc.coeff(i, 5))
		);
		
		//std::cout << pc.row(i).x() << " " << pc.row(i).y() << " " << pc.row(i).z() << std::endl;
		//std::cout << point[0] << " " << point[1] << " " << point[2] << std::endl;

		pcConv.push_back(point);
	}

	// Compute bounding box.
	Vec3f min_pt, max_pt;
	compute_bbox(pcConv, min_pt, max_pt);
	pcConv.setBBox(min_pt, max_pt);

	//std::cout << "Scale: " << pcConv.getScale() << std::endl;

	RansacShapeDetector::Options ransacOptions;
	
	ransacOptions.m_epsilon = params.epsilon * pcConv.getScale();
	ransacOptions.m_bitmapEpsilon = params.cluster_epsilon * pcConv.getScale();
	ransacOptions.m_normalThresh = params.normal_threshold;
	ransacOptions.m_minSupport = params.min_points;
	ransacOptions.m_probability = params.probability;

	RansacShapeDetector detector(ransacOptions);

	if (params.types.count(ManifoldType::Cylinder) || params.types.empty())
		detector.Add(new CylinderPrimitiveShapeConstructor());	
	if (params.types.count(ManifoldType::Plane) || params.types.empty())
		detector.Add(new PlanePrimitiveShapeConstructor());
	if (params.types.count(ManifoldType::Sphere) || params.types.empty())
		detector.Add(new SpherePrimitiveShapeConstructor());

	std::vector< std::pair< MiscLib::RefCountPtr< PrimitiveShape >, size_t > > shapes;

	//std::cout << "Orig Ransac" << std::endl;


	size_t remaining = detector.Detect(pcConv, 0, pcConv.size(), &shapes);
	std::cout << "detection finished " << remaining << std::endl;
	std::cout << "number of shapes: " << shapes.size() << std::endl;

	ManifoldSet manifolds;
	for (auto& shape : shapes)
	{
		lmu::ManifoldPtr m = nullptr; 

		switch (shape.first->Identifier())
		{
		case 0: // Plane
		{
			auto plane = dynamic_cast<PlanePrimitiveShape*>(shape.first.Ptr());
			auto planeParams = plane->Internal();

			m = std::make_shared<lmu::Manifold>(
				lmu::ManifoldType::Plane,
				Eigen::Vector3d(planeParams.getPosition()[0], planeParams.getPosition()[1], planeParams.getPosition()[2]),
				Eigen::Vector3d(planeParams.getNormal()[0], planeParams.getNormal()[1], planeParams.getNormal()[2]).normalized(),
				Eigen::Vector3d(0, 0, 0)
				);			
			break;
		}
		case 1: // Sphere
		{
			auto sphere = dynamic_cast<SpherePrimitiveShape*>(shape.first.Ptr());
			auto sphereParams = sphere->Internal();

			m = std::make_shared<lmu::Manifold>(
				lmu::ManifoldType::Sphere,
				Eigen::Vector3d(sphereParams.Center()[0], sphereParams.Center()[1], sphereParams.Center()[2]),
				Eigen::Vector3d(0, 0, 0),
				Eigen::Vector3d(sphereParams.Radius(), sphereParams.Radius(), sphereParams.Radius())
				);			
			break;
		}
		case 2: // Cylinder
		{
			auto cylinder = dynamic_cast<CylinderPrimitiveShape*>(shape.first.Ptr());
			auto cylinderParams = cylinder->Internal();

			auto m = std::make_shared<lmu::Manifold>(
				lmu::ManifoldType::Cylinder,
				Eigen::Vector3d(cylinderParams.AxisPosition()[0], cylinderParams.AxisPosition()[1], cylinderParams.AxisPosition()[2]),
				Eigen::Vector3d(cylinderParams.AxisDirection()[0], cylinderParams.AxisDirection()[1], cylinderParams.AxisDirection()[2]).normalized(),
				Eigen::Vector3d(cylinderParams.Radius(), cylinderParams.Radius(), cylinderParams.Radius())
				);			
			break;
		}
		}

		if (m)
		{
			manifolds.push_back(m);
			std::cout << *m << std::endl;
		}
	}

	//Assign point clouds to manifolds.
	size_t sum = 0;
	for (unsigned int i = 0; i < shapes.size(); ++i) 
	{
		manifolds[i]->pc = lmu::PointCloud(shapes[i].second, 6);

		size_t k = 0;
		for (unsigned int j = pcConv.size() - (sum + shapes[i].second); j < pcConv.size() - sum; ++j) 
		{
			Point p = pcConv[j];
			manifolds[i]->pc.row(k) << p.pos[0], p.pos[1], p.pos[2], p.normal[0], p.normal[1], p.normal[2];
			
			// Normalize normal.
			manifolds[i]->pc.row(k).rightCols(3).normalize();
			
			
			//std::cout << manifolds[i]->pc.row(k) << std::endl;
			
			k++;
		}
		sum += shapes[i].second;
	}

	RansacResult res;
	res.manifolds = manifolds;
	res.pc = pc;

	std::cout << "number of manifolds: " << manifolds.size() << std::endl;


	return res;
}


lmu::PointCloud readPointCloudFromStream(std::istream& str)
{
	size_t numRows;
	size_t numCols;
	str >> numRows;
	str >> numCols;

	auto pc = lmu::PointCloud(numRows, numCols);

	for (int i = 0; i < pc.rows(); i++)
	{
		for (int j = 0; j < pc.cols(); j++)
		{
			double v;
			str >> v;
			pc(i, j) = v;
		}
	}
	return pc;
}

void writePointCloudToStream(std::ostream& str, const lmu::PointCloud& pc)
{
	str << pc.rows() << " " << pc.cols() << std::endl;
	for (int i = 0; i < pc.rows(); i++)
	{
		for (int j = 0; j < pc.cols(); j++)
		{
			str << pc.row(i).col(j) << " ";
		}
		str << std::endl;
	}
}

void lmu::writeToFile(const std::string& file, const RansacResult& res)
{
	std::ofstream s(file);

	s.precision(16);

	// Point cloud.
	writePointCloudToStream(s, res.pc);

	// Manifolds.
	s << res.manifolds.size() << std::endl;
	for (const auto& m : res.manifolds)
	{
		s << manifoldTypeToString(m->type) << " "
			<< m->p.x() << " " << m->p.y() << " " << m->p.z() << " " 
			<< m->n.x() << " " << m->n.y() << " " << m->n.z() << " " 
			<< m->r.x() << " " << m->r.y() << " " << m->r.z() << std::endl;

		writePointCloudToStream(s, m->pc);
	}
}

lmu::RansacResult lmu::readFromFile(const std::string& file)
{
	RansacResult res;
	std::ifstream s(file);

	// Point cloud. 	
	res.pc = readPointCloudFromStream(s);

	// Manifolds.
	size_t numManifolds; 
	s >> numManifolds;
	for (int i = 0; i < numManifolds; ++i)
	{
		// Type.
		std::string tstr; 
		s >> tstr;
		ManifoldType t = manifoldTypeFromString(tstr);

		// Parameters.
		Eigen::Vector3d p, n, r;
		s >> p.x() >> p.y() >> p.z() >> n.x() >> n.y() >> n.z() >> r.x() >> r.y() >> r.z();

		// Point cloud.
		auto pc = readPointCloudFromStream(s);

		res.manifolds.push_back(std::make_shared<Manifold>(t, p, n, r, pc));
	}

	return res;
}


std::string lmu::manifoldTypeToString(ManifoldType type)
{	
	switch (type)
	{
	case ManifoldType::Plane:
		return "Plane";
	case ManifoldType::Cylinder:
		return "Cylinder";
	case ManifoldType::Sphere:
		return "Sphere";
	case ManifoldType::Cone:
		return "Cone";
	default:
	case ManifoldType::None:
		return "None";
	}	
}

lmu::ManifoldType lmu::manifoldTypeFromString(std::string type)
{
	std::transform(type.begin(), type.end(), type.begin(), ::tolower);

	if (type == "plane")
		return ManifoldType::Plane;
	else if (type == "cylinder")
		return ManifoldType::Cylinder;
	else if (type == "sphere")
		return ManifoldType::Sphere;
	else if (type == "cone")
		return ManifoldType::Cone;
	else
		return ManifoldType::None;
}

std::ostream& lmu::operator <<(std::ostream& os, const Manifold& m)
{
	os << manifoldTypeToString(m.type) << " n: " << m.n.x() << " " << m.n.y() << " " << m.n.z() << " p: " << m.p.x() << " " << m.p.y() << " " << m.p.z();

	return os;
}

std::ostream& lmu::operator <<(std::ostream& os, const Primitive& p)
{
	os << lmu::primitiveTypeToString(p.type) << " manifolds: " << std::endl;
	for (const auto& m : p.ms)
		os << *m << std::endl;

	return os;
}
