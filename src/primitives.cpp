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
typedef CGAL::Shape_detection_3::Cone<Traits>             Cone;
typedef CGAL::Shape_detection_3::Cylinder<Traits>         Cylinder;
typedef CGAL::Shape_detection_3::Plane<Traits>            Plane;
typedef CGAL::Shape_detection_3::Sphere<Traits>           Sphere;
typedef CGAL::Shape_detection_3::Torus<Traits>            Torus;

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
		ransac.add_shape_factory<Plane>();
		ransac.add_shape_factory<Sphere>();
		ransac.add_shape_factory<Cylinder>();
	}
	else
	{
		if (params.types.count(ManifoldType::Cylinder))
			ransac.add_shape_factory<Cylinder>();
		if (params.types.count(ManifoldType::Plane))
			ransac.add_shape_factory<Plane>();
		if (params.types.count(ManifoldType::Sphere))
			ransac.add_shape_factory<Sphere>();
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
	parameters.epsilon = params.epsilon;//2.0 / diagLength; //0.02

	// Sets maximum Euclidean distance between points to be clustered.
	parameters.cluster_epsilon = params.cluster_epsilon;//6.0 / diagLength; //0.1

	// Sets maximum normal deviation.
	// 0.9 < dot(surface_normal, point_normal); 
	parameters.normal_threshold = params.normal_threshold;//0.9;

	ransac.preprocess();
	ransac.detect(parameters);

	std::cout << ransac.shapes().end() - ransac.shapes().begin() << " detected shapes, "
		<< ransac.number_of_unassigned_points()
		<< " unassigned points." << std::endl;

	ManifoldSet manifolds;

	for (const auto& shape : ransac.shapes())
	{
		// Get specific parameters depending on detected shape.
		if (Sphere* sphere = dynamic_cast<Sphere*>(shape.get()))
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
		else if (Cylinder* cylinder = dynamic_cast<Cylinder*>(shape.get()))
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
		else if (Plane* plane = dynamic_cast<Plane*>(shape.get()))
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
