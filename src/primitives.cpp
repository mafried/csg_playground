#include "..\include\primitives.h"
#include "..\include\cluster.h"


#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/Point_with_normal_3.h>
#include <CGAL/property_map.h>
#include <CGAL/Shape_detection_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef K::FT                                           FT;
typedef std::pair<K::Point_3, K::Vector_3>         Point_with_normal;
typedef std::vector<Point_with_normal>                       Pwn_vector;
typedef CGAL::First_of_pair_property_map<Point_with_normal>  Point_map;
typedef CGAL::Second_of_pair_property_map<Point_with_normal> Normal_map;

// In Efficient_RANSAC_traits the basic types, i.e., Point and Vector types
// as well as iterator type and property maps, are defined.
typedef CGAL::Shape_detection_3::Shape_detection_traits<K,
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

lmu::ManifoldType lmu::fromPrimitiveType(PrimitiveType pt)
{
	switch (pt)
	{
		case PrimitiveType::Box:
		case PrimitiveType::Polytope:
			return ManifoldType::Plane;
		case PrimitiveType::Sphere: 
			return ManifoldType::Sphere;
		case PrimitiveType::Cylinder:
			return ManifoldType::Cylinder; 
		case PrimitiveType::Cone:
			return ManifoldType::Cone;
		default:
		case PrimitiveType::None: 
			return ManifoldType::None;
	}
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
	case PrimitiveType::Polytope:
		return "Polytope";
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
		return PrimitiveType::Box;
	else if (type == "polytope")
		return PrimitiveType::Polytope;
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
		pn.first = K::Point_3(p.x(), p.y(), p.z());
		pn.second = K::Vector_3(n.x(), n.y(), n.z());
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
#include <Merge.h>

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

Eigen::Vector3d estimateCylinderPosFromPointCloud(const lmu::Manifold& m)
{
	double min_t = std::numeric_limits<double>::max();
	double max_t = -std::numeric_limits<double>::max();

	for (int i = 0; i < m.pc.rows(); ++i)
	{
		Eigen::Vector3d p = m.pc.row(i).leftCols(3).transpose();
		Eigen::Vector3d a = m.p;
		Eigen::Vector3d ab = m.n;
		Eigen::Vector3d ap = p - a;

		//A + dot(AP, AB) / dot(AB, AB) * AB
		Eigen::Vector3d proj_p = a + ap.dot(ab) / ab.dot(ab) * ab;

		// proj_p = m.p + m.n *t 
		double t = (proj_p.x() - m.p.x()) / m.n.x();

		min_t = t < min_t ? t : min_t;
		max_t = t > max_t ? t : max_t;
	}

	Eigen::Vector3d min_p = m.p + m.n * min_t; 
	Eigen::Vector3d max_p = m.p + m.n * max_t;
	
	return min_p + (max_p - min_p) * 0.5;
}


lmu::RansacResult lmu::extractManifoldsWithOrigRansac(const std::vector<Cluster>& clusters, const lmu::RansacParams& params,
	bool projectPointsOnSurface, int ransacIterations, const lmu::RansacMergeParams& rmParams)
{
	std::vector< std::pair< MiscLib::RefCountPtr< PrimitiveShape >, size_t > > shapes;
	std::vector<::Primitive> primitives;
	std::vector<::PointCloud> pointClouds;


	std::vector<Cluster> filtered_clusters;
	std::copy_if(clusters.begin(), clusters.end(), std::back_inserter(filtered_clusters),
		[&params](const Cluster& c) { return c.pc.rows() > params.min_points; });

	lmu::TimeTicker t;

	for (const auto& cluster : filtered_clusters)
	{
		// Convert point cloud.
		::PointCloud pcConv;
		for (int i = 0; i < cluster.pc.rows(); ++i)
		{
			::Point point(
				Vec3f(cluster.pc.coeff(i, 0), cluster.pc.coeff(i, 1), cluster.pc.coeff(i, 2)),
				Vec3f(cluster.pc.coeff(i, 3), cluster.pc.coeff(i, 4), cluster.pc.coeff(i, 5))
			);

			pcConv.push_back(point);
		}

		// Compute bounding box and set parameters.
		Vec3f min_pt, max_pt;
		compute_bbox(pcConv, min_pt, max_pt);
		pcConv.setBBox(min_pt, max_pt);

		RansacShapeDetector::Options ransacOptions;
		ransacOptions.m_epsilon = params.epsilon * pcConv.getScale();
		ransacOptions.m_bitmapEpsilon = params.cluster_epsilon * pcConv.getScale();
		ransacOptions.m_normalThresh = params.normal_threshold;
		ransacOptions.m_minSupport = std::min((size_t)cluster.pc.rows(), params.min_points);
		ransacOptions.m_probability = params.probability;

		std::set<ManifoldType> types = cluster.manifoldTypes;

		RansacShapeDetector detector(ransacOptions);

		if (types.find(ManifoldType::Cylinder) != types.end())
		{
			detector.Add(new CylinderPrimitiveShapeConstructor());
			std::cout << "Added cylinder detector." << std::endl;
		}
		if (types.find(ManifoldType::Plane) != types.end())
		{
			detector.Add(new PlanePrimitiveShapeConstructor());
			std::cout << "Added plane detector." << std::endl;
		}
		if (types.find(ManifoldType::Sphere) != types.end())
		{
			detector.Add(new SpherePrimitiveShapeConstructor());
			std::cout << "Added sphere detector." << std::endl;
		}
		
		for (int i = 0; i < ransacIterations; ++i)
		{
			std::vector< std::pair< MiscLib::RefCountPtr< PrimitiveShape >, size_t > > perIterShapes;
			std::vector<::Primitive> perIterPrimitives;
			std::vector<::PointCloud> perIterPointClouds;

			size_t remaining = detector.Detect(pcConv, 0, pcConv.size(), &perIterShapes);
			// std::cout << "detection finished " << remaining << std::endl;
			// std::cout << "number of shapes: " << perIterShapes.size() << std::endl;

			// std::cout << "Split" << std::endl;
			SplitPointsPrimitives(perIterShapes, pcConv, perIterPrimitives, perIterPointClouds);

			shapes.insert(shapes.end(), perIterShapes.begin(), perIterShapes.end());
			primitives.insert(primitives.end(), perIterPrimitives.begin(), perIterPrimitives.end());
			pointClouds.insert(pointClouds.end(), perIterPointClouds.begin(), perIterPointClouds.end());
		}
	}
	std::vector<::Primitive> mergedShapes;
	std::vector<::PointCloud> mergedPointclouds;

	t.tick();
	std::cout << "RANSAC: " << t.current << "ms" << std::endl;

	// std::cout << "Merge" << std::endl;
	
	MergeSimilarPrimitives(primitives, pointClouds, 
		rmParams.dist_threshold, rmParams.dot_threshold, rmParams.angle_threshold, mergedShapes, mergedPointclouds);

	//mergedShapes = primitives;
	//mergedPointclouds = pointClouds;

	t.tick();
	std::cout << "MERGE: " << t.current << "ms" << std::endl;

	// Convert to manifolds.
	ManifoldSet manifolds;
	for (auto& shape : mergedShapes)
	{
		lmu::ManifoldPtr m = nullptr; 

		switch (shape->Identifier())
		{
		case 0: // Plane
		{
			auto plane = dynamic_cast<PlanePrimitiveShape*>(shape.Ptr());
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
			auto sphere = dynamic_cast<SpherePrimitiveShape*>(shape.Ptr());
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
			auto cylinder = dynamic_cast<CylinderPrimitiveShape*>(shape.Ptr());
			auto cylinderParams = cylinder->Internal();

			auto p = Eigen::Vector3d(cylinderParams.AxisPosition()[0], cylinderParams.AxisPosition()[1], cylinderParams.AxisPosition()[2]);
		
			m = std::make_shared<lmu::Manifold>(
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
	for (unsigned int i = 0; i < mergedPointclouds.size(); ++i)
	{
		//std::cout << "I: " << i << std::endl;

		manifolds[i]->pc = lmu::PointCloud(mergedPointclouds[i].size(), 6);

		for (unsigned int j = 0; j < mergedPointclouds[i].size(); ++j)
		{
			Point p = mergedPointclouds[i][j];
			manifolds[i]->pc.row(j) << p.pos[0], p.pos[1], p.pos[2], p.normal[0], p.normal[1], p.normal[2];
			
			// Normalize normal.
			manifolds[i]->pc.row(j).rightCols(3).normalize();
		}
	}

	// Correct cylinder position based on the assigned point cloud. 
	for (auto& m : manifolds) {		
		if (m->type == ManifoldType::Cylinder)
			m->p = estimateCylinderPosFromPointCloud(*m);
	}


	if (projectPointsOnSurface) {
		for (auto& m : manifolds) {
			m->projectPointsOnSurface();
		}
	}

	RansacResult res;
	res.manifolds = manifolds;

	std::vector<PointCloud> pcs; 
	std::transform(manifolds.begin(), manifolds.end(), std::back_inserter(pcs), [](const ManifoldPtr& mp) { return mp->pc; });
	res.pc = lmu::mergePointClouds(pcs);

	int i = 0;
	for (auto& m : manifolds)
		m->name = std::to_string(i++);

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
	case ManifoldType::Torus:
		return "Torus";
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
	else if (type == "torus")
		return ManifoldType::Torus;
	else
		return ManifoldType::None;
}

std::ostream& lmu::operator <<(std::ostream& os, const Manifold& m)
{
	os << m.name << " " << manifoldTypeToString(m.type) << " n: " << m.n.x() << " " << m.n.y() << " " << m.n.z() << " p: " << m.p.x() << " " << m.p.y() << " " << m.p.z();

	return os;
}

bool lmu::manifoldsEqual(const Manifold & m1, const Manifold & m2, double epsilon)
{
	return m1.type == m2.type && m1.n.isApprox(m2.n, epsilon) && m1.p.isApprox(m2.p, epsilon);
}

std::ostream& lmu::operator <<(std::ostream& os, const Primitive& p)
{
	os << lmu::primitiveTypeToString(p.type) << " manifolds: " << std::endl;
	for (const auto& m : p.ms)
		os << *m << std::endl;
	os << "Cutout: " << p.cutout << std::endl;

	return os;
}

lmu::PrimitiveSet lmu::PrimitiveSet::without_duplicates() const
{
	PrimitiveSet ps;
	std::unordered_map<size_t, Primitive> p_map;
	for (const auto& p : *this)
		p_map[p.hash(0)] = p;
	for (const auto& kp : p_map)
		ps.push_back(kp.second);

	return ps;
}
