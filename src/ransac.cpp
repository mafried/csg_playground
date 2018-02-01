#include "..\include\ransac.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/Point_with_normal_3.h>
#include <CGAL/property_map.h>
#include <CGAL/Shape_detection_3.h>

#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel  Kernel;
typedef Kernel::FT                                           FT;
typedef std::pair<Kernel::Point_3, Kernel::Vector_3>         Point_with_normal;
typedef std::vector<Point_with_normal>                       Pwn_vector;
typedef CGAL::First_of_pair_property_map<Point_with_normal>  Point_map;
typedef CGAL::Second_of_pair_property_map<Point_with_normal> Normal_map;

// In Efficient_RANSAC_traits the basic types, i.e., Point and Vector types
// as well as iterator type and property maps, are defined.
typedef CGAL::Shape_detection_3::Efficient_RANSAC_traits<Kernel,
	Pwn_vector, Point_map, Normal_map>            Traits;
typedef CGAL::Shape_detection_3::Efficient_RANSAC<Traits> Efficient_ransac;
typedef CGAL::Shape_detection_3::Cone<Traits>             Cone;
typedef CGAL::Shape_detection_3::Cylinder<Traits>         Cylinder;
typedef CGAL::Shape_detection_3::Plane<Traits>            Plane;
typedef CGAL::Shape_detection_3::Sphere<Traits>           Sphere;
typedef CGAL::Shape_detection_3::Torus<Traits>            Torus;

using namespace lmu;

Eigen::MatrixXd getPoints(const CGAL::Shape_detection_3::Shape_base<Traits>& shape, const Pwn_vector& pointsWithNormals)
{
	Eigen::MatrixXd points;
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

	std::cout << "Total points: " << pointsWithNormals.size() << " Assigned points: " << points.rows() << std::endl;

	return points;
}

std::vector<std::shared_ptr<ImplicitFunction>> lmu::ransacWithCGAL(const Eigen::MatrixXd & points, const Eigen::MatrixXd & normals)
{
	std::vector<std::shared_ptr<ImplicitFunction>> res;

	// Add points and normals to the correct structure.
	Pwn_vector pointsWithNormals;
	for (int i = 0; i < points.rows(); i++)
	{
		auto p = points.row(i);
		auto n = normals.row(i);

		Point_with_normal pn;
		pn.first = Kernel::Point_3(p.x(), p.y(), p.z());
		pn.second = Kernel::Vector_3(n.x(), n.y(), n.z());
		pointsWithNormals.push_back(pn);
	}

	Efficient_ransac ransac;
	// Provides the input data.
	ransac.set_input(pointsWithNormals);

	// Register shapes for detection
	//ransac.add_shape_factory<Plane>();
	ransac.add_shape_factory<Sphere>();
	//ransac.add_shape_factory<Cylinder>();
	//ransac.add_shape_factory<Cone>();
	//ransac.add_shape_factory<Torus>();
	
	
	// Sets parameters for shape detection.
	Efficient_ransac::Parameters parameters;
	// Sets probability to miss the largest primitive at each iteration.
	parameters.probability = 0.01;

	// Detect shapes with at least 500 points.
	parameters.min_points = 1000;

	// Sets maximum Euclidean distance between a point and a shape.
	parameters.epsilon = 0.02;

	// Sets maximum Euclidean distance between points to be clustered.
	parameters.cluster_epsilon = 0.1;

	// Sets maximum normal deviation.
	// 0.9 < dot(surface_normal, point_normal); 
	parameters.normal_threshold = 0.9;
	

	ransac.preprocess();

	ransac.detect(parameters);

	// Prints number of detected shapes and unassigned points.
	std::cout << ransac.shapes().end() - ransac.shapes().begin() << " detected shapes, "
		<< ransac.number_of_unassigned_points()
		<< " unassigned points." << std::endl;

	int sphereCount = 0; 
	int cylinderCount = 0; 

	for (const auto& shape : ransac.shapes())
	{	
		// Get specific parameters depending on detected shape.
		if (Sphere* sphere = dynamic_cast<Sphere*>(shape.get()))
		{
			Eigen::Affine3d translate(Eigen::Translation3d(sphere->center().x(), sphere->center().y(), sphere->center().z()));
			auto func = std::make_shared<lmu::IFSphere>(translate, sphere->radius(), 
				lmu::iFTypeToString(ImplicitFunctionType::Sphere) + "_" + std::to_string(sphereCount));			
			func->setPoints(getPoints(*shape, pointsWithNormals));
			
			//std::cout << "Sphere detected. Points: " << func->points().rows() << std::endl;

			res.push_back(func);

			sphereCount++;
		}	
		//else if (Cylinder* cylinder = dynamic_cast<Cylinder*>(shape.get()))
		//{	
			//imFunc = std::make_unique<lmu::IFCylinder>(Eigen::Affine3d::Identity(),cylinder->radius(), 0.0);
			
			//std::cout << "Cylinder detected." << std::endl;

		//	cylinderCount++;
		//}
		else
		{
			//Eigen::Affine3d translate(Eigen::Translation3d(sphere->center().x(), sphere->center().y(), sphere->center().z()));
			auto func = std::make_shared<lmu::IFSphere>(Eigen::Affine3d(Eigen::Translation3d(0,0,0)), 0,
				lmu::iFTypeToString(ImplicitFunctionType::Sphere) + "_" + std::to_string(sphereCount));
			func->setPoints(getPoints(*shape, pointsWithNormals));
			res.push_back(func);
			//std::cout << "Sphere detected. Points: " << func->points().rows() << std::endl;

			res.push_back(func);
		}
	
		// Prints the parameters of the detected shape.
		// This function is available for any type of shape.
		std::cout << shape->info() << std::endl;
	}

	return res;
}

std::vector<std::shared_ptr<ImplicitFunction>> lmu::ransacWithPCL(const Eigen::MatrixXd & points, const Eigen::MatrixXd & normals)
{
	std::vector<std::shared_ptr<ImplicitFunction>> res;

	pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>());

	pc->width = points.rows(); 
	pc->height = 1;
	pc->is_dense = false; 
	pc->points.resize(points.rows());
	for (int i = 0; i < points.rows(); ++i)
	{
		pc->points[i].x = points.row(i).x();
		pc->points[i].y = points.row(i).y();
		pc->points[i].z = points.row(i).z();
	}

	pcl::SampleConsensusModelSphere<pcl::PointXYZ>::Ptr
		sphereModel(new pcl::SampleConsensusModelSphere<pcl::PointXYZ>(pc));


	pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(sphereModel);
	ransac.setDistanceThreshold(.1);
	ransac.setProbability(1.0);
	std::vector<int> inliers;

	int primitiveCounter = 0;

	while(true)
	{	
		inliers.clear();
		
		ransac.computeModel();
		
		ransac.getInliers(inliers);

		if (inliers.empty())
			break;
		
		pcl::PointCloud<pcl::PointXYZ> inlierCloud;
		pcl::copyPointCloud<pcl::PointXYZ>(*pc, inliers, inlierCloud);

		Eigen::VectorXf coeffs;
		ransac.getModelCoefficients(coeffs);

		auto func = std::make_shared<lmu::IFSphere>(Eigen::Affine3d(Eigen::Translation3d(coeffs.x(), coeffs.y(), coeffs.z())), coeffs.w(), "sphere_" + std::to_string(primitiveCounter));

		std::cout << points.rows() << " Inliers: " << inliers.size() << std::endl;
		std::cout << "Coeffs: " << coeffs << std::endl;

		Eigen::MatrixXd m(inlierCloud.points.size(), 6);
		int j = 0;
		for (const auto& point : inlierCloud.points)
		{
			m.row(j)[0] = point.x;// << point.x, point.y, point.z;
			m.row(j)[1] = point.y;
			m.row(j)[2] = point.z;
			j++;
		}
		func->setPoints(m);
		res.push_back(func);

		std::cout << "HERE" << std::endl;

		pcl::PointXYZ specialPoint(std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());

		for (int idx : inliers)
		{
			pc->points[idx] = specialPoint;
		}

		pc->points.erase(remove_if(pc->points.begin(), pc->points.end(), [](pcl::PointXYZ p) { return p.x == std::numeric_limits<float>::max(); }), pc->points.end());

		std::cout << "HERE 2" << std::endl;

		primitiveCounter++;
	} 

	return res;
}
