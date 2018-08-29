#include "..\include\ransac.h"
#include "..\include\csgnode.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/Point_with_normal_3.h>
#include <CGAL/property_map.h>
#include <CGAL/Shape_detection_3.h>

#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/sample_consensus/msac.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/lmeds.h>
#include <pcl/sample_consensus/prosac.h>
#include <pcl/sample_consensus/mlesac.h>
#include <pcl/sample_consensus/rransac.h>
#include <pcl/sample_consensus/rmsac.h>

#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/sample_consensus/sac_model_normal_sphere.h>
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
	parameters.min_points = 500;

	// Sets maximum Euclidean distance between a point and a shape.
	parameters.epsilon = 0.02; //0.02

	// Sets maximum Euclidean distance between points to be clustered.
	parameters.cluster_epsilon = 0.1; //0.1

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

using Point = pcl::PointXYZ;// pcl::PointXYZRGBNormal;

std::vector<std::shared_ptr<ImplicitFunction>> lmu::ransacWithPCL(const Eigen::MatrixXd & points, const Eigen::MatrixXd & normals)
{
	std::vector<std::shared_ptr<ImplicitFunction>> res;

	pcl::PointCloud<Point>::Ptr pc(new pcl::PointCloud<Point>());

	pc->width = points.rows(); 
	pc->height = 1;
	pc->is_dense = false; 
	pc->points.resize(points.rows());
	for (int i = 0; i < points.rows(); ++i)
	{
		pc->points[i].x = points.row(i).x();
		pc->points[i].y = points.row(i).y();
		pc->points[i].z = points.row(i).z();

		//pc->points[i].normal[0] = normals.row(i).x();
		//pc->points[i].normal[1] = normals.row(i).x();
		//pc->points[i].normal[2] = normals.row(i).x();
	}

	pcl::SampleConsensusModelSphere<Point>::Ptr
		sphereModel(new pcl::SampleConsensusModelSphere<Point>(pc, true));

	

	//pcl::MEstimatorSampleConsensus<pcl::PointXYZ> method(sphereModel);
	 pcl::RandomSampleConsensus<Point> method(sphereModel);
	//pcl::LeastMedianSquares<Point> method(sphereModel);
	//pcl::ProgressiveSampleConsensus< pcl::PointXYZ > method(sphereModel);
	
	//pcl::MaximumLikelihoodSampleConsensus<pcl::PointXYZ >  method(sphereModel);
	//pcl::RandomizedMEstimatorSampleConsensus <pcl::PointXYZ >  method(sphereModel);
	//pcl::RandomizedRandomSampleConsensus <pcl::PointXYZ >  method(sphereModel);

	method.setDistanceThreshold(0.1); //0.1 //1 => alles eins, 0.1 zu viel clutter
	
	method.setProbability(0.8); //0.8
	
	std::vector<int> inliers;

	int primitiveCounter = 0;

	while(method.computeModel())
	{		
		std::cout << "Done Computing Model.";

		inliers.clear();
		method.getInliers(inliers);

		if (inliers.empty())
			break;
		
		pcl::PointCloud<Point> inlierCloud;
pcl::copyPointCloud<Point>(*pc, inliers, inlierCloud);

Eigen::VectorXf coeffs;
method.getModelCoefficients(coeffs);

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

Point specialPoint;
specialPoint.x = std::numeric_limits<float>::max();

for (int idx : inliers)
pc->points[idx] = specialPoint;


pc->points.erase(remove_if(pc->points.begin(), pc->points.end(), [](Point p) { return p.x == std::numeric_limits<float>::max(); }), pc->points.end());

primitiveCounter++;


if (points.rows() == inliers.size())
break;
	}

	return res;
}

void lmu::ransacWithSimMultiplePointOwners(const Eigen::MatrixXd & points, const Eigen::MatrixXd & normals, double maxDelta, const std::vector<std::shared_ptr<ImplicitFunction>>& knownFunctions)
{
	for (auto const& func : knownFunctions)
	{
		std::vector<Eigen::Matrix<double, 1, 6>> pointsAndNormals;

		for (int i = 0; i < points.rows(); ++i)
		{
			Eigen::Vector4d dg = func->signedDistanceAndGradient(points.row(i));

			if (std::abs(dg[0]) <= maxDelta)
			{
				//Eigen::Vector3d g = dg.bottomRows(3);

				//if (normals.row(i).transpose().dot(g) <= 0.0)
				//	continue;

				Eigen::Matrix<double, 1, 6> row;
				row << points.row(i), normals.row(i);
				pointsAndNormals.push_back(row);
			}
		}

		Eigen::MatrixXd points(pointsAndNormals.size(), 6);
		int i = 0;
		for (const auto& row : pointsAndNormals)
			points.row(i++) = row;


		func->setPoints(points);

	}
}

void lmu::ransacWithSim(const Eigen::MatrixXd & points, const Eigen::MatrixXd & normals, double maxDelta, const std::vector<std::shared_ptr<ImplicitFunction>>& knownFunctions)
{

	std::unordered_map<lmu::ImplicitFunctionPtr, std::vector<Eigen::Matrix<double, 1, 6>>> pointsAndNormalsMap;

	for (int i = 0; i < points.rows(); ++i)
	{
		lmu::ImplicitFunctionPtr curFunc = nullptr;
		double curMaxDelta = std::numeric_limits<double>::max();

		for (auto const& func : knownFunctions)
		{
			double d = std::abs(func->signedDistanceAndGradient(points.row(i))[0]);

			if (d <= maxDelta && d < curMaxDelta)
			{
				curMaxDelta = d;
				curFunc = func;
			}
		}

		if (curFunc)
		{
			Eigen::Vector3d p = points.row(i).transpose();
			Eigen::Vector3d n = normals.row(i).rightCols(3).transpose();

			Eigen::Vector3d g = curFunc->signedDistanceAndGradient(p).bottomRows(3);

			//if (n.dot(g) <= 0.0)
			//	continue;

			Eigen::Matrix<double, 1, 6> row;
			row << points.row(i), normals.row(i);

			pointsAndNormalsMap[curFunc].push_back(row);
		}
	}

	for (auto const& func : knownFunctions)
	{
		if (pointsAndNormalsMap.find(func) != pointsAndNormalsMap.end())
		{
			const auto& pointsAndNormals = pointsAndNormalsMap[func];

			Eigen::MatrixXd points(pointsAndNormals.size(), 6);
			int i = 0;
			for (const auto& row : pointsAndNormals)
			{
				points.row(i++) = row;
			}

			func->setPoints(points);
		}		
	}
}
