#include <Eigen/Core>
#include <iostream>
#include <tuple>
#include <chrono>

#include "mesh.h"
#include "ransac.h"
#include "pointcloud.h"

#include "csgnode.h"

#include "collision.h"
#include "congraph.h"
#include "csgtree.h"
#include "tests.h"

#include "csgnode_evo.h"
#include "csgnode_helper.h"

#include "csgnode_generator.h"

#include "evolution.h"

#include <igl/writeOBJ.h>

int main(int argc, char *argv[])
{
	using namespace Eigen;
	using namespace std;
	using namespace lmu;


	if (argc != 15)
	{
		std::cerr << "Not enough arguments: " << argc << std::endl;
		
		return -1;
	}

	try
	{
		int samples = std::stoi(std::string(argv[1]));

		double maxDistance = std::stod(std::string(argv[2]));
		double errorSigma = std::stod(std::string(argv[3]));
		int meshSampling = std::stoi(std::string(argv[4]));

		int numSpheres = std::stoi(std::string(argv[5]));
		int numBoxes = std::stoi(std::string(argv[6]));
		int numCylinders = std::stoi(std::string(argv[7]));
		int gridSize = std::stoi(std::string(argv[8]));
		double gridStep = std::stod(std::string(argv[9]));
		double maxObjSize = std::stod(std::string(argv[10]));
		double minObjSize = std::stod(std::string(argv[11]));
		int maxIter = std::stoi(std::string(argv[12]));
		int populationSize = std::stoi(std::string(argv[13]));
		int tournamentNum = std::stoi(std::string(argv[14]));


		std::cout << "Input:" << std::endl <<
			"Samples:              " << samples << std::endl <<
			"Max distance:          " << maxDistance << std::endl <<
			"Error sigma:           " << errorSigma << std::endl <<
			"Mesh sampling:          " << meshSampling << std::endl << std::endl;


		Eigen::MatrixXd pointCloud;
		std::vector<std::shared_ptr<lmu::ImplicitFunction>> shapes;

		GeometrySet set;
		for (int i = 0; i < numSpheres; ++i)
			set.geometries.push_back(Geometry<>(ImplicitFunctionType::Sphere, "Sphere_" + std::to_string(i)));
		for (int i = 0; i < numBoxes; ++i)
			set.geometries.push_back(Geometry<>(ImplicitFunctionType::Box, "Box_" + std::to_string(i)));
		for (int i = 0; i < numCylinders; ++i)
			set.geometries.push_back(Geometry<>(ImplicitFunctionType::Cylinder, "Cylinder_" + std::to_string(i)));
					
		GeometrySet newSet = generateConnectedGeometrySetWithGA(set, Eigen::Vector3i(gridSize, gridSize, gridSize), gridStep, maxObjSize, minObjSize, maxIter, populationSize, tournamentNum);
	
		std::cout << "GeometrySet AVAILABLE" << std::endl;

		auto node = createCSGNodeFromGeometrySet(newSet);

		Mesh mesh = computeMesh(node, Eigen::Vector3i(meshSampling, meshSampling, meshSampling));

		pointCloud = lmu::computePointCloud(node, samples, maxDistance, errorSigma);
		for (const auto& geo : allGeometryNodePtrs(node))
			shapes.push_back(geo->function());

		lmu::writeNode(node, "tree.dot");
		igl::writeOBJ("mesh.obj", mesh.vertices, mesh.indices);
		lmu::writePointCloud("pc.dat", pointCloud);

		lmu::ransacWithSim(pointCloud.leftCols(3), pointCloud.rightCols(3), shapes);


	std:cout << "Number of shapes: " << shapes.size() << std::endl;
		for (const auto& shape : shapes)
		{
			lmu::writePointCloud("pc_" + shape->name() + ".dat", shape->points());
		}
	}
	catch (const std::exception& ex)
	{
		std::cerr << "Unable to start generator. Reason: " << ex.what() << std::endl;
		return -1;
	}

	return 0;
}

/*int main(int argc, char *argv[])
{
	using namespace Eigen;
	using namespace std;
	
	igl::opengl::glfw::Viewer viewer;
	viewer.mouse_mode = igl::opengl::glfw::Viewer::MouseMode::Rotation;

	// Initialize
	update(viewer);

	GeometrySet set; 
	set.geometries = { 
		Geometry<>(ImplicitFunctionType::Sphere, "Sphere1"),
		Geometry<>(ImplicitFunctionType::Sphere, "Sphere2"),
		Geometry<>(ImplicitFunctionType::Sphere, "Sphere3"),
		Geometry<>(ImplicitFunctionType::Sphere, "Sphere4"),
		Geometry<>(ImplicitFunctionType::Box, "Sphere5"),
		Geometry<>(ImplicitFunctionType::Cylinder, "Sphere6"),
		Geometry<>(ImplicitFunctionType::Sphere, "Sphere7"),
		Geometry<>(ImplicitFunctionType::Sphere, "Sphere8"),
		Geometry<>(ImplicitFunctionType::Sphere, "Sphere9"),
		Geometry<>(ImplicitFunctionType::Sphere, "Sphere10"),
		Geometry<>(ImplicitFunctionType::Sphere, "Sphere1"),
		Geometry<>(ImplicitFunctionType::Sphere, "Sphere2"),
		Geometry<>(ImplicitFunctionType::Sphere, "Sphere3"),
		Geometry<>(ImplicitFunctionType::Sphere, "Sphere4"),
		Geometry<>(ImplicitFunctionType::Box, "Sphere5"),
		Geometry<>(ImplicitFunctionType::Cylinder, "Sphere6"),
		Geometry<>(ImplicitFunctionType::Sphere, "Sphere7"),
		Geometry<>(ImplicitFunctionType::Sphere, "Sphere8"),
		Geometry<>(ImplicitFunctionType::Sphere, "Sphere9"),
		Geometry<>(ImplicitFunctionType::Sphere, "Sphere10"),

	};

	GeometrySet newSet = generateConnectedGeometrySetWithGA(set);

	std::cout << "GeometrySet AVAILABLE" << std::endl;

	auto node = createCSGNodeFromGeometrySet(newSet);
	lmu::writeNode(node, "tree.dot");

	std::cout << "NODE AVAILABLE" << std::endl;

	auto mesh = computeMesh(node, Eigen::Vector3i(100, 100, 100));

	viewer.data().set_mesh(mesh.vertices, mesh.indices);

	


	//viewer.core. = true;
	viewer.core.background_color = Eigen::Vector4f(1,1,1,1);
	viewer.data().point_size = 2.0;
	viewer.callback_key_down = &key_down;
	viewer.core.camera_dnear = 0.1;
	viewer.core.lighting_factor = 0;

	
	viewer.launch();
}*/

