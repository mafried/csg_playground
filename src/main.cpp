#include <Eigen/Core>
#include <iostream>
#include <tuple>
#include <chrono>

#include "mesh.h"
#include "ransac.h"
#include "pointcloud.h"
#include "csgnode.h"

#include <igl/writeOBJ.h>

int main(int argc, char *argv[])
{
	using namespace Eigen;
	using namespace std;
	using namespace lmu;

	if (argc != 6)
	{
		std::cerr << "Not enough arguments: " << argc << std::endl;
		std::cerr << "Needed arguments: json file, ransac shape distance, sampling, max distance, error sigma." << std::endl;
		return -1;
	}

	try
	{
		std::string jsonFile = std::string(argv[1]);
		int sampling = std::stoi(std::string(argv[2]));

		double ransacShapeDist = std::stod(std::string(argv[3]));
		double maxDistance = std::stod(std::string(argv[4]));
		double errorSigma = std::stod(std::string(argv[5]));

		std::cout << "Input:" << std::endl <<
			"Json file:             " << jsonFile << std::endl <<
			"Sampling:              " << sampling << std::endl <<
			"Ransac shape distance: " << ransacShapeDist << std::endl <<
			"Max distance:          " << maxDistance << std::endl <<
			"Error sigma:           " << errorSigma << std::endl << std::endl;
		
		Eigen::MatrixXd pointCloud;
		std::vector<std::shared_ptr<lmu::ImplicitFunction>> shapes;

		std::cout << "Load CSG tree from Json file...";
		CSGNode node = lmu::fromJson(jsonFile);
		std::cout << "done." << std::endl;

		Mesh mesh = computeMesh(node, Eigen::Vector3i(sampling, sampling, sampling));

		pointCloud = lmu::computePointCloud(node, Eigen::Vector3i(sampling, sampling, sampling), maxDistance, errorSigma);

		writeNode(node, "tree.dot");
		igl::writeOBJ("mesh.obj", mesh.vertices, mesh.indices);
		lmu::writePointCloud("pc.dat", pointCloud);

		lmu::ransacWithSim(pointCloud.leftCols(3), pointCloud.rightCols(3), ransacShapeDist, shapes);

		int i = 0;
		for (const auto& shape : shapes)
		{
			lmu::writePointCloud("pc_" + std::to_string(i) + ".dat", shape->points());
		}		
	}
	catch (const std::exception& ex)
	{
		std::cerr << "Unable to start generator. Reason: " << ex.what() << std::endl;
		return -1;
	}

	return 0;
}
