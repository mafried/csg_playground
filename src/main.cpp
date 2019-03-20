#define BOOST_PARAMETER_MAX_ARITY 12

#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>

#include <Eigen/Core>
#include <iostream>
#include <tuple>
#include <chrono>

#include "csgnode.h"
#include "csgnode_helper.h"
#include "primitives.h"
#include "primitive_extraction.h"
#include "pointcloud.h"


void update(igl::opengl::glfw::Viewer& viewer)
{
}

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int mods)
{
	switch (key)
	{
	default:
		return false;
	case '-':
		viewer.core.camera_dnear -= 0.1;
		return true;
	case '+':
		viewer.core.camera_dnear += 0.1;
		return true;
	}
	update(viewer);
	return true;
}

int main(int argc, char *argv[])
{
	using namespace Eigen;
	using namespace std;

	//RUN_TEST(CSGNodeTest);


	igl::opengl::glfw::Viewer viewer;
	viewer.mouse_mode = igl::opengl::glfw::Viewer::MouseMode::Rotation;

	// Initialize
	update(viewer);
		
	try
	{
		//EITHER: Create RANSAC results based on csg tree.		
		double samplingStepSize = 0.2;
		double maxDistance = 0.2;
		double maxAngleDistance = 0.2;
		double noiseSigma = 0.03;
		lmu::CSGNode node = lmu::fromJSONFile("C:/Projekte/csg_playground_build/Debug/ransac.json");
		auto mesh = lmu::computeMesh(node, Eigen::Vector3i(50, 50, 50));
		auto pointCloud = pointCloudFromMesh(mesh, node, maxDistance, samplingStepSize, noiseSigma);		
		//viewer.data().set_mesh(mesh.vertices, mesh.indices);
		//viewer.data().set_points(pointCloud.leftCols(3), pointCloud.rightCols(3));				
		auto ransacRes = lmu::extractManifoldsWithCGALRansac(pointCloud, lmu::RansacParams());
		lmu::writeToFile("ransac_res.txt", ransacRes);
		
		//OR: Read RANSAC results from file.
		//auto ransacRes = lmu::readFromFile("ransac_res.txt");
		
		auto res = lmu::extractPrimitivesWithGA(ransacRes);
		auto primitives = res.primitives;
		auto manifolds = res.manifolds;
	
		for (const auto& p : primitives)
			std::cout << p << std::endl;
		
		//Display result primitives.
				
		std::vector<lmu::CSGNode> childs;
		for (const auto& p : primitives)
			childs.push_back(lmu::geometry(p.imFunc));		
		lmu::CSGNode n = lmu::opUnion(childs);
		Eigen::Vector3d min = ransacRes.pc.leftCols(3).colwise().minCoeff();
		Eigen::Vector3d max = ransacRes.pc.leftCols(3).colwise().maxCoeff();
		auto m = lmu::computeMesh(n, Eigen::Vector3i(20, 20, 20), min, max);

		viewer.data().set_mesh(m.vertices, m.indices);
				
		//Display result manifolds.

		int i = 0;
		for (const auto& m : manifolds)
		{	
			Eigen::Matrix<double, -1, 3> cm(m->pc.rows(),3);

			for (int j = 0; j < cm.rows(); ++j)
			{
				Eigen::Vector3d c;
				switch ((int)(m->type))
				{
				case 0:
					c = Eigen::Vector3d(1, 0, 0);
					break;
				case 1:
					c = Eigen::Vector3d(1, 1, 0);
					break;
				case 2:
					c = Eigen::Vector3d(1, 0, 1);
					break;
				case 3:
					c = Eigen::Vector3d(0, 0, 1);
					break;
				}
				cm.row(j) << c.transpose();
			}

			i++;
			
			viewer.data().add_points(m->pc.leftCols(3), cm);
		}

	}
	catch (const std::exception& ex)
	{
		std::cout << "ERROR: " << ex.what() << std::endl;
	}

	viewer.data().point_size = 5.0;
	viewer.core.background_color = Eigen::Vector4f(1, 1, 1, 1);

	viewer.launch();
	
}