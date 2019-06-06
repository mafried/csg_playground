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
#include "cluster.h"


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

		double samplingStepSize = 0.1;
		double maxDistance = 0.1;
		double maxAngleDistance = 0.1;
		double noiseSigma = 0.001;

		//	IFPolytope(const Eigen::Affine3d& transform, const std::vector<Eigen::Vector3d>& p, const std::vector<Eigen::Vector3d>& n, const std::string& name) :
	
		auto _p = { Eigen::Vector3d(1,0,0),Eigen::Vector3d(0,1,0), Eigen::Vector3d(0,0,1),
			Eigen::Vector3d(-1,0,0),Eigen::Vector3d(0,-1,0), Eigen::Vector3d(0,0,-1) };

		auto _n = { Eigen::Vector3d(1,0,0),Eigen::Vector3d(0,1,0), Eigen::Vector3d(0,0,1), 
			       Eigen::Vector3d(-1,0,0),Eigen::Vector3d(0,-1,0), Eigen::Vector3d(0,0,-1) };
				
		lmu::CSGNode node = lmu::geo<lmu::IFPolytope>(Eigen::Affine3d::Identity(), _p, _n, "P1");

		//lmu::createPolytope(Eigen::Affine3d::Identity(), _p, _n);
			
		//auto mesh = lmu::computeMesh(node, Eigen::Vector3i(50, 50, 50), Eigen::Vector3d(-2,-2,-2), Eigen::Vector3d(2,2,2));

		lmu::CSGNodeSamplingParams params(maxDistance, maxAngleDistance, noiseSigma, samplingStepSize, Eigen::Vector3d(-1, -1, -1), Eigen::Vector3d(2, 2, 2));
		auto pointCloud = lmu::computePointCloud(node, params);
		std::cout << pointCloud.size();
		viewer.data().set_points(pointCloud.leftCols(3), pointCloud.rightCols(3));

		//auto pointCloud = pointCloudFromMesh(mesh, node, maxDistance, samplingStepSize, noiseSigma);
			
		std::cout << "HERE" << std::endl;

		//auto mesh = lmu::createPolytope(Eigen::Affine3d::Identity(), _p, _n);

		std::cout << "Finished" << std::endl;

		//std::cout << mesh.vertices << std::endl;
		std::cout << "Finished" << std::endl;

		//std::cout << mesh.indices << std::endl;
		std::cout << "Finished" << std::endl;

		//std::cout << mesh.vertices << std::endl;

		//std::cout << "Faces:" << std::endl;

		//std::cout << mesh.indices;

		//viewer.data().set_points(mesh.vertices, Eigen::MatrixXd());

		viewer.data().set_mesh(node.function()->meshRef().vertices, node.function()->meshRef().indices);


		goto _LAUNCH;

		/*double samplingStepSize = 0.2;
		double maxDistance = 0.2;
		double maxAngleDistance = 0.2;
		double noiseSigma = 0.03;
		
		lmu::CSGNode node = lmu::fromJSONFile("C:/Projekte/csg_playground_build/Debug/ransac.json");
		auto mesh = lmu::computeMesh(node, Eigen::Vector3i(50, 50, 50));
		auto pointCloud = pointCloudFromMesh(mesh, node, maxDistance, samplingStepSize, noiseSigma);		
		//viewer.data().set_mesh(mesh.vertices, mesh.indices);
		//viewer.data().set_points(pointCloud.leftCols(3), pointCloud.rightCols(3));

		auto params = lmu::RansacParams();
		params.probability = 0.1;
		params.min_points = 500;
		params.normal_threshold = 0.9; 
		params.cluster_epsilon = 0.2;
		params.epsilon = 0.2;

		auto ransacRes = lmu::extractManifoldsWithCGALRansac(pointCloud, params);
			lmu::writeToFile("ransac_res.txt", ransacRes);

		return 0;
		*/


		auto clusters = lmu::readClusterFromFile("C:/Users/friedrich/PycharmProjects/open3d_test/test.txt", 1.0);
		lmu::TimeTicker t;
		std::vector<lmu::RansacResult> ransacResults; 		
		for (const auto& cluster : clusters)
		{		
			//std::cout << "cluster" << std::endl;
			
			auto params = lmu::RansacParams();
			params.probability = 0.1;
			params.min_points = 500;
			params.normal_threshold = 0.9; 
			params.cluster_epsilon = 0.2;
			params.epsilon = 0.2;
			params.types = { cluster.manifoldType };
						
			ransacResults.push_back(lmu::extractManifoldsWithCGALRansac(cluster.pc, params));
			//viewer.data().add_points(ransacRes.pc.leftCols(3), ransacRes.pc.rightCols(3));
		}
		auto ransacRes = lmu::mergeRansacResults(ransacResults);

		t.tick();
		std::cout << "RANSAC Time: " << t.current << std::endl;


		auto res = lmu::extractPrimitivesWithGA(ransacRes);
		lmu::PrimitiveSet primitives = res.primitives;
		lmu::ManifoldSet manifolds = ransacRes.manifolds;//res.manifolds;

		for (const auto& p : primitives)
			std::cout << p << std::endl;
		
		//Display result primitives.
				
		std::vector<lmu::CSGNode> childs;
		//for (const auto& p : primitives)
		//	childs.push_back(lmu::geometry(p.imFunc));		
		lmu::CSGNode n = lmu::opUnion(childs);
		Eigen::Vector3d min = ransacRes.pc.leftCols(3).colwise().minCoeff();
		Eigen::Vector3d max = ransacRes.pc.leftCols(3).colwise().maxCoeff();
		auto m = lmu::computeMesh(n, Eigen::Vector3i(20, 20, 20), min, max);

		viewer.data().set_mesh(m.vertices, m.indices);

		////EITHER: Create RANSAC results based on csg tree.		
		//double samplingStepSize = 0.2;
		//double maxDistance = 0.2;
		//double maxAngleDistance = 0.2;
		//double noiseSigma = 0.03;
		//
		//lmu::CSGNode node = lmu::fromJSONFile("C:/Projekte/csg_playground_build/Debug/ransac.json");
		//auto mesh = lmu::computeMesh(node, Eigen::Vector3i(50, 50, 50));
		//auto pointCloud = pointCloudFromMesh(mesh, node, maxDistance, samplingStepSize, noiseSigma);		
		////viewer.data().set_mesh(mesh.vertices, mesh.indices);
		////viewer.data().set_points(pointCloud.leftCols(3), pointCloud.rightCols(3));

		//auto params = lmu::RansacParams();
		//params.probability = 0.1;
		//params.min_points = 500;
		//params.normal_threshold = 0.9; 
		//params.cluster_epsilon = 0.2;
		//params.epsilon = 0.2;

		//auto ransacRes = lmu::extractManifoldsWithCGALRansac(pointCloud, params);
		//	lmu::writeToFile("ransac_res.txt", ransacRes);
		//

		////OR: Read RANSAC results from file.
		////auto ransacRes = lmu::readFromFile("ransac_res.txt");
		//
		////auto res = lmu::extractPrimitivesWithGA(ransacRes);
		////lmu::PrimitiveSet primitives = res.primitives;
		//lmu::ManifoldSet manifolds = ransacRes.manifolds;//res.manifolds;
	
		////for (const auto& p : primitives)
		////	std::cout << p << std::endl;
		//
		////Display result primitives.
		//		
		//std::vector<lmu::CSGNode> childs;
		////for (const auto& p : primitives)
		////	childs.push_back(lmu::geometry(p.imFunc));		
		//lmu::CSGNode n = lmu::opUnion(childs);
		//Eigen::Vector3d min = ransacRes.pc.leftCols(3).colwise().minCoeff();
		//Eigen::Vector3d max = ransacRes.pc.leftCols(3).colwise().maxCoeff();
		//auto m = lmu::computeMesh(n, Eigen::Vector3i(20, 20, 20), min, max);

		//viewer.data().set_mesh(m.vertices, m.indices);
		//		
		////Display result manifolds.

		//int i = 0;
		//for (const auto& m : manifolds)
		//{	
		//	Eigen::Matrix<double, -1, 3> cm(m->pc.rows(),3);

		//	for (int j = 0; j < cm.rows(); ++j)
		//	{
		//		Eigen::Vector3d c;
		//		switch ((int)(m->type))
		//		{
		//		case 0:
		//			c = Eigen::Vector3d(1, 0, 0);
		//			break;
		//		case 1:
		//			c = Eigen::Vector3d(1, 1, 0);
		//			break;
		//		case 2:
		//			c = Eigen::Vector3d(1, 0, 1);
		//			break;
		//		case 3:
		//			c = Eigen::Vector3d(0, 0, 1);
		//			break;
		//		}
		//		cm.row(j) << c.transpose();
		//	}

		//	i++;
		//	
		//	viewer.data().add_points(m->pc.leftCols(3), cm);
		//  }

	}
	catch (const std::exception& ex)
	{
		std::cout << "ERROR: " << ex.what() << std::endl;
	}

_LAUNCH:

	viewer.data().point_size = 5.0;
	viewer.core.background_color = Eigen::Vector4f(1, 1, 1, 1);

	viewer.launch();
	
}