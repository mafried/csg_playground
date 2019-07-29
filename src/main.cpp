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
		// =========================================================================================================

		// Extraction using RANSAC

		/*double samplingStepSize = 0.3;
		double maxDistance = 0.2;
		double maxAngleDistance = 0.2;
		double noiseSigma = 0.03;
		
		lmu::CSGNode node = lmu::fromJSONFile("C:/Projekte/csg_playground_build/Debug/ransac.json");
		//auto mesh = lmu::computeMesh(node, Eigen::Vector3i(50, 50, 50));
		//auto pointCloud = lmu::readPointCloudXYZ("C:/Users/friedrich/Downloads/RANSAC_TEST/RANSAC_TEST/cms_seg-segmented.xyzn"); 
		auto pointCloud = lmu::computePointCloud(node,lmu::CSGNodeSamplingParams(maxDistance, maxAngleDistance, noiseSigma,samplingStepSize));
		//viewer.data().set_mesh(mesh.vertices, mesh.indices);
	 
		std::cout << "Point cloud size: " << pointCloud.rows() << std::endl;

		auto params = lmu::RansacParams();
		params.probability = 0.001;
		params.min_points = 500;
		params.normal_threshold = 0.9; 
		params.cluster_epsilon = 0.02;
		params.epsilon = 0.01;

		auto ransacRes = lmu::extractManifoldsWithOrigRansac(pointCloud, params, false, 1, lmu::RansacMergeParams(0.01, 0.95, 0.62831));
			lmu::writeToFile("ransac_res.txt", ransacRes);

		for(auto const& m : ransacRes.manifolds)
			viewer.data().add_points(m->pc.leftCols(3), m->pc.rightCols(3));

		//viewer.data().add_points(pointCloud.leftCols(3), pointCloud.rightCols(3));

		goto _LAUNCH;
		*/

		/*double samplingStepSize = 0.05;
		double maxDistance = 0.05;
		double maxAngleDistance = 1;
		double noiseSigma = 0.0;

		auto _p = std::vector<Eigen::Vector3d>({ Eigen::Vector3d(0,0,1),Eigen::Vector3d(0,0,-1), Eigen::Vector3d(0, -1, 0),
			Eigen::Vector3d(0,1,0),Eigen::Vector3d(1,0,0), Eigen::Vector3d(-1,0,0) });

		auto _n = std::vector<Eigen::Vector3d>({ Eigen::Vector3d(0,0,1),Eigen::Vector3d(0,0,-1), Eigen::Vector3d(0, -1, 0),
			Eigen::Vector3d(0,1,1),Eigen::Vector3d(1,0,0), Eigen::Vector3d(-1,0,0) });

		//viewer.data().add_points(pointCloud.leftCols(3), pointCloud.rightCols(3));

		auto poly = std::make_shared<lmu::IFPolytope>(Eigen::Affine3d::Identity(), _p, _n, "P1");
		
		auto pointCloud = lmu::computePointCloud(lmu::geometry(poly), 
			lmu::CSGNodeSamplingParams(maxDistance, maxAngleDistance, noiseSigma, samplingStepSize, Eigen::Vector3d(-1,-1,-1), Eigen::Vector3d(1,1,1)));
		
		viewer.data().add_points(pointCloud.leftCols(3), pointCloud.rightCols(3));

		goto _LAUNCH;*/
		
		
		
		// =========================================================================================================
		
		// Primitive estimation based on clusters.

		auto clusters = lmu::readClusterFromFile("C:/Users/friedrich/PycharmProjects/open3d_test/test.txt", 1.0);
		lmu::TimeTicker t;
		std::vector<lmu::RansacResult> ransacResults; 		
		for (auto& cluster : clusters)
		{		
			//std::cout << "cluster" << std::endl;
			
			auto params = lmu::RansacParams();
			params.probability = 0.1;
			params.min_points = 500;
			params.normal_threshold = 0.9; 
			params.cluster_epsilon = 0.2;
			params.epsilon = 0.2;
			params.types = { cluster.manifoldType };
					
			std::cout << "CLUSTER PC: " << cluster.pc.rows() << std::endl;
			if (cluster.pc.rows() < params.min_points)
			{
				std::cout << "Not enough points." << std::endl;
				continue;
			}
			ransacResults.push_back(
				lmu::extractManifoldsWithOrigRansac(
					cluster.pc, params, true, 1, lmu::RansacMergeParams(0.01, 0.95, 0.62831)));

			//auto plane = ransacResults.back().manifolds[0];
			//if(plane->type == lmu::ManifoldType::Plane)
			//	lmu::generateGhostPlanes({plane}, 0.0, 0.0);

			
			//viewer.data().add_points(cluster.pc.leftCols(3), cluster.pc.rightCols(3));
		}

		std::cout << "Merge RANSAC Results" << std::endl;
		auto ransacRes = lmu::mergeRansacResults(ransacResults);
	
		std::cout << "Manifolds: " << ransacRes.manifolds.size() << std::endl;
		for (const auto& m : ransacRes.manifolds)
		{
			std::cout << manifoldTypeToString(m->type) << std::endl;

			m->pc = lmu::farthestPointSampling(m->pc, 50);
			//viewer.data().add_points(m->pc.leftCols(3), m->pc.rightCols(3));
		}

		t.tick();
		std::cout << "RANSAC Time: " << t.current << std::endl;

		//goto _LAUNCH;

		auto res = lmu::extractPrimitivesWithGA(ransacRes);
		lmu::PrimitiveSet primitives = res.primitives;
		lmu::ManifoldSet manifolds = ransacRes.manifolds;//res.manifolds;

		for (const auto& p : primitives)
		{
			std::cout << p << std::endl;
			//viewer.data().add_points(p.imFunc->meshCRef().vertices, p.imFunc->meshCRef().vertices);
			//viewer.data().set_mesh(p.imFunc->meshCRef().vertices, p.imFunc->meshCRef().indices);

		}

		//Display result primitives.
			
		int vRows = 0; 
		int iRows = 0; 
		std::vector<lmu::CSGNode> childs;
		for (const auto& p : primitives)
		{
			//if (p.type == lmu::PrimitiveType::Box)
			//	continue;

			auto mesh = p.imFunc->createMesh();

			childs.push_back(lmu::geometry(p.imFunc));

			vRows += mesh.vertices.rows();
			iRows += mesh.indices.rows();
		}

		Eigen::MatrixXi indices(iRows, 3);
		Eigen::MatrixXd vertices(vRows, 3);
		int vOffset = 0; 
		int iOffset = 0;
		for (const auto& p : primitives)
		{
			//if (p.type == lmu::PrimitiveType::Box)
			//	continue;

			auto mesh = p.imFunc->createMesh();

			Eigen::MatrixXi newIndices(mesh.indices.rows(), 3);
			newIndices << mesh.indices;

			newIndices.array() += vOffset;

			indices.block(iOffset,0, mesh.indices.rows(),3) << newIndices;
			vertices.block(vOffset, 0, mesh.vertices.rows(), 3) << mesh.vertices;
		
			vOffset += mesh.vertices.rows();
			iOffset += mesh.indices.rows();
		}
		//viewer.data().set_mesh(vertices, indices);
 
		//Eigen::Vector3d min = Eigen::Vector3d(-2, -2, -2);
		//Eigen::Vector3d max = Eigen::Vector3d(2, 2, 2);

		auto node = lmu::opUnion(childs);
		lmu::CSGNodeSamplingParams p(0.02, 0.02, 0.00, 0.02, Eigen::Vector3d(-1,-1,-1), Eigen::Vector3d(1,1,1));
		auto m = lmu::computePointCloud(node, p);
		viewer.data().set_points(m.leftCols(3), m.rightCols(3));

		//auto m = lmu::computeMesh(node, Eigen::Vector3i(50, 50, 50));
		//viewer.data().set_mesh(m.vertices, m.indices);

		

		/*auto _p = std::vector<Eigen::Vector3d>({ Eigen::Vector3d(-0.0700974, -0.0241359, 0.682183),Eigen::Vector3d(-0.00842728, -0.0110942, -0.281548), Eigen::Vector3d(0.00995219, -0.47185, -0.00839383),
			Eigen::Vector3d(0.00245973, 0.287677, 0.00328013),Eigen::Vector3d(0.151568, -0.0039502, -0.00301725), Eigen::Vector3d(-0.457327, 0.00124607, 0.065478) });

		//auto _p = std::vector<Eigen::Vector3d>({ Eigen::Vector3d(0, 0, 0.682183),Eigen::Vector3d(0,0,-0.281548), Eigen::Vector3d(0, -0.47185, 0),
		//	Eigen::Vector3d(0, 0.287677, 0),Eigen::Vector3d(0.151568, 0, 0), Eigen::Vector3d(-0.457327, 0, 0) });

		auto _n = std::vector<Eigen::Vector3d>({ Eigen::Vector3d(-0.102153, -0.0351733, 0.994147),Eigen::Vector3d(-0.0298953, -0.0393559, -0.998778), Eigen::Vector3d(0.0210838, -0.99962, -0.0177824),
			Eigen::Vector3d(-0.00854944, -0.999898, -0.01140),Eigen::Vector3d(-0.999463, 0.0260481, 0.0198961), Eigen::Vector3d(0.989902, -0.00269717, -0.14173) });

		//auto _n = std::vector<Eigen::Vector3d>({ Eigen::Vector3d(0,0,1),Eigen::Vector3d(0,0,-1), Eigen::Vector3d(0, -1, 0),
		//	Eigen::Vector3d(0,1,0),Eigen::Vector3d(1,0,0), Eigen::Vector3d(-1,0,0) });

		//_p = _n; 

		auto poly = std::make_shared<lmu::IFPolytope>(Eigen::Affine3d::Identity(), _p, _n, "P1");
		//	Plane n : -0.102153 - 0.0351733 0.994147 p : -0.0700974 - 0.0241359 0.682183
		//	Plane n : -0.0298953 - 0.0393559 - 0.998778 p : -0.00842728 - 0.0110942 - 0.281548
		//	Plane n : 0.0210838 - 0.99962 - 0.0177824 p : 0.00995219 - 0.47185 - 0.00839383
		//	Plane n : -0.00854944 - 0.999898 - 0.011401 p : 0.00245973 0.287677 0.00328013
		//	Plane n : -0.999463 0.0260481 0.0198961 p : 0.151568 - 0.0039502 - 0.00301725
		//	Plane n : 0.989902 - 0.00269717 - 0.14173 p : -0.457327 0.00124607 0.065478

		viewer.data().set_mesh(poly->meshCRef().vertices, poly->meshCRef().indices);
		
		//viewer.data().set_points(ransacRes.pc.leftCols(3), ransacRes.pc.rightCols(3));
		std::cout << "P: " << _p.size() << std::endl;
		
		for (int i = 0; i < _p.size(); ++i)
			viewer.data().add_points(_p[i].transpose(), _n[i].transpose());

		viewer.data().add_points(poly->meshCRef().vertices, poly->meshCRef().vertices);
		*/



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