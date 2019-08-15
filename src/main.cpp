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


lmu::ManifoldSet g_manifoldSet; 
int g_manifoldIdx = 0;
lmu::PointCloud g_res_pc;
bool g_show_res = false;

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

	case '1':
		g_manifoldIdx++;
		if (g_manifoldSet.size() <= g_manifoldIdx)
			g_manifoldIdx = 0;
		break;
	
	case '2':
		g_manifoldIdx--;
		if (g_manifoldIdx < 0)
			g_manifoldIdx = g_manifoldSet.empty() ? 0 : g_manifoldSet.size() - 1;
		break;
	case '3':
		g_show_res = !g_show_res;
		break;
	}

	std::cout << "Manifold Idx: " << g_manifoldIdx << std::endl;
	std::cout << "Show Result: " << g_show_res << std::endl;

	viewer.data().set_points(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0));
	update(viewer);

	if (!g_show_res)
	{		
		if (!g_manifoldSet.empty())
		{
			for (int i = 0; i < g_manifoldSet.size(); ++i)
			{
				if (i != g_manifoldIdx)
				{
					Eigen::Matrix<double, -1, 3> cm(g_manifoldSet[i]->pc.rows(), 3);
					cm.setZero();
					viewer.data().add_points(g_manifoldSet[i]->pc.leftCols(3), cm);
				}
				else
				{
					Eigen::Matrix<double, -1, 3> cm(g_manifoldSet[i]->pc.rows(), 3);

					for (int j = 0; j < cm.rows(); ++j)
					{
						Eigen::Vector3d c;
						switch ((int)(g_manifoldSet[i]->type))
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

					viewer.data().add_points(g_manifoldSet[i]->pc.leftCols(3), cm);
				}
			}
		}
	}
	else
	{
		viewer.data().set_points(g_res_pc.leftCols(3), g_res_pc.rightCols(3));
	}

	update(viewer);
	return true;
}

int main(int argc, char *argv[])
{
	using namespace Eigen;
	using namespace std;
	
	igl::opengl::glfw::Viewer viewer;
	viewer.mouse_mode = igl::opengl::glfw::Viewer::MouseMode::Rotation;
	viewer.callback_key_down = &key_down;

	// Initialize
	update(viewer);

	try
	{
		// Primitive estimation based on clusters.

		//auto clusters = lmu::readClusterFromFile("C:/Projekte/labeling-primitives-with-point2net/predict/clusters.txt", 1.0);
		auto clusters = lmu::readClusterFromFile("C:/Users/friedrich/Desktop/test.txt", 1.0);

		//auto clusters = lmu::readClusterFromFile("C:/work/code/csg_playground/seg4csg/data/test.txt", 1.0);
		lmu::TimeTicker t;
		std::vector<lmu::RansacResult> ransacResults;
		for (auto& cluster : clusters)
		{
			//std::cout << "cluster" << std::endl;

			auto params = lmu::RansacParams();
			params.probability = 0.05;//0.1;
			params.min_points = std::min((int)cluster.pc.rows(), 200);// 500);
			params.normal_threshold = 0.9;
			params.cluster_epsilon = 0.1;// 0.2;
			params.epsilon = 0.002;// 0.2;
			params.types = cluster.manifoldTypes;

			std::cout << "CLUSTER PC: " << cluster.pc.rows() << std::endl;
			if (cluster.pc.rows() < params.min_points)
			{
				std::cout << "Not enough points." << std::endl;
				continue;
			}
			ransacResults.push_back(
				lmu::extractManifoldsWithOrigRansac(
					cluster.pc, params, true, 3, lmu::RansacMergeParams(0.02, 0.9, 0.62831)));
			
			// HELPER for analysis - to REMOVE LATER
			//std::cout << "Press a key to continue" << std::endl;
			//char key;
			//std::cin >> key;
			// END OF HELPER


			//auto plane = ransacResults.back().manifolds[0];
			//if(plane->type == lmu::ManifoldType::Plane)
			//	lmu::generateGhostPlanes({plane}, 0.0, 0.0);


			//viewer.data().add_points(cluster.pc.leftCols(3), cluster.pc.rightCols(3));
		}

		std::cout << "Merge RANSAC Results" << std::endl;
		auto ransacRes = lmu::mergeRansacResults(ransacResults);

		g_manifoldSet = ransacRes.manifolds;

		/*lmu::PrimitiveSet prims;
		for (const auto& manifold : ransacRes.manifolds)
		{			
			if (manifold->type == lmu::ManifoldType::Cylinder) {
				lmu::ManifoldSet planes;
				prims.push_back(lmu::createCylinderPrimitive(manifold, planes));
			}
		}

		int vRows = 0;
		int iRows = 0;
		for (const auto& p : prims)
		{
			auto mesh = p.imFunc->createMesh();
			vRows += mesh.vertices.rows();
			iRows += mesh.indices.rows();
		}

		Eigen::MatrixXi indices(iRows, 3);
		Eigen::MatrixXd vertices(vRows, 3);
		int vOffset = 0;
		int iOffset = 0;
		for (const auto& p : prims)
		{
			auto mesh = p.imFunc->createMesh();

			Eigen::MatrixXi newIndices(mesh.indices.rows(), 3);
			newIndices << mesh.indices;

			newIndices.array() += vOffset;

			indices.block(iOffset, 0, mesh.indices.rows(), 3) << newIndices;
			vertices.block(vOffset, 0, mesh.vertices.rows(), 3) << mesh.vertices;

			vOffset += mesh.vertices.rows();
			iOffset += mesh.indices.rows();
		}
		viewer.data().set_mesh(vertices, indices);

		goto _LAUNCH;*/

		// Farthest point sampling applied to all manifolds.
		for (const auto& m : ransacRes.manifolds)
		{
			m->pc = lmu::farthestPointSampling(m->pc, 100);
		}
				
		auto res = lmu::extractPrimitivesWithGA(ransacRes);
		lmu::PrimitiveSet primitives = res.primitives;
		lmu::ManifoldSet manifolds = ransacRes.manifolds;//res.manifolds;
		
		std::vector<lmu::CSGNode> childs;
		std::cout << std::endl;
		for (const auto& p : primitives)
		{
			childs.push_back(p.cutout ? lmu::opComp({ lmu::geometry(p.imFunc) }) : lmu::geometry(p.imFunc));
			std::cout << p << std::endl;
		}

		auto node = lmu::opUnion(childs);
		lmu::CSGNodeSamplingParams p(0.02, 0.01, 0.00, 0.02, Eigen::Vector3d(-1, -1, -1), Eigen::Vector3d(1, 1, 1));
		auto m = lmu::computePointCloud(node, p);
		g_res_pc = m;
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