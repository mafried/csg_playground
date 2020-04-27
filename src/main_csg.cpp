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
lmu::PrimitiveSet g_primitiveSet;
int g_prim_idx = 0;

lmu::Mesh computeMeshFromPrimitives(const lmu::PrimitiveSet& ps, int primitive_idx = -1)
{
	if (ps.empty())
		return lmu::Mesh();

	lmu::PrimitiveSet filtered_ps;
	if (primitive_idx < 0)
		filtered_ps = ps;
	else
		filtered_ps.push_back(ps[primitive_idx]);

	int vRows = 0;
	int iRows = 0;
	for (const auto& p : filtered_ps)
	{
		auto mesh = p.imFunc->createMesh();
		vRows += mesh.vertices.rows();
		iRows += mesh.indices.rows();
	}

	Eigen::MatrixXi indices(iRows, 3);
	Eigen::MatrixXd vertices(vRows, 3);
	int vOffset = 0;
	int iOffset = 0;
	for (const auto& p : filtered_ps)
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

	return lmu::Mesh(vertices, indices);
}

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
	case '4':
		g_prim_idx--;
		if (g_prim_idx < 0)
			g_prim_idx = g_primitiveSet.size() - 1;
		break;
	case '5':
		g_prim_idx++;
		if (g_prim_idx >= g_primitiveSet.size())
			g_prim_idx = 0;
		break;
	case '6':
		g_prim_idx = -1;
		break;
	case '7':
		std::cout << "Serialize meshes" << std::endl;
		std::string basename = "out_mesh";
		for (int i = 0; i < g_primitiveSet.size(); ++i) {
			auto mesh = computeMeshFromPrimitives(g_primitiveSet, i);
			if (!mesh.empty()) {
				std::string mesh_name = basename + std::to_string(i) + ".obj";
				igl::writeOBJ(mesh_name, mesh.vertices, mesh.indices);
			}
		}
		break;
	}

	std::cout << "Manifold Idx: " << g_manifoldIdx << std::endl;
	std::cout << "Primitive Idx: " << g_prim_idx << std::endl;

	std::cout << "Show Result: " << g_show_res << std::endl;

	viewer.data().clear();

	viewer.data().set_points(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0));
	update(viewer);


	if (!g_manifoldSet.empty())
	{
		for (int i = 0; i < g_manifoldSet.size(); ++i)
		{
			//if (i != g_manifoldIdx)
			//{
			//	Eigen::Matrix<double, -1, 3> cm(g_manifoldSet[i]->pc.rows(), 3);
			//	cm.setZero();
			//	viewer.data().add_points(g_manifoldSet[i]->pc.leftCols(3), cm);
			//}
			//else
			//{
			Eigen::Matrix<double, -1, 3> cm(g_manifoldSet[i]->pc.rows(), 3);

			for (int j = 0; j < cm.rows(); ++j)
			{
				Eigen::Vector3d c;
				switch ((int)(g_manifoldSet[i]->type))
				{
				case 4:
					c = Eigen::Vector3d(1, 0, 0);
					break;
				case 0:
					c = Eigen::Vector3d(0, 1, 0);
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
			//}
		}
	}


	auto mesh = computeMeshFromPrimitives(g_primitiveSet, g_prim_idx);
	if (!mesh.empty())
		viewer.data().set_mesh(mesh.vertices, mesh.indices);

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

	bool use_clusters = false;


	std::vector<std::string> models = { "test1", "test2", "test8", "test12", "test15" };
	std::string m = { "test1" };

	ofstream f;
	f.open("ransac_info.txt");

	try
	{		
		f << m << std::endl;

		std::string path = "C:/Projekte/visigrapp2020/data/" + m;
						

		// Primitive estimation based on clusters.
		std::vector<lmu::Cluster> clusters;
		if (use_clusters)
		{
			clusters = lmu::readClusterFromFile(path + "/clusters.txt", 1.0);
		}
		else
		{
			auto pc = lmu::readPointCloud(path + "/pc.txt");
			lmu::Cluster cl(pc, 0, { lmu::ManifoldType::Sphere, lmu::ManifoldType::Plane, lmu::ManifoldType::Cylinder });
			clusters = { cl };
		}

		auto params = lmu::RansacParams();
		params.probability = 0.05;//0.1;
		params.min_points = 200;
		params.normal_threshold = 0.9;
		params.cluster_epsilon = 0.1;// 0.2;
		params.epsilon = 0.01;// 0.2;

		auto ransacRes = lmu::extractManifoldsWithOrigRansac(clusters, params, true, 3, lmu::RansacMergeParams(0.02, 0.9, 0.62831));

		g_manifoldSet = ransacRes.manifolds;

		f << ransacRes.manifolds.size() << " ";

		lmu::TimeTicker t;

		t.tick();
			
		f << std::endl;

		// Farthest point sampling applied to all manifolds.
		for (const auto& m : ransacRes.manifolds)
		{
			m->pc = lmu::farthestPointSampling(m->pc, 100);
			f << std::endl;
		}		

		t.tick();

		std::cout << "FPS: " << t.current << "ms" << std::endl;


		// Extract primitives 
		auto res = lmu::extractPrimitivesWithGA(ransacRes);
		lmu::PrimitiveSet primitives = res.primitives;
		lmu::ManifoldSet manifolds = ransacRes.manifolds;//res.manifolds;

		for (const auto& p : primitives)
			std::cout << "Primitive: " << p << std::endl;
		
		g_primitiveSet = primitives;

		// Extract CSG tree 

		// TODO

		f.close();
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