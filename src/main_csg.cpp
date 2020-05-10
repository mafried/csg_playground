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
lmu::PointCloud g_sdf_model_pc;
bool g_show_res = false;
lmu::PrimitiveSet g_primitiveSet;
int g_prim_idx = 0;
std::shared_ptr<lmu::PrimitiveSetRanker> g_ranker = nullptr;
bool g_show_sdf = false;

lmu::Mesh computeMeshFromPrimitives2(const lmu::PrimitiveSet& ps, int primitive_idx = -1)
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

	case '8':
		g_show_sdf = !g_show_sdf;
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
			auto mesh = computeMeshFromPrimitives2(g_primitiveSet, i);
			if (!mesh.empty()) {
				std::string mesh_name = basename + std::to_string(i) + ".obj";
				igl::writeOBJ(mesh_name, mesh.vertices, mesh.indices);
			}
		}
		break;
	}

	std::cout << "Manifold Idx: " << g_manifoldIdx << std::endl;
	std::cout << "Primitive Idx: " << g_prim_idx << std::endl;

	lmu::PrimitiveSet ps;
	ps.push_back(g_primitiveSet[g_prim_idx > 0 ? g_prim_idx : 0]);
	std::vector<Eigen::Matrix<double, 1, 6>> points;
	std::cout << "Primitive score: " << g_ranker->get_per_prim_geo_score(ps, points, true)[0] << std::endl;

	std::cout << "Show Result: " << g_show_res << std::endl;

	viewer.data().clear();

	if (g_show_sdf)
		viewer.data().set_points(g_sdf_model_pc.leftCols(3), g_sdf_model_pc.rightCols(3));
	
	//viewer.data().set_points(g_res_pc.leftCols(3), g_res_pc.rightCols(3));

	auto points_pc = lmu::pointCloudFromVector(points);
	viewer.data().add_points(points_pc.leftCols(3), points_pc.rightCols(3));

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

			//viewer.data().add_points(g_manifoldSet[i]->pc.leftCols(3), cm);
			//}
		}
	}


	auto mesh = computeMeshFromPrimitives2(g_primitiveSet, g_prim_idx);
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

	bool use_clusters = true;


	std::vector<std::string> models = { "test1", "test2", "test8", "test12", "test15" };
	std::string m = { "test12" };

	ofstream f;
	f.open("ransac_info.txt");

	try
	{		
		f << m << std::endl;

		std::string path = "C:/Projekte/visigrapp2020/data/" + m;
						
		// read complete point cloud
		auto pc = lmu::readPointCloud(path + "/pc.txt");

		// Primitive estimation based on clusters.
		std::vector<lmu::Cluster> clusters;
		if (use_clusters)
		{
			clusters = lmu::readClusterFromFile(path + "/clusters.txt", 1.0);
		}
		else
		{
			lmu::Cluster cl(pc, 0, { lmu::ManifoldType::Sphere, lmu::ManifoldType::Plane, lmu::ManifoldType::Cylinder });
			clusters = { cl };
		}

		// Scale input pc.
		pc = lmu::to_canonical_frame(pc);

		// Scale cluster point clouds to canonical frame defined by complete point cloud.
		std::vector<lmu::PointCloud> cluster_pcs;
		std::transform(clusters.begin(), clusters.end(), std::back_inserter(cluster_pcs),[](const auto& c) { return c.pc; });
		auto merged_cluster_pc = lmu::mergePointClouds(cluster_pcs);
		for (auto& c : clusters)
		{
			c.pc = lmu::to_canonical_frame(c.pc, &merged_cluster_pc);
		}

		// Check if everything went right with the pc transformation.
		cluster_pcs.clear();
		std::transform(clusters.begin(), clusters.end(), std::back_inserter(cluster_pcs), [](const auto& c) { return c.pc; });
		merged_cluster_pc = lmu::mergePointClouds(cluster_pcs);
		std::cout << "Complete point cloud dims: " << lmu::computeAABBDims(pc).transpose() << std::endl;
		std::cout << "Combined cluster point cloud dims: " << lmu::computeAABBDims(merged_cluster_pc).transpose() << std::endl;

		/*
		auto in_pc = lmu::to_canonical_frame(clusters[0].pc);
		
		double voxel_size = 0.01;
		double distance_epsilon = 0.001;

		auto m = make_shared<lmu::ModelSDF>(in_pc, voxel_size, 0.1);
		auto ms = lmu::ManifoldSet();
		
		auto box_t = Eigen::Affine3d(Eigen::Translation3d(Eigen::Vector3d(0, 0, 0)));
		//box_t.rotate(Eigen::AngleAxis<double>(1.0, Vector3d::UnitX()));
		auto box = std::make_shared<lmu::IFBox>(box_t, Eigen::Vector3d(0.2,0.1,0.1),0, "");
		auto box_prim = lmu::Primitive(box, ms, lmu::PrimitiveType::Box);
		lmu::PrimitiveSet ps; 
		ps.push_back(box_prim);

		lmu::PrimitiveSetRanker ranker(in_pc, ms, ps, distance_epsilon, 16, voxel_size, m);


		std::vector<Eigen::Matrix<double, 1, 6>> points;
		auto scores = ranker.get_per_prim_geo_score(ps, voxel_size / 2.0, distance_epsilon, *m, points);
		std::cout << "Score: " << scores[0] << std::endl;
		
		auto mesh = box->meshCRef();
		viewer.data().set_mesh(mesh.vertices, mesh.indices);
		
		auto out_pc = in_pc;//m->to_pc();
		viewer.data().set_points(out_pc.leftCols(3), out_pc.rightCols(3));
		auto box_pc = lmu::pointCloudFromVector(points);
		viewer.data().add_points(box_pc.leftCols(3), box_pc.rightCols(3));


		goto _LAUNCH;
		*/


		auto params = lmu::RansacParams();
		params.probability = 0.05;//0.1;
		params.min_points = 200;
		params.normal_threshold = 0.9;
		params.cluster_epsilon = 0.1;// 0.2;
		params.epsilon = 0.005;// 0.2;

		auto ransacRes = lmu::extractManifoldsWithOrigRansac(clusters, params, true, 5, lmu::RansacMergeParams(0.02, 0.9, 0.62831));

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
		auto res = lmu::extractPrimitivesWithGA(ransacRes, pc);
		lmu::PrimitiveSet primitives = res.primitives;
		lmu::ManifoldSet manifolds = ransacRes.manifolds;//res.manifolds;

		for (const auto& p : primitives)
			std::cout << "Primitive: " << p << std::endl;
		
		g_primitiveSet = primitives;
		g_ranker = res.ranker;
		g_sdf_model_pc = res.ranker->model_sdf->to_pc();
		//viewer.data().set_points(g_sdf_model_pc.leftCols(3), g_sdf_model_pc.rightCols(3));

		g_res_pc = pc;

		// Extract CSG tree 
		auto node = lmu::generate_tree(res, 0.9, 0.05);

		auto m = lmu::computeMesh(node, Eigen::Vector3i(100, 100, 100), Eigen::Vector3d(-1,-1,-1), Eigen::Vector3d(1,1,1));
		igl::writeOBJ("ex_node.obj", m.vertices, m.indices);
		lmu::toJSONFile(node, "ex_node.json");

		lmu::writeNode(node, "extracted_node.gv");

		f.close();
	}
	catch (const std::exception& ex)
	{
		std::cout << "ERROR: " << ex.what() << std::endl;
	}

_LAUNCH:

	viewer.data().point_size = 5.0;
	viewer.core.background_color = Eigen::Vector4f(0.5, 0.5, 0.5, 0.5);

	viewer.launch();

}