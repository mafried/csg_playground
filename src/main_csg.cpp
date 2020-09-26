#define BOOST_PARAMETER_MAX_ARITY 12

#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>

#include <Eigen/Core>
#include <iostream>
#include <tuple>
#include <chrono>

#include "params.h"
#include "csgnode.h"
#include "csgnode_helper.h"
#include "primitives.h"
#include "primitive_extraction.h"
#include "prim_select_ga.h"
#include "pointcloud.h"
#include "cluster.h"

#include "optimizer_red.h"

#include "pc_structure.h"
#include "point_vis.h"
#include "polytope_extraction.h"


lmu::ManifoldSet g_manifoldSet;
int g_manifoldIdx = 0;
lmu::PointCloud g_res_pc;
lmu::PointCloud g_sdf_model_pc;
bool g_show_res = false;
lmu::PrimitiveSet g_primitiveSet;
int g_prim_idx = 0;
bool g_show_sdf = false;

lmu::PrimitiveGaParams g_prim_params;


std::vector<lmu::ConvexCluster> g_convex_clusters;
int g_cluster_idx = 0;

double g_voxel_size = 0.0;
double g_t_inside = 0.0;
double g_t_outside = 0.0;

Eigen::Vector3d HSVtoRGB(int H, double S, double V)
{
	double C = S * V;
	double X = C * (1 - abs(fmod(H / 60.0, 2) - 1));
	double m = V - C;
	double Rs, Gs, Bs;

	if (H >= 0 && H < 60) {
		Rs = C;
		Gs = X;
		Bs = 0;
	}
	else if (H >= 60 && H < 120) {
		Rs = X;
		Gs = C;
		Bs = 0;
	}
	else if (H >= 120 && H < 180) {
		Rs = 0;
		Gs = C;
		Bs = X;
	}
	else if (H >= 180 && H < 240) {
		Rs = 0;
		Gs = X;
		Bs = C;
	}
	else if (H >= 240 && H < 300) {
		Rs = X;
		Gs = 0;
		Bs = C;
	}
	else {
		Rs = C;
		Gs = 0;
		Bs = X;
	}

	return Eigen::Vector3d((Rs + m), (Gs + m), (Bs + m));
}

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
	{
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
	case '9':
		g_cluster_idx++;
		if (g_cluster_idx == g_convex_clusters.size())
			g_cluster_idx = 0;
		break;
	case '0':
		g_cluster_idx--;
		if (g_cluster_idx < 0)
			g_cluster_idx = g_convex_clusters.size()-1;
		break;
	}

	std::cout << "Manifold Idx: " << g_manifoldIdx << std::endl;
	std::cout << "Primitive Idx: " << g_prim_idx << std::endl;

	/*
	lmu::PrimitiveSet ps;
	if (g_primitiveSet.size() > 0)
	{
		ps.push_back(g_primitiveSet[g_prim_idx > 0 ? g_prim_idx : 0]);

		std::vector<Eigen::Matrix<double, 1, 6>> points;
		std::cout << "Primitive score: " << (g_ranker ? g_ranker->get_per_prim_geo_score(ps, points, true)[0] : 0.0) << std::endl;
		points.clear();
		std::cout << "Primitive DH:" << g_ranker->model_sdf->get_dh_type(ps[0], g_t_inside, g_t_outside, g_voxel_size, points, true) << std::endl;
		std::cout << "Show Result: " << g_show_res << std::endl;
		std::cout << "====================" << std::endl;

		viewer.data().clear();

		auto points_pc = lmu::pointCloudFromVector(points);
		viewer.data().set_points(points_pc.leftCols(3), points_pc.rightCols(3));
	}
	*/
	
	//viewer.data().set_points(g_res_pc.leftCols(3), g_res_pc.rightCols(3));
	/*
	auto points_pc = lmu::pointCloudFromVector(points);
	viewer.data().set_points(points_pc.leftCols(3), points_pc.rightCols(3));
	*/
	
	viewer.data().clear();

	viewer.data().show_lines = true;
	viewer.data().add_edges(lmu::g_p1, lmu::g_p2, lmu::g_c);


	if (g_show_sdf)
		viewer.data().set_points(g_res_pc.leftCols(3), g_res_pc.rightCols(3));

	return true;

	if (!g_convex_clusters.empty())
	{

		auto msdf = std::make_shared<lmu::ModelSDF>(g_convex_clusters[g_cluster_idx].pc, g_voxel_size, std::ofstream());

		auto ranker = std::make_shared<lmu::PrimitiveSetRanker>(
			lmu::farthestPointSampling(g_convex_clusters[g_cluster_idx].pc, g_prim_params.num_geo_score_samples),
			g_prim_params.max_dist, 2, g_prim_params.ranker_voxel_size, g_prim_params.allow_cube_cutout, msdf,
			g_prim_params.geo_weight, g_prim_params.per_prim_geo_weight, g_prim_params.size_weight);

		if (g_prim_idx >= 0)
		{
			lmu::PrimitiveSet ps;
			std::vector<Eigen::Matrix<double, 1, 6>> debug_points;

			ps.push_back(g_primitiveSet[g_prim_idx]);

			auto polytope = (lmu::IFPolytope*)g_primitiveSet[g_prim_idx].imFunc.get();
			for (int i = 0; i < polytope->n().size(); ++i)
			{
				std::cout << "n: " << polytope->n()[i].transpose() << " p: " << polytope->p()[i].transpose() << std::endl;
			}


			auto rank = ranker->rank(ps, debug_points);
			std::cout << "POLYTOPE " << g_prim_idx;
			std::cout << rank << std::endl;

			auto debug_pc = lmu::pointCloudFromVector(debug_points);

			viewer.data().add_points(debug_pc.leftCols(3), debug_pc.rightCols(3));
		}
		
		std::cout << "CLUSTER: " << g_cluster_idx << ": " << "Planes: " << g_convex_clusters[g_cluster_idx].planes.size() << std::endl;
		viewer.data().add_points(g_convex_clusters[g_cluster_idx].pc.leftCols(3), g_convex_clusters[g_cluster_idx].pc.rightCols(3));

		auto c = g_convex_clusters[g_cluster_idx].compute_center(*msdf);
		viewer.data().add_points(c.transpose(), Eigen::Vector3d(1, 0, 1).transpose());

		auto mesh = computeMeshFromPrimitives2(g_primitiveSet, g_prim_idx);
		if (!mesh.empty())
		{
			viewer.data().set_mesh(mesh.vertices, mesh.indices);
		}

		//viewer.data().set_mesh(msdf->surface_mesh.vertices, msdf->surface_mesh.indices);

	}

	update(viewer);
	
	//return true;

	/*
	if (!g_manifoldSet.empty())
	{
		viewer.data().clear();

		int cyl_n = 0;
		int sph_n = 0;
		int pla_n = 0;

		for (int i = 0; i < g_manifoldSet.size(); ++i)
		{
			switch (g_manifoldSet[i]->type)
			{
			case lmu::ManifoldType::Cylinder:
				cyl_n++;
				break;
			case lmu::ManifoldType::Plane:
				pla_n++;
				break;
			case lmu::ManifoldType::Sphere:
				sph_n++;
				break;
			}
		}

		int cyl_c = 0;
		int sph_c = 0;
		int pla_c = 0;

		for (int i = 0; i < g_manifoldSet.size(); ++i)
		{
			//if (i == g_manifoldIdx)
			//{
			//	Eigen::Matrix<double, -1, 3> cm(g_manifoldSet[i]->pc.rows(), 3);
			//	cm.setZero();
			//	viewer.data().add_points(g_manifoldSet[i]->pc.leftCols(3), cm);

			//	std::cout << " Type: " << lmu::manifoldTypeToString(g_manifoldSet[g_manifoldIdx]->type) << std::endl;

			//}
			//else
			{
				switch (g_manifoldSet[i]->type)
				{
				case lmu::ManifoldType::Cylinder:
					cyl_c++;
					break;
				case lmu::ManifoldType::Plane:
					pla_c++;
					break;
				case lmu::ManifoldType::Sphere:
					sph_c++;
					break;
				}

				Eigen::Matrix<double, -1, 3> cm(g_manifoldSet[i]->pc.rows(), 3);

			
				for (int j = 0; j < cm.rows(); ++j)
				{
					Eigen::Vector3d c;
					switch ((int)g_manifoldSet[i]->type)
					{
					case 0:
						c = Eigen::Vector3d(0, 1, 0);
						break;
					case 1:
						c = Eigen::Vector3d(1, 0, 0);
						break;
					case 2:
						c = Eigen::Vector3d(.5, .5, .5);
						break;
					case 3:
						c = Eigen::Vector3d(0, 0, 1);
						break;
					case 4:
						c = Eigen::Vector3d(1.0,96.0/255.0, 0);
						break;
					}

					double v = 0.0;
					switch (g_manifoldSet[i]->type)
					{
					case lmu::ManifoldType::Cylinder:
						v = (double)cyl_c / (double)cyl_n;
						c = HSVtoRGB(200, 1.0, v);
						break;
					case lmu::ManifoldType::Plane:
						v = (double)pla_c / (double)pla_n;
						c = HSVtoRGB(22, 1.0, v);
						break;
					case lmu::ManifoldType::Sphere:
						v = (double)sph_c / (double)sph_n;
						c = HSVtoRGB(0, 0.0, v - 0.2);
						break;
					}

					cm.row(j) << c.transpose();
				}


				viewer.data().add_points(g_manifoldSet[i]->pc.leftCols(3), cm);
			}			
		}
	}
	*/	

	/*
	auto mesh = computeMeshFromPrimitives2(g_primitiveSet, g_prim_idx);
	if (!mesh.empty())
	{
		viewer.data().clear();

		viewer.data().set_mesh(mesh.vertices, mesh.indices);

		auto aabb = g_primitiveSet[g_prim_idx >= 0 ? g_prim_idx : 0].imFunc->aabb();

		viewer.data().add_points(aabb.c.transpose(), Eigen::Vector3d(1,0,0).transpose());
		viewer.data().add_points((aabb.c - aabb.s).transpose(), Eigen::Vector3d(1, 0, 0).transpose());
		viewer.data().add_points((aabb.c + aabb.s).transpose(), Eigen::Vector3d(1, 0, 0).transpose());
	}
	*/
	

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

	// Load config
	auto config_file = std::string(argv[1]);
	std::cout << "--------------------------- CONFIG ---------------------------" << std::endl;
	std::cout << "Load config from " << config_file << std::endl;

	lmu::ParameterSet s(config_file);
	
	auto params = lmu::RansacParams();
	params.probability = s.getDouble("Ransac", "Probability", 0.1);//0.1;
	params.min_points = s.getInt("Ransac", "MinPoints", 30);
	params.normal_threshold = s.getDouble("Ransac", "NormalThreshold", 0.9);
	params.cluster_epsilon = s.getDouble("Ransac", "ClusterEpsilon", 0.02);// 0.2;
	params.epsilon = s.getDouble("Ransac", "Epsilon", 0.02);// 0.2;
	auto ransac_iterations = s.getInt("Ransac", "Iterations", 3);

	auto ransac_params = lmu::RansacMergeParams();
	ransac_params.angle_threshold = s.getDouble("Ransac", "Merge.AngleThreshold", 0.6283);
	ransac_params.dist_threshold = s.getDouble("Ransac", "Merge.DistanceThreshold", 0.01);
	ransac_params.dot_threshold = s.getDouble("Ransac", "Merge.DotThreshold", 0.9);

	auto inside_threshold = s.getDouble("Decomposition", "InsideThreshold", 0.9);
	auto outside_threshold = s.getDouble("Decomposition", "OutsideThreshold", 0.1);
	auto voxel_size = s.getDouble("Decomposition", "VoxelSize", 0.01);

	g_voxel_size = voxel_size;
	g_t_inside = inside_threshold;
	g_t_outside = outside_threshold;

	lmu::CSGNodeGenerationParams ng_params;
	ng_params.create_new_prob = s.getDouble("NodeGeneration", "CreateNewProbability", 0.5);
	ng_params.active_prob = s.getDouble("NodeGeneration", "ActiveProbability", 0.5);
	ng_params.dh_type_prob = s.getDouble("NodeGeneration", "DhTypeProbability", 0.5);
	ng_params.evolve_dh_type = s.getBool("NodeGeneration", "EvolveDhType", false);
	ng_params.use_prim_geo_scores_as_active_prob = s.getBool("NodeGeneration", "UsePrimitiveGeoScoresAsActiveProbability", false);
	ng_params.use_all_prims_for_ga = s.getBool("NodeGeneration", "UseAllPrimitivesForGa", false);
	ng_params.max_tree_depth = s.getInt("NodeGeneration", "MaxTreeDepth", 25);
	ng_params.subtree_prob = s.getDouble("NodeGeneration", "SubtreeProbability", 0.5);
	ng_params.creator_strategy = s.getStr("NodeGeneration", "CreatorStrategy", "Selection") == "Node"? lmu::CreatorStrategy::NODE : lmu::CreatorStrategy::SELECTION;

	ng_params.size_weight = s.getDouble("NodeGeneration", "SizeWeight", 0.01);
	ng_params.geo_weight = s.getDouble("NodeGeneration", "GeoWeight", 1.0);
	ng_params.max_iterations = s.getInt("NodeGeneration", "MaxIterations", 100); 
	ng_params.max_count = s.getInt("NodeGeneration", "MaxCount", 10);
	ng_params.cap_plane_adjustment_max_dist = s.getDouble("NodeGeneration", "CapPlaneAdjustmentMaxDistance", 0.0);
	ng_params.use_mesh_refinement = s.getBool("NodeGeneration", "UseMeshRefinement", false);
	ng_params.use_redundancy_removal = s.getBool("NodeGeneration", "UseRedundancyRemoval", false);

	std::string path = s.getStr("Data", "InputFolder", "C:/Projekte/visigrapp2020/data/");
	std::string out_path = s.getStr("Data", "OutputFolder", "");
	std::string source = s.getStr("Data", "Source", "pointcloud");
	std::transform(source.begin(), source.end(), source.begin(), ::tolower);
	double sampling_rate = s.getDouble("Data", "SamplingRate", 0.05);
	bool use_clusters = s.getBool("Data", "UseClusters", true);


	lmu::PrimitiveGaParams prim_params;

	prim_params.cluster_script_folder = s.getStr("Primitives", "ClusterScriptFolder", "C:/Projekte/pointcloud_viewer");

	prim_params.size_weight = s.getDouble("Primitives", "SizeWeight", 0.1);// = 0.1;
	prim_params.geo_weight = s.getDouble("Primitives", "GeoWeight", 0.0);// = 0.0;
	prim_params.per_prim_geo_weight = s.getDouble("Primitives", "PerPrimGeoWeight", 1.0);// = 1.0;//0.1;

	prim_params.num_geo_score_samples = s.getInt("Primitives", "NumGeoScoreSamples", 100);

	prim_params.maxPrimitiveSetSize = s.getInt("Primitives", "MaxPrimitiveSetSize", 75);// = 75;
	prim_params.polytope_prob = s.getDouble("Primitives", "PolytopeProbability", 0.0); // = 0.0;
	prim_params.min_polytope_planes = s.getInt("Primitives", "MinPolytopePlanes", 4); // = 0.0;
	prim_params.max_polytope_planes = s.getInt("Primitives", "MaxPolytopePlanes", 6); // = 0.0;

	prim_params.sdf_voxel_size = s.getDouble("Primitives", "SdfVoxelSize", 0.05);// = 0.05;
	prim_params.ranker_voxel_size = s.getDouble("Primitives", "RankerVoxelSize", 0.05);// = 0.05;
	prim_params.max_dist = s.getDouble("Primitives", "MaxDistance", 0.05);// = 0.05;
	prim_params.allow_cube_cutout = s.getBool("Primitives", "AllowCubeCutout", true);// = true;

	prim_params.max_iterations = s.getInt("Primitives", "MaxIterations", 30); //30
	prim_params.max_count = s.getInt("Primitives", "MaxCount", 30); //30

	prim_params.similarity_filter_epsilon = s.getDouble("Primitives", "SimilarityFilter.Epsilon", 0.0); //0.0
	prim_params.similarity_filter_similarity_only = s.getBool("Primitives", "SimilarityFilter.SimilarityOnly", true);
	prim_params.similarity_filter_perfectness_t = s.getDouble("Primitives", "SimilarityFilter.PerfectnessThreshold", 1.0);
	prim_params.similarity_filter_voxel_size = s.getDouble("Primitives", "SimilarityFilter.VoxelSize", 0.01);

	prim_params.filter_threshold = s.getDouble("Primitives", "GeoScoreFilter.Threshold", 0.01); //0.01

	prim_params.num_elite_injections = s.getInt("Primitives", "NumEliteInjections", 1);

	prim_params.ga_threshold = s.getDouble("Primitives", "GaThreshold", 0.99);
	prim_params.am_quality_threshold = s.getDouble("Primitives", "AmQualityThreshold", 0.9);
	prim_params.am_clustering_param = s.getDouble("Primitives", "AmClusteringParam", 0.4);

	g_prim_params = prim_params;

	s.print();
	std::cout << "--------------------------------------------------------------" << std::endl;

	// Initialize
	update(viewer);

	ofstream res_f, node_ga_f, prim_ga_f;
	res_f.open(out_path + "result.txt");
	node_ga_f.open(out_path + "node_ga.txt");
	prim_ga_f.open(out_path + "prim_ga.txt");

	try
	{							
		// read complete point cloud

		//viewer.data().set_points(pc.leftCols(3), pc.rightCols(3));

		//goto _LAUNCH;


		// Primitive estimation based on clusters.
		std::vector<lmu::Cluster> clusters;
		if (use_clusters)
		{
			clusters = lmu::readClusterFromFile(path + "clusters.txt", 1.0);
			
			for (auto& c : clusters)
			{
				c.manifoldTypes.clear();
				c.manifoldTypes.insert( lmu::ManifoldType::Plane);
			}
		}
		else
		{
			lmu::PointCloud pc;

			std::cout << "Source: " << source << std::endl;

			if (source == "mesh")
			{
				auto mesh = lmu::to_canonical_frame(lmu::fromOBJFile(path + "mesh.obj"));
				pc = lmu::pointCloudFromMesh(mesh, sampling_rate, sampling_rate, 0.0);
				lmu::writePointCloud("out_pc.txt", pc);
			}
			else
			{
				pc = lmu::readPointCloud(path + "pc.txt");
			}

			lmu::Cluster cl(pc, 0, { lmu::ManifoldType::Plane, lmu::ManifoldType::Sphere,lmu::ManifoldType::Cylinder });
			clusters = { cl };
		}
				
		// Scale cluster point clouds to canonical frame defined by complete point cloud.
		std::vector<lmu::PointCloud> cluster_pcs;
		std::transform(clusters.begin(), clusters.end(), std::back_inserter(cluster_pcs),[](const auto& c) { return c.pc; });
		
		
		auto merged_cluster_pc = lmu::mergePointClouds(cluster_pcs);
			
		Eigen::Vector3d mc_min = merged_cluster_pc.leftCols(3).colwise().minCoeff();
		Eigen::Vector3d mc_max = merged_cluster_pc.leftCols(3).colwise().maxCoeff();

		merged_cluster_pc = lmu::to_canonical_frame(merged_cluster_pc);


		std::cout << "Complete point cloud dims: " << lmu::computeAABBDims(merged_cluster_pc).transpose() << std::endl;

		for (auto& c : clusters)
		{
			c.pc = lmu::to_canonical_frame(c.pc, mc_min, mc_max);
		}
		
		// Check if everything went right with the pc transformation.
		cluster_pcs.clear();
		std::transform(clusters.begin(), clusters.end(), std::back_inserter(cluster_pcs), [](const auto& c) { return c.pc; });
		merged_cluster_pc = lmu::mergePointClouds(cluster_pcs);
		
		lmu::TimeTicker t;

		// RANSAC 
		auto ransacRes = lmu::extractManifoldsWithOrigRansac(clusters, params, true, ransac_iterations, ransac_params);

		g_manifoldSet = ransacRes.manifolds;
					
		res_f << "RANSAC Duration=" << t.tick() << std::endl;
		res_f << "Number of Manifolds=" << ransacRes.manifolds.size() << std::endl;
		
		
		std::vector<lmu::PointCloud> pointclouds;
		for (const auto& m : g_manifoldSet)
		{
			pointclouds.push_back(m->pc);
		}
		lmu::writePointCloud("plane_pc.txt", lmu::mergePointClouds(pointclouds));
	

		// Create plane graph.
		auto plane_graph = lmu::create_plane_graph(ransacRes.manifolds, g_res_pc, g_res_pc);
		plane_graph.to_file("plane_graph.gv");

		//goto _LAUNCH;

		//auto mesh_prox = lmu::createFromPointCloud(g_res_pc);
		//igl::writeOBJ("res_mesh_prox_00.obj", mesh_prox.vertices, mesh_prox.indices);

		lmu::resample_proportionally(plane_graph.planes(), 10000);
						
		auto convex_clusters = lmu::get_convex_clusters(plane_graph, prim_params.cluster_script_folder, prim_params.am_clustering_param);

		g_convex_clusters = convex_clusters;
				
		goto _LAUNCH;

		auto polytopes = lmu::generate_polytopes(convex_clusters, plane_graph, prim_params, prim_ga_f);

		polytopes = lmu::merge_polytopes(polytopes, prim_params.am_quality_threshold);

		int i = 0;
		for (const auto& polytope : polytopes)
		{
			igl::writeOBJ("res_mesh_" + std::to_string(i++) + ".obj", polytope.imFunc->meshCRef().vertices, polytope.imFunc->meshCRef().indices);
		}
		
		g_primitiveSet = polytopes;
		
		goto _LAUNCH;

		auto model_sdf = std::make_shared<lmu::ModelSDF>(merged_cluster_pc, prim_params.sdf_voxel_size, res_f);

		// Extract primitives 
		auto non_planar_primitives = lmu::extractNonPlanarPrimitives(ransacRes.manifolds);

		auto res = lmu::extractPolytopePrimitivesWithGA(plane_graph, model_sdf, prim_params, prim_ga_f);

		auto primitives = res.polytopes;
		primitives.insert(primitives.end(), non_planar_primitives.begin(), non_planar_primitives.end());

		res_f << "PrimitiveGA Duration=" << t.tick() << std::endl;

		// Filter primitives
		lmu::ThresholdOutlierDetector od(prim_params.filter_threshold);
		lmu::SimilarityFilter sf(prim_params.similarity_filter_epsilon, prim_params.similarity_filter_voxel_size, prim_params.similarity_filter_similarity_only, 
			prim_params.similarity_filter_perfectness_t);
		
		t.tick();
		primitives = primitives.without_duplicates();
		res_f << "Duplicate Filter=" << t.tick() << std::endl;

		t.tick();
		primitives = od.remove_outliers(primitives, *res.ranker);
		res_f << "Outlier Filter=" << t.tick() << std::endl;

		t.tick();
		primitives = sf.filter(primitives, *res.ranker);
		res_f << "Similarity Filter=" << t.tick() << std::endl;
				
		lmu::ManifoldSet manifolds = ransacRes.manifolds;

		for (const auto& p : primitives)
			std::cout << "Primitive: " << p << std::endl;
		
		g_primitiveSet = primitives;
		g_sdf_model_pc = res.ranker->model_sdf->to_pc();
		g_res_pc = merged_cluster_pc;

		goto _LAUNCH;

		// Extract CSG tree 
		t.tick();
		auto node = lmu::opNo();
		auto decomposition = lmu::decompose_primitives(primitives, *res.ranker->model_sdf, inside_threshold, outside_threshold, voxel_size);
		res_f << "Decomposition Duration=" << t.tick() << std::endl;
		if (decomposition.remaining_primitives.empty() && !ng_params.use_all_prims_for_ga)
		{
			node = decomposition.node;
		}		
		else
		{
			t.tick();
			
			auto gen_res = lmu::generate_csg_node(decomposition, res.ranker, ng_params, node_ga_f);
			node = gen_res.node;
			auto points_pc = lmu::pointCloudFromVector(gen_res.points);
			viewer.data().set_points(points_pc.leftCols(3), points_pc.rightCols(3));

			res_f << "NodeGa Duration=" << t.tick() << std::endl;
		}				

		// Optimize CSG tree
		t.tick();
		lmu::CapOptimizer cap_opt(ng_params.cap_plane_adjustment_max_dist);
		node = cap_opt.optimize_caps(decomposition.get_primitives(true), node);
		res_f << "CapOptimizer Duration=" << t.tick() << std::endl;
		node = lmu::to_binary_tree(node);

		if (ng_params.use_redundancy_removal)
		{
			std::cout << "Num nodes before redundancy removal: " << lmu::numNodes(node) << std::endl;;
			t.tick();
			node = lmu::remove_redundancies(node, 0.01, lmu::PointCloud());
			res_f << "RedundancyRemover Duration=" << t.tick() << std::endl;
			std::cout << "Num nodes after redundancy removal: " << lmu::numNodes(node) << std::endl;;
		}

		lmu::toJSONFile(node, out_path + "tree.json");
		lmu::writeNode(node, out_path + "tree.gv");

		auto pc_n = lmu::computePointCloud(node,lmu::CSGNodeSamplingParams(0.02,0.9, 0.02, 0.02));
		viewer.data().add_points(pc_n.leftCols(3), pc_n.rightCols(3));

		auto m = lmu::computeMesh(node, Eigen::Vector3i(100, 100, 100), Eigen::Vector3d(-1, -1, -1), Eigen::Vector3d(1, 1, 1));
		igl::writeOBJ(out_path + "mesh.obj", m.vertices, m.indices);	
				
		res_f.close();
	}
	catch (const std::exception& ex)
	{
		std::cout << "ERROR: " << ex.what() << std::endl;
	}

_LAUNCH:

	viewer.data().point_size = 5.0;
	viewer.core.background_color = Eigen::Vector4f(1.0, 1.0, 1.0, 1.0);
	viewer.launch();
}