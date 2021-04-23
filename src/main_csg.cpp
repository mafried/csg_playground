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
#include "cit.h"


void update(igl::opengl::glfw::Viewer& viewer)
{
}

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int mods)
{
	switch (key)
	{
	default:
		return false;	
	}

	update(viewer);
	return true;
}


struct Timings
{
	Timings() :
		model_sdf(0),
		ga(0),
		articulation_points(0),
		connection_graph(0),
		connected_components(0),
		decomposition(0),
		model_sdf_adaptations(0)
	{
	}

	std::string to_string() const
	{
		std::stringstream ss;
		ss << "{ ";
		ss << "'model_sdf': " << model_sdf << ", ";
		ss << "'ga': " << ga << ", ";
		ss << "'articulation_points': " << articulation_points << ", ";
		ss << "'connection_graph': " << connection_graph << ", ";
		ss << "'connected_components': " << connected_components << ", ";
		ss << "'decomposition': " << decomposition << ", ";
		ss << "'model_sdf_adaptations': " << model_sdf_adaptations;
		ss << " }";

		return ss.str();
	}

	int model_sdf;
	int model_sdf_adaptations;
	int ga;
	int articulation_points;
	int connection_graph;
	int connected_components;
	int decomposition;
};

struct Stats
{
	Stats() : 
		num_ga_calls(0),
		num_inside_dhs(0),
		num_outside_dhs(0),
		num_aps(0),
		num_neighbor_aps(0),
		num_initial_components(0),
		num_pruned(0)
	{
	}

	std::string to_string() const
	{
		std::stringstream ss;
		ss << "{ ";
		ss << "'num_ga_calls': " << num_ga_calls << ", ";
		ss << "'num_inside_dhs': " << num_inside_dhs << ", ";
		ss << "'num_outside_dhs': " << num_outside_dhs << ", ";
		ss << "'num_aps': " << num_aps << ", ";
		ss << "'num_neighbor_aps': " << num_neighbor_aps << ", ";
		ss << "'num_pruned': " << num_pruned << ", ";
		ss << "'num_initial_components': " << num_initial_components;
		ss << " }";

		return ss.str();
	}

	int num_ga_calls; 
	int num_inside_dhs; 
	int num_outside_dhs;
	int num_aps;
	int num_neighbor_aps;
	int num_initial_components;
	int num_pruned;
};

std::string to_list_str(const std::vector<lmu::ImplicitFunctionPtr>& funcs);

lmu::CSGNode decompose(const std::vector<lmu::Graph>& graphs, const std::shared_ptr<lmu::ModelSDF>& model_sdf,
	const lmu::CSGNodeGenerationParams& ng_params, const lmu::CSGNode& gt_node, const std::string& output_path, Timings& timings, Stats& stats, int rec_level = 0);

bool is_inside_dh(const lmu::ImplicitFunctionPtr& prim, const lmu::ModelSDF& model_sdf)
{
	int inside_points = 0;
	int points = 0;

	lmu::iterate_over_prim_volume(lmu::Primitive(prim, lmu::ManifoldSet(), lmu::PrimitiveType::None), model_sdf.voxel_size,
		[&inside_points, &points, &model_sdf, &prim ](const auto& p) 
		{
			if (prim->signedDistance(p) <= 0.0)
			{
				if (model_sdf.distance(p) <= 0.0)
					inside_points++;

				points++;
			}
		});

	double inside_ratio = (double)(inside_points) / (double)(points);

	std::cout << "INSIDE " << prim->name() << ": " << inside_ratio << " points: " << points << std::endl;

	return inside_ratio > 0.5;
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
	
	auto input_node_file = s.getStr("Data", "InputNodeFile", "input.json");
	auto input_mesh_folder = s.getStr("Data", "InputMeshFolder", "");
	auto num_input_meshes = s.getInt("Data", "NumInputMeshes", 0);
	auto model_to_change = s.getStr("Data", "ModelToChange", "");

	auto input_pc_file = s.getStr("Data", "InputPointCloudFile", "");
	auto sample_cell_size = s.getDouble("Data", "SampleCellSize", 0.1);
	auto do_decomposition = s.getBool("NodeGeneration", "UseDecomposition", true);
	auto out_path = s.getStr("Data", "OutputFolder", "\\");

	auto num_cells = s.getInt("ConnectionGraph", "NumCells", 100);

	lmu::CSGNodeGenerationParams ng_params;

	ng_params.create_new_prob = s.getDouble("NodeGeneration", "CreateNewProbability", 0.5);
	ng_params.max_tree_depth = s.getInt("NodeGeneration", "MaxTreeDepth", 25);
	ng_params.subtree_prob = s.getDouble("NodeGeneration", "SubtreeProbability", 0.5);
	ng_params.size_weight = s.getDouble("NodeGeneration", "SizeWeight", 0.01);
	ng_params.geo_weight = s.getDouble("NodeGeneration", "GeoWeight", 1.0);

	ng_params.max_iterations = s.getInt("NodeGeneration", "MaxIterations", 100);
	ng_params.max_count = s.getInt("NodeGeneration", "MaxCount", 50);
	ng_params.dh_type_prob = s.getDouble("NodeGeneration", "DHTypeProbability", 0.5);
	ng_params.node_creator_prob = s.getDouble("NodeGeneration", "NodeCreatorProbability", 0.5);
	ng_params.max_budget = s.getInt("NodeGeneration", "MaxBudget", 10);
	
	ng_params.node_ratio = s.getDouble("NodeGeneration", "NodeRatio", 0.7);

	s.print();
	std::cout << "--------------------------------------------------------------" << std::endl;

	// Initialize
	update(viewer);

	ofstream res_f;
	res_f.open(out_path + "result.txt");
		
	//try
	{
		// Load input node for primitive parameters. 

		auto node = lmu::opNo();
		
		if (num_input_meshes == 0)
		{
			std::cout << "load node from " << input_node_file << std::endl;
			node = lmu::fromJSONFile(input_node_file);
		}
		else
		{
			lmu::initializePolytopeCreator();

			node = lmu::opUnion();
			for (int i = 0; i < num_input_meshes; ++i)
			{
				auto mesh = lmu::fromOBJFile(input_mesh_folder + "res_mesh_" + std::to_string(i)+".obj");
				auto polytope = lmu::createPolytope(mesh, "Polytope_" + std::to_string(i));
				if (polytope)
				{
					node.addChild(lmu::geometry(polytope));
				}
				else
				{
					std::cout << "ERROR: Could not create polytope" << std::endl;
				}
			}
		}
		node = lmu::to_binary_tree(node);

	

		lmu::writeNode(node, out_path + "input_node.gv");

		auto aabb = lmu::aabb_from_node(node);

		res_f << "GraphNodes= " << lmu::numNodes(node) << std::endl;

		Eigen::Vector3d min = aabb.c - aabb.s;
		Eigen::Vector3d max = aabb.c + aabb.s;
		double cell_size = (max.maxCoeff() - min.minCoeff()) / (double)num_cells;
		std::cout << "node aabb: (" << min.transpose() << ")(" << max.transpose() << ")" << std::endl;
		std::cout << "cell size: " << cell_size << std::endl;
		
		lmu::Mesh m = lmu::computeMesh(node, Eigen::Vector3i(50,50,50), min, max);
		igl::writeOBJ(out_path + "node.obj", m.vertices, m.indices);

		// Load point cloud. 

		auto pc = lmu::PointCloud();
		if (!input_pc_file.empty())
		{
			std::cout << "load pc from " << input_pc_file << std::endl;
			pc = lmu::readPointCloudXYZ(input_pc_file);
			pc = lmu::to_frame(pc, min, max);
		}
		else
		{
			std::cout << "pc file is not available. Try to sample node." << std::endl;

			lmu::CSGNodeSamplingParams sampling_params(sample_cell_size * 0.3, 0.0, sample_cell_size * 0.5, sample_cell_size, min, max);
			pc = computePointCloud(node, sampling_params);
		}
		res_f << "Points= " << pc.rows() << std::endl;
		lmu::writePointCloudXYZ(out_path + "pc.xyz", pc);

		//pc = lmu::farthestPointSampling(pc, 32000);
		//res_f << "PointsAfterSampling: " << pc.rows() << std::endl;

		Timings timings;
		Stats stats;
		lmu::TimeTicker t;

		// Create connection graph.

		auto prims = lmu::allDistinctFunctions(node);

		t.tick();
		lmu::Graph graph;
		std::ifstream gs(input_node_file + ".col");
		if (gs.is_open())
		{
			std::cout << "Read graph from file from " << (input_node_file + ".col") << std::endl;

			graph = lmu::createConnectionGraph(gs, prims);
		}
		else
		{
			std::cout << "Could not find connection graph file. Create graph via sampling." << std::endl;
			graph = lmu::createConnectionGraph(prims, cell_size);
		}
		timings.connection_graph += t.tick();
		lmu::writeConnectionGraph(out_path + "input_graph.gv", graph);

		// Create discrete model sdf. 

		std::cout << "create discrete model sdf. " << std::endl;
		t.tick();
		auto model_sdf = std::make_shared<lmu::ModelSDF>(pc, sample_cell_size, res_f);
		timings.model_sdf += t.tick();
		igl::writeOBJ(out_path + "sdf_model.obj", model_sdf->surface_mesh.vertices, model_sdf->surface_mesh.indices);

		auto mesh_grid = model_sdf->to_mesh();
		igl::writeOBJ(out_path + "sdf_model_grid.obj", mesh_grid.vertices, mesh_grid.indices);



		t.tick();
		auto initial_components = lmu::getConnectedComponents(graph);
		timings.connected_components += t.tick();

		stats.num_initial_components = initial_components.size();

		t.tick();
		auto result_node = lmu::opNo();

		if (do_decomposition)
		{
			result_node = decompose(initial_components, model_sdf, ng_params, node, out_path, timings, stats);
		}
		else
		{
			std::ofstream res_ga_1, res_ga_2;
			res_ga_1.open(out_path + "result_ga_" + std::to_string(stats.num_ga_calls) + "_1.txt");
			res_ga_2.open(out_path + "result_ga_" + std::to_string(stats.num_ga_calls) + "_2.txt");

			auto primitives = lmu::allDistinctFunctions(node);
			std::cout << "Generate node for primitives " << to_list_str(primitives) << std::endl;

			t.tick();
			auto res = lmu::generate_csg_node(primitives, model_sdf, ng_params, res_ga_1, res_ga_2, node);
			result_node = res.node;
			timings.ga += t.tick();

			res_ga_1.close();
			res_ga_2.close();

		}

		result_node = lmu::to_binary_tree(result_node);

		timings.decomposition += t.tick();

		auto cits = lmu::generate_cits(*model_sdf, lmu::allDistinctFunctions(node), model_sdf->voxel_size, true);
		lmu::SelectionRanker ranker(std::get<0>(cits), std::get<1>(cits));

		auto final_rank = ranker.rank(lmu::PrimitiveSelection(result_node));

		res_f << "FinalRank= " << final_rank << std::endl;
		res_f << "InputNodes= " << lmu::numNodes(node) << std::endl;
		res_f << "InputNodesWithoutLeaves= " << lmu::numNodes(node, true) << std::endl;

		res_f << "OutputNodes= " << lmu::numNodes(result_node) << std::endl;
		res_f << "OutputNodesWithoutLeaves= " << lmu::numNodes(result_node, true) << std::endl;

		auto r_gt = ranker.rank(lmu::PrimitiveSelection(node));
		res_f << "GroundTruthRank= " << r_gt << std::endl;


		if (!model_to_change.empty())
		{
			std::vector<std::string> experiments = { "0_ga_size_dec", "1_ga_size_dec", "2_ga_size_dec", "0_ga_size", "1_ga_size", "2_ga_size" };

			auto r_gt = ranker.rank(lmu::PrimitiveSelection(node));

			for (int i = 0; i < experiments.size(); ++i)
			{
				std::ofstream outfile("C:/Projekte/dissertation_csg_synth/output/" + experiments[i] + "/" + model_to_change + "/result.txt", std::ios_base::app);
				auto n = lmu::fromJSONFile("C:/Projekte/dissertation_csg_synth/output/" + experiments[i] + "/" + model_to_change + "/result_node.json");
				auto r = ranker.rank(lmu::PrimitiveSelection(n));

				outfile << "CommonFinalRank= " << r << std::endl;
				outfile << "GroundTruthRank= " << r_gt << std::endl;
			}
		}
		


		lmu::writeNode(result_node, out_path + "result_node.gv");
		lmu::toJSONFile(result_node, out_path + "result_node.json");
		
		lmu::Mesh result_mesh = lmu::computeMesh(result_node, Eigen::Vector3i(100,100,100), min, max);
		igl::writeOBJ(out_path + "result_node.obj", result_mesh.vertices, result_mesh.indices);

		res_f << "Timings= " << timings.to_string() << std::endl;
		res_f << "Stats= " << stats.to_string() << std::endl;
		res_f << "NumPrims= " << lmu::allDistinctFunctions(node).size() << std::endl;


		std::cout << "Close file" << std::endl;
		res_f.close();
	}
	//catch (const std::exception& ex)
	//{
	//	std::cout << "ERROR: " << ex.what() << std::endl;
	//}

_LAUNCH:

	return 0;
}

std::string to_list_str(const std::vector<lmu::ImplicitFunctionPtr>& funcs)
{
	std::stringstream str;
	for (const auto& p : funcs)
		str << p->name() << " ";
	return str.str();
}

std::string to_list_str(const std::vector<size_t>& vertices, const lmu::Graph& g)
{
	std::stringstream str;
	for (const auto& v : vertices)
		str << g.structure[v]->name() << " ";
	return str.str();
}

lmu::CSGNode decompose(const std::vector<lmu::Graph>& graphs, const std::shared_ptr<lmu::ModelSDF>& model_sdf, const lmu::CSGNodeGenerationParams& ng_params, 
	const lmu::CSGNode& gt_node, const std::string& output_path, Timings& timings, Stats& stats, int rec_level)
{	
	
	lmu::TimeTicker t;

	auto node = lmu::opUnion();

	int iter = 0;
	for (const auto& g : graphs)
	{
		iter++;

		std::vector<lmu::ImplicitFunctionPtr> dh_in;
		std::vector<lmu::ImplicitFunctionPtr> dh_out;
		auto per_graph_node = lmu::opUnion();

		// If component contains only a single element, it must be a dominant halfspace fully inside the volume.
		if (lmu::numVertices(g) == 1)
		{
			dh_in.push_back(getImplicitFunctions(g)[0]);
		}
		else
		{
			// Prune component.
			auto g_pruned = lmu::pruneGraph(g);

			// Check if pruned primitives are in- or out-dhs.
			auto pruned_primitives = lmu::get_pruned_primitives(g, g_pruned);
			for (const auto& pp : pruned_primitives)
			{
				if (is_inside_dh(pp, *model_sdf))
				{
					dh_in.push_back(pp);
				}
				else
				{
					dh_out.push_back(pp);
				}
			}
			bool something_was_pruned = !pruned_primitives.empty();

			if (something_was_pruned)
			{
				std::cout << "Level " << rec_level << " Primitives were pruned: " << to_list_str(pruned_primitives) << std::endl;
				
				lmu::writeConnectionGraph(output_path + "pruned_graph_" + std::to_string(iter - 1) + "_" + std::to_string(rec_level) + ".gv", g_pruned);
				stats.num_pruned += pruned_primitives.size();
			}

			// Remove outside DHs from model sdf. 
			t.tick();
			auto model_sdf_wo_out_dhs = something_was_pruned ? model_sdf->create_with_union(dh_out) : model_sdf;
			timings.model_sdf_adaptations += t.tick();

			// Get articulation points.
			t.tick();
			auto aps = lmu::get_articulation_points(g);
			timings.articulation_points += t.tick();
			stats.num_aps += aps.size();
			std::cout << "APS: " << to_list_str(aps, g) << std::endl;

			// Select articulation points surrounded only by other articulation points.
			auto aps_with_neighbors = lmu::select_aps_with_aps_as_neighbors(aps, g);
			stats.num_neighbor_aps += aps_with_neighbors.size();
			std::cout << "APS with Neighbors:" << to_list_str(aps_with_neighbors, g) << std::endl;
			
			// Remove selected articulation points from pruned graph and add them to the list of in-dhs.
			
			auto g_rem_art = lmu::filterGraph(g_pruned, [&g, &g_pruned, &aps_with_neighbors, &dh_in](const auto& v)
			{
				for (const auto& ap : aps_with_neighbors)
				{					
					if (g.structure[ap] == g_pruned.structure[v])
					{
						std::cout << "Remove primitive " << g_pruned.structure[v]->name() << std::endl;

						dh_in.push_back(g_pruned.structure[v]);

						return false;
					}
				}
				return true;
			}
			, [](const auto& e) { return true; });

			lmu::writeConnectionGraph(output_path + "pruned_graph_rem_art_" + std::to_string(iter - 1) + "_" + std::to_string(rec_level) + ".gv", g_rem_art);

			std::cout << (iter - 1) << " " << rec_level << " Inside DHs: " << to_list_str(dh_in) << std::endl;
			std::cout << (iter - 1) << " " << rec_level << " Outside DHs: " << to_list_str(dh_out) << std::endl;
			
			// Get connected components. 
			t.tick();
			auto rem_art_components = lmu::getConnectedComponents(g_rem_art);
			timings.connected_components += t.tick();

			auto sub_node = lmu::opNo();
			if (rem_art_components.size() == 1 && !something_was_pruned)
			{
				std::ofstream res_ga_1, res_ga_2;
				res_ga_1.open(output_path + "result_ga_" + std::to_string(stats.num_ga_calls) + "_1.txt");
				res_ga_2.open(output_path + "result_ga_" + std::to_string(stats.num_ga_calls) + "_2.txt");
				stats.num_ga_calls++;

				auto primitives = lmu::getImplicitFunctions(rem_art_components[0]);
				std::cout << "Generate node for primitives " << to_list_str(primitives) << std::endl;

				auto sdf_mesh = model_sdf_wo_out_dhs->to_mesh();
				auto sdf_pc = model_sdf_wo_out_dhs->to_pc();
				igl::writeOBJ(output_path + "sdf_mesh_" + std::to_string(stats.num_ga_calls - 1) + ".obj", sdf_mesh.vertices, sdf_mesh.indices);
				lmu::writePointCloudXYZ(output_path + "sdf_pc_" + std::to_string(stats.num_ga_calls - 1) + ".xyz", sdf_pc);

				t.tick(); 
				auto res = lmu::generate_csg_node(primitives, model_sdf_wo_out_dhs, ng_params, res_ga_1, res_ga_2, gt_node);
				sub_node = res.node;
				timings.ga += t.tick();

				lmu::writePointCloudXYZ(output_path + "inside_cit_pc_" + std::to_string(stats.num_ga_calls - 1) + ".xyz", lmu::pointCloudFromVector(res.points));

				res_ga_1.close();
				res_ga_2.close();
			}
			else
			{
				sub_node = decompose(rem_art_components, model_sdf_wo_out_dhs, ng_params, gt_node, output_path, timings, stats, rec_level + 1);
			}

			if (sub_node.type() == lmu::CSGNodeType::Geometry || sub_node.operationType() != lmu::CSGNodeOperationType::Noop)
			{
				per_graph_node.addChild(sub_node);
			}
		}

		stats.num_outside_dhs += dh_out.size();
		stats.num_inside_dhs += dh_in.size();

		for (const auto& dh : dh_in)
		{
			per_graph_node.addChild(lmu::geometry(dh));
		}

		// Avoid single child operations.
		if (per_graph_node.childsCRef().size() == 1)
		{
			per_graph_node = per_graph_node.childsCRef()[0];
		}

		for (const auto& dh : dh_out)
		{
			//per_graph_node.addChild(lmu::opComp({ lmu::geometry(dh) }));
			per_graph_node = lmu::opDiff({ per_graph_node, lmu::geometry(dh) });
		}
		
		// Avoid single child operations.
		if (per_graph_node.childsCRef().size() == 1)
		{
			per_graph_node = per_graph_node.childsCRef()[0];
		}

		node.addChild(per_graph_node);
	}

	lmu::writeNode(node, output_path + "node_" + std::to_string(rec_level) + ".json");
	
	// Avoid single child operations.
	if (node.childsCRef().size() == 1)
		node = node.childsCRef()[0];

	return node;
}