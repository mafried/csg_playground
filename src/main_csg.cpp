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

lmu::CSGNode decompose(const std::vector<lmu::Graph>& graphs, const std::shared_ptr<lmu::ModelSDF>& model_sdf,
	const lmu::CSGNodeGenerationParams& ng_params, const lmu::CSGNode& gt_node, const std::string& output_path, Timings& timings, Stats& stats, int rec_level = 0);

bool is_inside_dh(const lmu::ImplicitFunctionPtr& prim, const lmu::ModelSDF& model_sdf)
{
	Eigen::Vector3d center_point(0, 0, 0);
	for (int i = 0; i < prim->meshCRef().vertices.rows(); ++i)
	{
		center_point += prim->meshCRef().vertices.row(i).transpose();
	}

	center_point = center_point / (double)prim->meshCRef().vertices.rows();

	return model_sdf.distance(center_point) <= 0.0;
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
	auto input_pc_file = s.getStr("Data", "InputPointCloudFile", "");
	auto sample_cell_size = s.getDouble("Data", "SampleCellSize", 0.1);

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

		std::cout << "load node from " << input_node_file << std::endl;
		auto node = lmu::fromJSONFile(input_node_file);
		lmu::writeNode(node, out_path + "input_node.gv");

		res_f << "GraphNodes= " << lmu::numNodes(node) << std::endl;

		auto aabb = lmu::aabb_from_node(node);
		Eigen::Vector3d min = aabb.c - aabb.s;
		Eigen::Vector3d max = aabb.c + aabb.s;
		double cell_size = (max.maxCoeff() - min.minCoeff()) / (double)num_cells;
		std::cout << "node aabb: (" << min.transpose() << ")(" << max.transpose() << ")" << std::endl;
		std::cout << "cell size: " << cell_size << std::endl;

		// Load point cloud. 
		
		auto pc = lmu::PointCloud();
		if (!input_pc_file.empty())
		{
			std::cout << "load pc from " << input_pc_file << std::endl;
			pc = lmu::readPointCloudXYZ(input_pc_file);
		}
		else
		{
			std::cout << "pc file is not available. Try to sample node." << std::endl;

			lmu::CSGNodeSamplingParams sampling_params(sample_cell_size * 0.5, 0.0, 0.0, sample_cell_size,  min, max);
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
		auto result_node = decompose(initial_components, model_sdf, ng_params, node, out_path, timings, stats);
		timings.decomposition += t.tick();

		/*
		std::vector<lmu::ImplicitFunctionPtr> dh_in; 
		std::vector<lmu::ImplicitFunctionPtr> dh_out;
		std::vector<lmu::CSGNode> sub_nodes; 

		int i = 0;
		int num_pruned_primitives = 0;
		int ga_counter = 0;

		// Find partitions. 

		for (const auto& c : initial_components)
		{
			// If component contains only a single element, it must be a dominant halfspace fully inside the volume.
			if (lmu::numVertices(c) == 1)
			{
				dh_in.push_back(getImplicitFunctions(c)[0]);
				continue; 
			}

			// Prune component.
			auto c_pruned = lmu::pruneGraph(c);

			// Check if pruned primitives are in- or out-dhs.
			auto pruned_primitives = lmu::get_pruned_primitives(c, c_pruned); 
			num_pruned_primitives += pruned_primitives.size();
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

			// Find articulation points.
			auto aps = get_articulation_points(c);
			res_f << "NumArticulationPoints= " << aps.size() << std::endl;

			// Select articulation points surrounded only by other articulation points.
			auto aps_with_neighbors = lmu::select_aps_with_aps_as_neighbors(aps, c);
			res_f << "NumArticulationPointsWithAPNeighbors= " << aps_with_neighbors.size() << std::endl;

			// Remove selected articulation points from pruned graph and add them to the list of in-dhs.
			auto c_rem_art = c_pruned;
			for (const auto& ap : aps_with_neighbors)
			{	
				auto prim = c.structure[ap];

				dh_in.push_back(prim);
				
				boost::remove_vertex(c_rem_art.vertexLookup[prim], c_rem_art.structure);
			}
			lmu::recreateVertexLookup(c_rem_art);
			
			// Get connected components. 
			auto rem_art_components = lmu::getConnectedComponents(c_rem_art);

			for (const auto& rem_art_c : rem_art_components)
			{
				if (lmu::numVertices(c) == 1)
				{
					std::cout << "This should not happen. Right?" << std::endl;
					continue;
				}

				// Get optimal node with ga. 
				ofstream res_ga_1, res_ga_2;
				res_ga_1.open(out_path + "result_ga_" + std::to_string(ga_counter) + "_1.txt");
				res_ga_2.open(out_path + "result_ga_" + std::to_string(ga_counter) + "_2.txt");
				ga_counter++;

				auto sub_node = lmu::generate_csg_node(lmu::getImplicitFunctions(rem_art_c), model_sdf, ng_params, res_ga_1, res_ga_2, node).node;
				if (sub_node.operationType() != lmu::CSGNodeOperationType::Noop)
				{
					sub_nodes.push_back(sub_node);
				}

				res_ga_1.close(); 
				res_ga_2.close(); 
			}
			
			lmu::writeConnectionGraph(out_path + "pruned_initial_component_" + std::to_string(i) + ".gv", c_pruned);
			lmu::writeConnectionGraph(out_path + "rem_art_components_" + std::to_string(i) + ".gv", c_rem_art);
			i++;
		}

		res_f << "NumPrunedPrimitives= " << num_pruned_primitives << std::endl;

		// Assemble result node.
		auto result_node = lmu::opUnion(); 
		for (const auto& sn : sub_nodes)
			result_node.addChild(sn);
		for (const auto& dh : dh_in)
			result_node.addChild(lmu::geometry(dh));
		for (const auto& dh : dh_out)
			result_node = lmu::opDiff({ result_node, lmu::geometry(dh) });

		if (result_node.childsCRef().size() == 1)
			result_node = result_node.childsCRef()[0];
		*/

		lmu::writeNode(result_node, out_path + "result_node.gv");
		lmu::toJSONFile(result_node, out_path + "result_node.json");
		
		lmu::Mesh result_mesh = lmu::computeMesh(result_node, Eigen::Vector3i((max.x() - min.x()) / sample_cell_size, (max.y() - min.y()) / sample_cell_size, (max.z() - min.z()) / sample_cell_size), min, max);
		igl::writeOBJ(out_path + "result_node.obj", result_mesh.vertices, result_mesh.indices);

		res_f << "Timings= " << timings.to_string() << std::endl;
		res_f << "Stats= " << stats.to_string() << std::endl;

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
			auto g_rem_art = g_pruned;
			for (const auto& ap : aps_with_neighbors)
			{
				auto prim = g.structure[ap];

				dh_in.push_back(prim);

				boost::clear_vertex(g_rem_art.vertexLookup[prim], g_rem_art.structure);
				boost::remove_vertex(g_rem_art.vertexLookup[prim], g_rem_art.structure);
			}
			lmu::recreateVertexLookup(g_rem_art);
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

				lmu::writePointCloudXYZ(output_path + "insice_cit_pc_" + std::to_string(stats.num_ga_calls - 1) + ".xyz", lmu::pointCloudFromVector(res.points));

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