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

	auto out_path = s.getStr("Data", "OutputFolder", "\\");
	auto num_cells = s.getInt("ConnectionGraph", "NumCells", 50);

	lmu::CSGNodeGenerationParams ng_params;
	ng_params.create_new_prob = s.getDouble("NodeGeneration", "CreateNewProbability", 0.5);
	ng_params.max_tree_depth = s.getInt("NodeGeneration", "MaxTreeDepth", 25);
	ng_params.subtree_prob = s.getDouble("NodeGeneration", "SubtreeProbability", 0.5);

	ng_params.size_weight = s.getDouble("NodeGeneration", "SizeWeight", 0.01);
	ng_params.geo_weight = s.getDouble("NodeGeneration", "GeoWeight", 1.0);
	ng_params.max_iterations = s.getInt("NodeGeneration", "MaxIterations", 100);
	ng_params.max_count = s.getInt("NodeGeneration", "MaxCount", 50);

	s.print();
	std::cout << "--------------------------------------------------------------" << std::endl;

	// Initialize
	update(viewer);

	ofstream res_f;
	res_f.open(out_path + "result.txt");
	
	try
	{
		// Load input node for primitive parameters. 

		std::cout << "load node from " << input_node_file << std::endl;
		auto node = lmu::fromJSONFile(input_node_file);
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
			pc = lmu::readPointCloudXYZ(input_node_file);
		}
		else
		{
			std::cout << "pc file is not available. Try to sample node." << std::endl;

			lmu::CSGNodeSamplingParams sampling_params(cell_size * 2.0, 0.0, 0.0, cell_size, min, max);
			pc = computePointCloud(node, sampling_params);
		}
		res_f << "Points= " << pc.rows() << std::endl;
		lmu::writePointCloudXYZ(out_path + "pc.xyz", pc);

		//pc = lmu::farthestPointSampling(pc, 32000);
		//res_f << "PointsAfterSampling: " << pc.rows() << std::endl;

		// Create discrete model sdf. 

		std::cout << "create discrete model sdf. " << std::endl;
		auto model_sdf = std::make_shared<lmu::ModelSDF>(pc, cell_size, res_f);
		igl::writeOBJ(out_path + "sdf_model.obj", model_sdf->surface_mesh.vertices, model_sdf->surface_mesh.indices);

		// Create connection graph.

		auto prims = lmu::allDistinctFunctions(node);

		auto graph = lmu::createConnectionGraph(prims, cell_size);

		auto initial_components = lmu::getConnectedComponents(graph);
		res_f << "InitialComponents= " << initial_components.size() << std::endl;

		std::vector<lmu::ImplicitFunctionPtr> dh_in; 
		std::vector<lmu::ImplicitFunctionPtr> dh_out;
		std::vector<lmu::CSGNode> sub_nodes; 

		int i = 0;
		int num_pruned_primitives = 0;

		// Find partitions. 

		for (const auto& c : initial_components)
		{
			// If component contains only a single element, it must be a dominant halfspace fully inside.
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
			lmu::Graph c_rem_art(c_pruned);
			for (const auto& ap : aps_with_neighbors)
			{	
				dh_in.push_back(c_rem_art.structure[ap]);
				boost::remove_vertex(ap, c_rem_art.structure);
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

				// Get optimal node with ga 
				auto sub_node = lmu::generate_csg_node(lmu::getImplicitFunctions(rem_art_c), model_sdf, ng_params, res_f, node).node;

				sub_nodes.push_back(sub_node);
			}
			
			lmu::writeConnectionGraph(out_path + "pruned_initial_component_" + std::to_string(i++) + ".gv", c_pruned);
			lmu::writeConnectionGraph(out_path + "rem_art_components_" + std::to_string(i++) + ".gv", c_rem_art);
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

		lmu::writeNode(result_node, out_path + "result_node.gv");
		lmu::toJSONFile(result_node, out_path + "result_node.json");
			
		lmu::Mesh result_mesh = lmu::computeMesh(result_node, Eigen::Vector3i(num_cells, num_cells, num_cells), min, max);
		igl::writeOBJ(out_path + "result_node.obj", result_mesh.vertices, result_mesh.indices);

		lmu::writeConnectionGraph(out_path + "input_graph.gv", graph);
		
		std::cout << "Close file" << std::endl;
		res_f.close();
	}
	catch (const std::exception& ex)
	{
		std::cout << "ERROR: " << ex.what() << std::endl;
	}

_LAUNCH:

	return 0;
}