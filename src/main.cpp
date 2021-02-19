#include <iostream>
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

int main(int argc, char *argv[])
{
	if (argc != 4)
	{
		std::cerr << "Wrong number of arguments." << std::endl;
		return 1;
	}
		
	try
	{
		auto input_cluster_file = std::string(argv[1]);
		auto input_pc_file = std::string(argv[2]);

		auto output_cluster_file = std::string(argv[3]);

		std::cout << "input cluster file: " << input_cluster_file << std::endl;
		auto convex_clusters = lmu::get_convex_clusters_without_planes(input_cluster_file, true);

		std::vector<lmu::PointCloud> convex_cluster_pcs;
		std::transform(convex_clusters.begin(), convex_clusters.end(), std::back_inserter(convex_cluster_pcs), [](const auto& c) { return c.pc; });
		auto merged_convex_cluster_pcs = lmu::mergePointClouds(convex_cluster_pcs);

		Eigen::Vector3d cc_min = merged_convex_cluster_pcs.leftCols(3).colwise().minCoeff();
		Eigen::Vector3d cc_max = merged_convex_cluster_pcs.leftCols(3).colwise().maxCoeff();

		for (auto& c : convex_clusters)
		{
			c.pc = lmu::to_canonical_frame(c.pc, cc_min, cc_max);
		}


		std::cout << "pc file: " << input_pc_file << std::endl;
		auto pc = lmu::readPointCloudXYZ(input_pc_file);

		Eigen::Vector3d min = pc.leftCols(3).colwise().minCoeff();
		Eigen::Vector3d max = pc.leftCols(3).colwise().maxCoeff();
				
		pc = lmu::to_canonical_frame(pc, min, max);

		std::cout << "pc size: " << pc.rows() << std::endl;

		lmu::reassign_convex_cluster_pointclouds(convex_clusters, pc);

		lmu::write_convex_clusters_to_ply(output_cluster_file, convex_clusters);

		std::cout << "DONE" << std::endl;
	}
	catch (const std::exception& ex)
	{
		std::cerr << "An Error occurred: " << ex.what() << std::endl;
		return 1;
	}
}

 