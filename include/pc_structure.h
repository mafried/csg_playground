#ifndef PC_STRUCTURE_H
#define PC_STRUCTURE_H

#include "pointcloud.h"
#include "primitives.h"
#include <boost/graph/adjacency_list.hpp>

namespace lmu
{

	struct PlaneGraph
	{
		typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
			ManifoldPtr> PlaneGraphStructure;

		using PlaneEdgeDescriptor = boost::graph_traits<GraphStructure>::edge_descriptor;
		using PlaneVertexDescriptor = boost::graph_traits<GraphStructure>::vertex_descriptor;

		void add_plane(const ManifoldPtr& plane);
		void add_connection(const ManifoldPtr& p1, const ManifoldPtr& p2);

		ManifoldSet connected(const ManifoldPtr& plane) const;
		bool is_connected(const ManifoldPtr& p1, const ManifoldPtr& p2) const;

		ManifoldSet planes() const;

		PointCloud plane_points() const;

		void to_file(const std::string& file) const;

	private: 
		PlaneGraphStructure graph;
		std::unordered_map<ManifoldPtr, PlaneVertexDescriptor> vertex_map;
	};

	lmu::PlaneGraph create_plane_graph(const lmu::ManifoldSet& ms, lmu::PointCloud& debug_pc, lmu::PointCloud& pcwn, double epsilon = 0.0);
	void resample_proportionally(const lmu::ManifoldSet& ms, int total_max_points);

	std::pair<lmu::PointCloud, std::vector<int>> resample_pointcloud(const lmu::PlaneGraph& pg, double range_scale_factor);
	std::pair<lmu::PointCloud, std::vector<int>> resample_pointcloud(const lmu::PlaneGraph& pg);

}

#endif