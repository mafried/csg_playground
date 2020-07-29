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

	lmu::PlaneGraph structure_pointcloud(const lmu::ManifoldSet& ms, double epsilon, lmu::PointCloud& debug_pc);
}

#endif