#include "optimizer_clustering.h"
#include "csgnode.h"
#include "csgnode_helper.h"

void cluster_rec(const lmu::CSGNode& n, std::vector<lmu::CSGNode>& clusters)
{
	if (n.operationType() == lmu::CSGNodeOperationType::Union)
	{
		for (const auto& c : n.childsCRef())
			cluster_rec(c, clusters);
	}
	else
	{	
		clusters.push_back(n);
	}
}

std::vector<lmu::CSGNode> lmu::cluster_union_paths(const CSGNode& node)
{
	std::vector<CSGNode> clusters;

	cluster_rec(node, clusters);

	return clusters; 
}

bool is_fully_contained_in(const lmu::ImplicitFunctionPtr& primitive, const lmu::CSGNode& node, double sampling_grid_size)
{
	lmu::AABB aabb = primitive->aabb();
	Eigen::Vector3d min = aabb.c - aabb.s;
	Eigen::Vector3d max = aabb.c + aabb.s;

	return _is_in(primitive, node, sampling_grid_size, min, max);
}

std::vector<lmu::ImplicitFunctionPtr> lmu::find_dominating_prims(const CSGNode& node, double sampling_grid_size)
{
	std::vector<ImplicitFunctionPtr> dps;

	for (const auto& primitive : allDistinctFunctions(node))
	{
		if (is_fully_contained_in(primitive, node, sampling_grid_size))
			dps.push_back(primitive);
	}

	return dps;
}

std::vector<lmu::PrimitiveCluster> lmu::cluster_with_dominating_prims(const CSGNode& node, const std::vector<ImplicitFunctionPtr>& dom_prims)
{
	std::vector<PrimitiveCluster> clusters;
	Graph graph = createConnectionGraph(allDistinctFunctions(node));
	auto pruneList = createPruneList(graph, createNeighborMap(graph));
	auto prunedGraph = pruneGraph(graph, pruneList);

	//Remove prime implicants from graph.	
	struct Predicate
	{
		bool operator()(GraphStructure::edge_descriptor) const
		{
			return true;
		}
		bool operator()(GraphStructure::vertex_descriptor vd) const
		{
			//std::cout << g->structure[vd]->name() << ": Pruned: " << wasPruned(*g, g->structure[vd]) << " PI: " << (std::find(pis->begin(), pis->end(), g->structure[vd]) != pis->end())  << std::endl;

			return std::find(pis->begin(), pis->end(), g->structure[vd]) == pis->end();
		}

		const Graph* g;
		const std::vector<lmu::ImplicitFunctionPtr>* pis;

	} predicate{ &graph, &dom_prims };

	boost::filtered_graph<GraphStructure, Predicate, Predicate> fg(graph.structure, predicate, predicate);

	lmu::Graph newGraph;
	boost::copy_graph(fg, newGraph.structure);
	lmu::recreateVertexLookup(newGraph);

	//std::cout << "New graph created." << std::endl;
	//lmu::writeConnectionGraph("connectionGraph.dot", newGraph);

	//Get connected components. 
	auto partitions = lmu::getConnectedComponents(newGraph);

	for (const auto& partition : partitions)
	{	
		PrimitiveCluster cluster;
		for (const auto kv : partition.vertexLookup)
		{
			cluster.push_back(kv.first);
		}
		clusters.push_back(cluster);
	}

	for (const auto& pi : dom_prims)
	{
		clusters.push_back({ pi });
	}

	return clusters;
}

lmu::CSGNode lmu::apply_per_cluster_optimization(std::vector<CSGNode> nodes, 
	const std::function<CSGNode(const CSGNode&)>& optimizer, 
	const std::function<CSGNode(const std::vector<CSGNode>&)>& merger)
{
	std::vector<CSGNode> opt_nodes;
	opt_nodes.reserve(nodes.size());
	for (const auto& node : nodes)
	{
		opt_nodes.push_back(optimizer(node));
	}

	return merger(opt_nodes);
}

lmu::CSGNode lmu::apply_per_cluster_optimization(std::vector<lmu::PrimitiveCluster> primitive_clusters,
	const std::function<CSGNode(const lmu::PrimitiveCluster&)>& optimizer,
	const std::function<CSGNode(const std::vector<CSGNode>&)>& merger)
{
	std::vector<CSGNode> opt_nodes;
	opt_nodes.reserve(primitive_clusters.size());
	for (const auto& c : primitive_clusters)
	{
		opt_nodes.push_back(optimizer(c));
	}

	return merger(opt_nodes);
}

lmu::CSGNode lmu::union_merge(const std::vector<CSGNode>& nodes)
{
	if (nodes.empty()) return lmu::opNo();

	if (nodes.size() == 1) return nodes[0];

	return lmu::opUnion(nodes);
}
