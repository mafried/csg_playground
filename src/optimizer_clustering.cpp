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

std::vector<lmu::ImplicitFunctionPtr> lmu::find_dominating_prims(const CSGNode & node, double sampling_grid_size)
{
	// TODO

	return std::vector<ImplicitFunctionPtr>();
}

std::vector<lmu::CSGNode> lmu::cluster_with_dominating_prims(const CSGNode & node, const std::vector<ImplicitFunctionPtr>& dom_prims)
{
	std::vector<CSGNode> clusters;

	// TODO

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

lmu::CSGNode lmu::union_merge(const std::vector<CSGNode>& nodes)
{
	if (nodes.empty()) return lmu::opNo();

	if (nodes.size() == 1) return nodes[0];

	return lmu::opUnion(nodes);
}
