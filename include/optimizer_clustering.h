#ifndef OPTIMIZER_CLUSTERING_H
#define OPTIMIZER_CLUSTERING_H

#include <vector>
#include "mesh.h"

namespace lmu 
{
	class CSGNode;
	
	std::vector<CSGNode> cluster_union_paths(const CSGNode& node);

	std::vector<ImplicitFunctionPtr> find_dominating_prims(const CSGNode& node, double sampling_grid_size);

	using PrimitiveCluster = std::vector<ImplicitFunctionPtr>;

	std::vector<PrimitiveCluster> cluster_with_dominating_prims(const CSGNode& node, const std::vector<ImplicitFunctionPtr>& dom_prims);

	CSGNode union_merge(const std::vector<CSGNode>& nodes);

	CSGNode apply_per_cluster_optimization(std::vector<CSGNode> nodes, const std::function<CSGNode(const CSGNode&)>& optimizer,
		const std::function<CSGNode(const std::vector<CSGNode>&)>& merger);

	CSGNode apply_per_cluster_optimization(std::vector<PrimitiveCluster> nodes, const std::function<CSGNode(const PrimitiveCluster&)>& optimizer,
		const std::function<CSGNode(const std::vector<CSGNode>&)>& merger);
}

#endif
