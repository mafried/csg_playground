#ifndef OPTIMIZER_CLUSTERING_H
#define OPTIMIZER_CLUSTERING_H

#include <vector>
#include "mesh.h"

namespace lmu 
{
	class CSGNode;
	
	std::vector<CSGNode> cluster_union_paths(const CSGNode& node);

	std::vector<ImplicitFunctionPtr> find_dominating_prims(const CSGNode& node, double sampling_grid_size);

	std::vector<CSGNode> cluster_with_dominating_prims(const CSGNode& node, const std::vector<ImplicitFunctionPtr>& dom_prims);

	CSGNode union_merge(const std::vector<CSGNode>& nodes);

	CSGNode apply_per_cluster_optimization(std::vector<CSGNode> nodes, const std::function<CSGNode(const CSGNode&)>& optimizer,
		const std::function<CSGNode(const std::vector<CSGNode>&)>& merger);
}

#endif
