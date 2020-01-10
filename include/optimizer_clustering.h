#ifndef OPTIMIZER_CLUSTERING_H
#define OPTIMIZER_CLUSTERING_H

#include <vector>

#include "mesh.h"
#include "csgnode.h"

namespace lmu 
{	
	std::vector<CSGNode> cluster_union_paths(const CSGNode& node);

	std::vector<ImplicitFunctionPtr> find_dominating_prims(const CSGNode& node, double sampling_grid_size);

	std::vector<ImplicitFunctionPtr> find_negated_dominating_prims(const CSGNode& node, double sampling_grid_size);

	using PrimitiveCluster = std::vector<ImplicitFunctionPtr>;

	std::vector<PrimitiveCluster> cluster_with_dominating_prims(const CSGNode& node, const std::vector<ImplicitFunctionPtr>& dom_prims);

	CSGNode union_merge(const std::vector<CSGNode>& nodes);

	CSGNode apply_per_cluster_optimization(std::vector<CSGNode> nodes, const std::function<CSGNode(const CSGNode&)>& optimizer,
		const std::function<CSGNode(const std::vector<CSGNode>&)>& merger);

	CSGNode apply_per_cluster_optimization(std::vector<PrimitiveCluster> nodes, const std::function<CSGNode(const PrimitiveCluster&)>& optimizer,
		const std::function<CSGNode(const std::vector<CSGNode>&)>& merger);

	struct DecompositionResult
	{
		DecompositionResult() : 
			node(std::make_shared<NoOperation>("")),
			noop_node_idx(0)
		{
		}

		PrimitiveCluster rest_prims;
		CSGNode node; 

		// If the tree is not already complete, the position in the tree
		// that is not complete is marked as a No-Operation at this index.
		int noop_node_idx; 
	
		// Already complete means that the there are no rest primitives 
		// and the corresponding tree is already complete.
		bool already_complete() const
		{
			return rest_prims.empty();
		}
	};

	DecompositionResult dom_prim_decomposition(const CSGNode& node, double sampling_grid_size, bool use_diff_op);

}

#endif
