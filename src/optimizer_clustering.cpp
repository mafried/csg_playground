#include "optimizer_clustering.h"
#include "csgnode_helper.h"
#include <boost/optional.hpp>
#include "optimizer_red.h"

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

std::vector<lmu::ImplicitFunctionPtr> lmu::find_dominating_prims(const CSGNode& node, double sampling_grid_size)
{
	std::vector<ImplicitFunctionPtr> dps;

	for (const auto& primitive : allDistinctFunctions(node))
	{
		lmu::AABB aabb = primitive->aabb();
		Eigen::Vector3d min = aabb.c - aabb.s;
		Eigen::Vector3d max = aabb.c + aabb.s;

		if(_is_in(primitive, node, sampling_grid_size, min, max))
		{
			dps.push_back(primitive);
		}
	}
	return dps;
}

std::vector<lmu::ImplicitFunctionPtr> lmu::find_dominating_prims(const CSGNode& node, const lmu::PointCloud& out)
{
	std::vector<ImplicitFunctionPtr> dps;

	for (const auto& primitive : allDistinctFunctions(node))
	{
		if (_is_in(primitive, out))
		{
			dps.push_back(primitive);
		}
	}

	return dps;
}

std::vector<lmu::ImplicitFunctionPtr> lmu::find_negated_dominating_prims(const CSGNode& node, double sampling_grid_size)
{
	return find_dominating_prims(opComp({ node }), sampling_grid_size);
}

std::vector<lmu::ImplicitFunctionPtr> lmu::find_negated_dominating_prims(const CSGNode& node, const lmu::PointCloud& in)
{
	std::vector<ImplicitFunctionPtr> dps;

	for (const auto& primitive : allDistinctFunctions(node))
	{
		if (_is_out(primitive, in))
		{
			dps.push_back(primitive);
		}
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

	} predicate{ &prunedGraph, &dom_prims };

	boost::filtered_graph<GraphStructure, Predicate, Predicate> fg(prunedGraph.structure, predicate, predicate);

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

lmu::PrimitiveCluster get_rest_prims(const lmu::PrimitiveCluster& base, 
	const lmu::PrimitiveCluster& minus)
{	
	lmu::PrimitiveCluster rest;

	auto sorted_base = base;
	auto sorted_minus = minus;

	std::sort(sorted_base.begin(), sorted_base.end());
	std::sort(sorted_minus.begin(), sorted_minus.end());

	std::set_difference(sorted_base.begin(), sorted_base.end(), sorted_minus.begin(), sorted_minus.end(),
		std::inserter(rest, rest.begin()));

	return rest;
}

using DominantPrim = std::pair<lmu::ImplicitFunctionPtr, bool>;
using DominantPrims = std::vector<DominantPrim>;

boost::optional<DominantPrim> select_next_from(const lmu::CSGNode& node, DominantPrims& dom_prims, 
	double sampling_grid_size, lmu::EmptySetLookup& esl)
{
	if (dom_prims.empty())
		return boost::none;

	if (node.operationType() != lmu::CSGNodeOperationType::Noop)
	{
		// Try to select a dominant primitive that is spatially connected to the existing node.
		for (int i = 0; i < dom_prims.size(); ++i)
		{
			if (!is_empty_set(lmu::opInter({ node, lmu::geometry(dom_prims[i].first) }), sampling_grid_size, lmu::empty_pc(), esl))
			{
				auto res = dom_prims[i];
				dom_prims.erase(dom_prims.begin() + i);

				return res;
			}
		}
	}
	
	auto res = dom_prims.back();
	dom_prims.pop_back();

	return boost::optional<DominantPrim>(res);
}

lmu::CSGNode compute_decomposed_expression(const lmu::PrimitiveCluster& dom_prims,
	const lmu::PrimitiveCluster& neg_dom_prims, bool use_diff_op, double sampling_grid_size)
{
	DominantPrims combined_dom_prims;
	std::transform(dom_prims.begin(), dom_prims.end(), std::back_inserter(combined_dom_prims), [](const lmu::ImplicitFunctionPtr& f) {return std::make_pair(f, true); });
	std::transform(neg_dom_prims.begin(), neg_dom_prims.end(), std::back_inserter(combined_dom_prims), [](const lmu::ImplicitFunctionPtr& f) {return std::make_pair(f, false); });

	std::cout << "# Dominant Prims: " << combined_dom_prims.size() << std::endl;

	lmu::EmptySetLookup esl;

	auto node = lmu::opNo();
	while(auto dom_prim = select_next_from(node, combined_dom_prims, sampling_grid_size, esl))
	{			
		if (dom_prim->second)
		{ /*union*/
			node = lmu::opUnion({ geometry(dom_prim->first), node });
		}
		else
		{ /*negated => intersection*/
			node = use_diff_op ? lmu::opDiff({ node, geometry(dom_prim->first) }) : 
				lmu::opInter({ node, lmu::opComp({geometry(dom_prim->first)}) });
		}
	} 

	//TODO: find a replacement for select_next_from() which takes proximity into account.

	return node;
}

lmu::DecompositionResult lmu::dom_prim_decomposition(const CSGNode& node, double sampling_grid_size, bool use_diff_op,
	const lmu::PointCloud& in, const lmu::PointCloud& out)
{
	PrimitiveCluster dom_prims; 
	PrimitiveCluster neg_dom_prims;
	
	// If there are no sampling points, sample hierarchically.
	if (in.rows() == 0)
	{
		dom_prims = find_dominating_prims(node, sampling_grid_size);
		neg_dom_prims = find_negated_dominating_prims(node, sampling_grid_size);
	}
	else
	{
		dom_prims = find_dominating_prims(node, out);
		neg_dom_prims = find_negated_dominating_prims(node, in);
	}

	auto rest_prims = get_rest_prims(
		get_rest_prims(allDistinctFunctions(node), dom_prims), neg_dom_prims);

	std::cout << "Dom Prims: " << dom_prims.size() << " Negated Dom Prims: " << neg_dom_prims.size() << " Rest Prims: " << rest_prims.size() << std::endl;

	DecompositionResult res;
	res.node = compute_decomposed_expression(dom_prims, neg_dom_prims, use_diff_op, sampling_grid_size);

	if (rest_prims.empty())
	{
		lmu::visit(res.node, [&rest_prims](lmu::CSGNode& n)
		{
			if (n.childsCRef().size() == 2)
			{
				if (n.childsCRef()[0].operationType() == CSGNodeOperationType::Noop) 
					n = n.childsCRef()[1];
				else if (n.childsCRef()[1].operationType() == CSGNodeOperationType::Noop) 
					n = n.childsCRef()[0];
			}
		});
	}
	else if (rest_prims.size() == 1)
	{
		lmu::visit(res.node, [&rest_prims](lmu::CSGNode& n)
		{
			if (n.operationType() == CSGNodeOperationType::Noop)
				n = geometry(rest_prims.back());			
		});
	}
	else
	{
		int idx = 0;
		lmu::visit(res.node, [&res, &idx](const lmu::CSGNode& n)
		{
			if (n.operationType() == CSGNodeOperationType::Noop)
			{
				res.noop_node_idx = idx;
			}

			idx++;
		});

		res.rest_prims = rest_prims;	
	}

	res.used_prims = allDistinctFunctions(res.node);

	return res;
}

lmu::CSGNode lmu::optimize_with_decomposition(const CSGNode& node, double sampling_grid_size, bool use_diff_op,
	const lmu::PointCloud& in, const lmu::PointCloud& out, const lmu::PointCloud& in_out,
	const std::function<CSGNode(const CSGNode& node, const PrimitiveCluster& prims)>& optimizer)
{
	std::cout << "Decompose node." << std::endl;
	
	auto dec = dom_prim_decomposition(node, sampling_grid_size, use_diff_op, in, out);

	if (!dec.already_complete())
	{
		auto opt_node = node;

		std::cout << "Replace decomposed prims with empty set marker." << std::endl;

		// Replace primitives used in the decomposition node part with empty set marker. 
		bool prim_replaced = false;
		visit(opt_node, [&dec, &prim_replaced](CSGNode& n)
		{
			static auto const empty_set = lmu::CSGNode(std::make_shared<lmu::NoOperation>("0"));

			if (n.type() == CSGNodeType::Geometry &&
				std::find(dec.used_prims.begin(), dec.used_prims.end(), n.function()) != dec.used_prims.end())
			{
				n = empty_set;
				prim_replaced = true;
			}
		});

		// Simplify.
		if (prim_replaced)
		{
			std::cout << "Remove redundancies." << std::endl;

			writeNode(opt_node, "opt_node_" + std::to_string(numNodes(opt_node)) + ".gv");

			opt_node = remove_redundancies(opt_node, sampling_grid_size, in_out);
			
			opt_node = optimize_with_decomposition(opt_node, sampling_grid_size, use_diff_op, in, out, in_out, optimizer);
		}
		else
		{
			std::cout << "Use external optimizer." << std::endl;

			// Use different technique to optimize opt_node.
			opt_node = optimizer(opt_node, dec.rest_prims);
		}

		// Combine rest node with decomposed part. 
		CSGNode* node_ptr = nodePtrAt(dec.node, dec.noop_node_idx);
		*node_ptr = opt_node;		
	}	
	else
	{
		std::cout << "Decomposition already complete." << std::endl;
	}

	return dec.node;
}
