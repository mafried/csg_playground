#include "prim_select_ga.h"
#include "csgnode_helper.h"
#include "optimizer_red.h"

using namespace lmu;

lmu::SelectionValue::SelectionValue(DHType dh_type, bool active) : 
	dh_type(dh_type),
	active(active)
{
}

size_t lmu::SelectionValue::hash(size_t seed) const
{
	boost::hash_combine(seed, dh_type);
	boost::hash_combine(seed, active);

	return seed;
}

lmu::PrimitiveSelection::PrimitiveSelection(const PrimitiveSet* primitives) : 
	prims(primitives),
	node(opNo())
{
	std::default_random_engine rnd_engine;
	std::random_device rnd_device;
	rnd_engine.seed(rnd_device());

	std::bernoulli_distribution d{};
	using parm_t = decltype(d)::param_type;

	if (primitives)
	{
		for (const auto& p : *primitives)
		{
			bool active = d(rnd_engine, parm_t{ 0.5 });
			DHType dh_type = d(rnd_engine, parm_t{ 0.5 }) ? DHType::INSIDE : DHType::OUTSIDE;

			selection.push_back(SelectionValue(dh_type, active));
		}
	}
}

lmu::PrimitiveSelection::PrimitiveSelection(const PrimitiveSet* primitives, const std::vector<DHType>& dh_types) :
	prims(primitives),
	node(opNo())
{
	if (primitives)
	{
		for (int i = 0; i < primitives->size(); ++i)
		{
			selection.push_back(SelectionValue(dh_types[i], true));
		}
	}
}

lmu::PrimitiveSelection::PrimitiveSelection(const CSGNode& node) : 
	node(node),
	prims(nullptr)
{
}

lmu::CSGNode lmu::PrimitiveSelection::to_node() const
{
	std::vector<CSGNode> diff_prims;
	std::vector<CSGNode> union_prims;

	for (int i = 0; i < selection.size(); ++i)
	{
		if (selection[i].active)
		{
			if (selection[i].dh_type == DHType::OUTSIDE)
				diff_prims.push_back(geometry(prims->at(i).imFunc));
			else
				union_prims.push_back(geometry(prims->at(i).imFunc));
		}
	}
	
	if (union_prims.empty() && diff_prims.empty())
	{
		return opNo();
	}
	else if (!union_prims.empty() && diff_prims.empty())
	{
		return union_prims.size() > 1 ? opUnion(union_prims) : union_prims[0];
	}
	else if (union_prims.empty() && !diff_prims.empty())
	{
		return opDiff({ opNo(),  diff_prims.size() > 1 ? opUnion(diff_prims) : diff_prims[0]});
	}
	else
	{
		return opDiff({ opUnion(union_prims), opUnion(diff_prims) });
	}
}

int lmu::PrimitiveSelection::get_num_active() const
{
	int n = 0;
	for (const auto& s : selection)
		n += s.active ? 1 : 0;
	return n;
}

size_t lmu::PrimitiveSelection::hash(size_t seed) const
{	
	if (node.operationType() == CSGNodeOperationType::Noop)
	{
		for (const auto& s : selection)
		{
			boost::hash_combine(seed, s.dh_type);
			boost::hash_combine(seed, s.active);
		}
	}
	else
	{
		seed = node.hash(seed);
	}

	return seed;
}

std::ostream& lmu::operator<<(std::ostream& out, const PrimitiveSelection& ps)
{
	int i = 0;
	for (const auto& s : ps.selection)
	{
		out << (ps.prims ? ps.prims->at(i++).imFunc->name() : "No name") << ": " << s.active << " " << (s.dh_type == DHType::INSIDE ? "IN" : s.dh_type == DHType::OUTSIDE ? "OUT" : "NONE") << " | ";
	}
	out << std::endl;

	return out;
}

//////////////////////////// CREATOR ////////////////////////////

struct SelectionCreator
{
	SelectionCreator(const lmu::PrimitiveSelection& ps, const std::shared_ptr<PrimitiveSetRanker>& primitive_ranker, const CSGNodeGenerationParams& params) :
		ps(ps),
		params(params)
	{
		_rndEngine.seed(_rndDevice());

		per_primitive_geo_scores = primitive_ranker->rank(*ps.prims).per_primitive_geo_scores;
	}

	lmu::PrimitiveSelection mutate(const lmu::PrimitiveSelection& s) const
	{
		static std::bernoulli_distribution d{};
		using parm_t = decltype(d)::param_type;

		static std::uniform_int_distribution<> uniform_d{};
		using p_ud = decltype(uniform_d)::param_type;
		
		std::vector<double> inv_geo_scores;
		std::transform(per_primitive_geo_scores.begin(), per_primitive_geo_scores.end(), std::back_inserter(inv_geo_scores),
			[](double s) -> double { return 1.0 - s; });
		std::discrete_distribution<int> discrete_d(inv_geo_scores.begin(), inv_geo_scores.end());
		
		if (d(_rndEngine, parm_t{ params.create_new_prob }))
		{
			return create();
		}
		else
		{
			int num_selections = ps.selection.size();
			auto new_s = s;

			for (int i = 0; i < num_selections; ++i)
			{
				int s_idx = params.use_prim_geo_scores_as_active_prob ? discrete_d(_rndEngine) : uniform_d(_rndEngine, p_ud{ 0, num_selections - 1 });
							
				new_s.selection[s_idx].active = d(_rndEngine, parm_t{ 0.5 });
				if (params.evolve_dh_type)
				{
					new_s.selection[s_idx].dh_type = s.selection[s_idx].dh_type == DHType::INSIDE ? DHType::OUTSIDE : DHType::INSIDE; //TODO
				}
			}
			
			return new_s;
		}
	}

	std::vector<PrimitiveSelection> crossover(const PrimitiveSelection& s1, const PrimitiveSelection& s2) const
	{
		int num_selections = ps.selection.size();

		auto new_s1 = s1;
		auto new_s2 = s2;

		static std::uniform_int_distribution<> uniform_d{};
		using p_ud = decltype(uniform_d)::param_type;
	
		std::vector<double> inv_geo_scores;
		std::transform(per_primitive_geo_scores.begin(), per_primitive_geo_scores.end(), std::back_inserter(inv_geo_scores),
			[](double s) -> double { return 1.0 - s; });
		std::discrete_distribution<int> discrete_d(inv_geo_scores.begin(), inv_geo_scores.end());
		
		int x1 = params.use_prim_geo_scores_as_active_prob ? discrete_d(_rndEngine) : uniform_d(_rndEngine, p_ud{ 0, num_selections - 1 });
		int x2 = params.use_prim_geo_scores_as_active_prob ? discrete_d(_rndEngine) : uniform_d(_rndEngine, p_ud{ 0, num_selections - 1 });

		int start = std::min(x1, x2);
		int stop = std::max(x1, x2);

		for (int i = start; i <= stop; ++i)
		{
			auto tmp_s = new_s1;
			new_s1.selection[i].active = new_s2.selection[i].active;
			new_s2.selection[i].active = tmp_s.selection[i].active;

			if (params.evolve_dh_type)
			{
				new_s1.selection[i].dh_type = new_s2.selection[i].dh_type;
				new_s2.selection[i].dh_type = tmp_s.selection[i].dh_type;
			}
		}

		return std::vector<lmu::PrimitiveSelection>
		{
			new_s1, new_s2
		};
	}

	PrimitiveSelection create() const
	{
		static std::bernoulli_distribution d{};
		using parm_t = decltype(d)::param_type;

		auto new_ps = ps;

		for (int i = 0; i < new_ps.selection.size(); ++i)
		{
			new_ps.selection[i].active = d(_rndEngine, parm_t{ params.use_prim_geo_scores_as_active_prob ? per_primitive_geo_scores[i] : params.active_prob });

			if (params.evolve_dh_type)
				new_ps.selection[i].dh_type = d(_rndEngine, parm_t{ params.dh_type_prob }) ? DHType::INSIDE : DHType::OUTSIDE;
		}

		//std::cout << "CREATED: " << new_ps << std::endl;

		return new_ps;
	}

	std::string info() const
	{
		return std::string();
	}

	private:

		lmu::PrimitiveSelection ps;
		std::vector<double> per_primitive_geo_scores;

	
		CSGNodeGenerationParams params;

		mutable std::default_random_engine _rndEngine;
		mutable std::random_device _rndDevice;
};


struct CSGNodeCreator
{
	CSGNodeCreator(const lmu::PrimitiveSelection& ps, const std::shared_ptr<PrimitiveSetRanker>& primitive_ranker, const CSGNodeGenerationParams& params) :
		params(params)
	{
		for (const auto& p : *ps.prims)
			primitives.push_back(p.imFunc);	

		per_primitive_geo_scores = primitive_ranker->rank(*ps.prims).per_primitive_geo_scores;

		_rndEngine.seed(_rndDevice());
	}

	PrimitiveSelection mutate(const PrimitiveSelection& ps) const
	{
		auto node = ps.node;

		static std::bernoulli_distribution d{};
		using parm_t = decltype(d)::param_type;

		static std::uniform_int_distribution<> du{};
		using parmu_t = decltype(du)::param_type;

		static std::uniform_real_distribution<double> dur(-0.1, 0.1);
		using parmur_t = decltype(dur)::param_type;

		if (d(_rndEngine, parm_t{ params.create_new_prob }))
		{
			return create(params.max_tree_depth);
		}
		else
		{
			int nodeIdx = du(_rndEngine, parmu_t{ 0, numNodes(node) - 1 });

			auto newNode = node;

			CSGNode* subNode = nodePtrAt(newNode, nodeIdx);

			create(*subNode, params.max_tree_depth, 0);

			return PrimitiveSelection(newNode);
		}
	}

	std::vector<PrimitiveSelection> crossover(const PrimitiveSelection& ps1, const PrimitiveSelection& ps2) const
	{
		auto node1 = ps1.node;
		auto node2 = ps2.node;

		if (!node1.isValid() || !node2.isValid())
			return std::vector<PrimitiveSelection> {PrimitiveSelection(node1), PrimitiveSelection(node2)};

		int numNodes1 = numNodes(node1);
		int numNodes2 = numNodes(node2);

		auto newNode1 = node1;
		auto newNode2 = node2;

		static std::uniform_int_distribution<> du{};
		using parmu_t = decltype(du)::param_type;

		int nodeIdx1 = du(_rndEngine, parmu_t{ 0, numNodes1 - 1 });
		int nodeIdx2 = du(_rndEngine, parmu_t{ 0, numNodes2 - 1 });

		CSGNode* subNode1 = nodePtrAt(newNode1, nodeIdx1);
		CSGNode* subNode2 = nodePtrAt(newNode2, nodeIdx2);

		std::swap(*subNode1, *subNode2);

		return std::vector<PrimitiveSelection>
		{
			PrimitiveSelection(newNode1), PrimitiveSelection(newNode2)
		};
	}

	PrimitiveSelection create() const
	{
		return create(params.max_tree_depth);
	}

	PrimitiveSelection create(int max_depth) const
	{
		auto node = opNo();
		create(node, max_depth, 0);
		return PrimitiveSelection(node);
	}

	std::string info() const
	{
		return std::string();
	}

private:

	void create(CSGNode& node, int max_depth, int cur_depth) const
	{
		static std::bernoulli_distribution db{};
		using parmb_t = decltype(db)::param_type;

		if (cur_depth >= max_depth)
		{
			node = create_rnd_primitive_node();
		}
		else
		{
			if (db(_rndEngine, parmb_t{ params.subtree_prob }))
			{
				node = create_rnd_operation_node();

				auto numAllowedChilds = node.numAllowedChilds();
				int numChilds = clamp(std::get<1>(numAllowedChilds), std::get<0>(numAllowedChilds), 2); //2 is the maximum number of childs allowed for create

				for (int i = 0; i < numChilds; ++i)
				{
					auto child = CSGNode::invalidNode;
					create(child, max_depth, cur_depth + 1);
					node.addChild(child);
				}
			}
			else
			{
				node = create_rnd_primitive_node();
			}
		}
	}

	CSGNode create_rnd_primitive_node() const
	{
		static std::uniform_int_distribution<> du{};
		using parmu_t = decltype(du)::param_type;

		std::discrete_distribution<int> discrete_d(per_primitive_geo_scores.begin(), per_primitive_geo_scores.end());
		
		int funcIdx = params.use_prim_geo_scores_as_active_prob ? discrete_d(_rndEngine) : du(_rndEngine, parmu_t{ 0, static_cast<int>(primitives.size() - 1) });
		return geometry(primitives[funcIdx]);
	}

	CSGNode create_rnd_operation_node() const
	{
		std::discrete_distribution<> d({ 1, 1, 1, 1 });
		int op = d(_rndEngine) + 1; //0 is OperationType::Unknown, 6 is OperationType::Invalid.

		return createOperation(static_cast<CSGNodeOperationType>(op));
	}

	CSGNodeGenerationParams params;
	std::vector<ImplicitFunctionPtr> primitives;

	std::vector<double> per_primitive_geo_scores;

	mutable std::default_random_engine _rndEngine;
	mutable std::random_device _rndDevice;
};

//////////////////////////// RANKER ////////////////////////////

struct SelectionRanker
{
	SelectionRanker(const std::shared_ptr<ModelSDF>& model_sdf, const CSGNode& start_node, CreatorStrategy creator_strategy) :
		model_sdf(model_sdf),
		start_node(start_node),
		creator_strategy(creator_strategy)
	{
	}

	SelectionRank rank(const PrimitiveSelection& s) const
	{
		//static int counter = 0;
		//std::cout << "counter: " << counter << std::endl;
		//counter++;

		auto n = creator_strategy == CreatorStrategy::SELECTION ? integrate_node(start_node, s) : integrate_node(start_node, s.node);
		auto d = 0.0;

		//Invalid node?
		if(numNodes(n) == 0 || n.operationType() == CSGNodeOperationType::Noop)
			return SelectionRank(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 0.0);

		auto grid_size = model_sdf->grid_size;
		auto voxel_size = model_sdf->voxel_size;
		auto origin = model_sdf->origin;

		for (int x = 0; x < grid_size.x(); ++x)
		{
			for (int y = 0; y < grid_size.y(); ++y)
			{
				for (int z = 0; z < grid_size.z(); ++z)
				{
					Eigen::Vector3d p = Eigen::Vector3d(x, y, z) * voxel_size + origin;

					auto v = model_sdf->sdf_value(p);

					d += std::abs(v.v - n.signedDistance(p));					
				}
			}
		}

		auto size = creator_strategy == CreatorStrategy::SELECTION ? (double)s.get_num_active() : numNodes(s.node);
		
		auto sr = SelectionRank(d, size, 0.0);

		//std::cout << "Size: " << (numNodes(s.node)) << " Score: " << sr << std::endl;
		
		return sr;
	}

	std::string info() const
	{
		return std::string();
	}

private:

	std::shared_ptr<ModelSDF> model_sdf;
	CSGNode start_node;
	CreatorStrategy creator_strategy;
};


//////////////////////////// POP MAN ////////////////////////////

struct SelectionPopMan
{
	SelectionPopMan(double geo_weight, double size_weight) :
		geo_weight(geo_weight),
		size_weight(size_weight)
	{
	}

	void manipulateBeforeRanking(std::vector<RankedCreature<PrimitiveSelection, SelectionRank>>& population) const
	{
	}

	void manipulateAfterRanking(std::vector<RankedCreature<PrimitiveSelection, SelectionRank>>& population) const
	{
		// Re-normalize scores and compute combined score. 
		SelectionRank max_r(-std::numeric_limits<double>::max()), min_r(std::numeric_limits<double>::max());

		for (auto& s : population)
		{
			max_r.size = std::max(max_r.size, s.rank.size);
			min_r.size = std::min(min_r.size, s.rank.size);
			max_r.geo = std::max(max_r.geo, s.rank.geo);
			min_r.geo = std::min(min_r.geo, s.rank.geo);
		}
		auto diff_r = max_r - min_r;

		for (auto& ps : population)
		{
			//ps.rank.combined = - ps.rank.geo * geo_weight - ps.rank.size * size_weight;

			// Normalized
			
			ps.rank.geo = ps.rank.geo < 0.0 || diff_r.geo == 0.0 ? 0.0 : (ps.rank.geo - min_r.geo) / diff_r.geo;
			ps.rank.size = ps.rank.size < 0.0 || diff_r.size == 0.0 ? 0.0 : (ps.rank.size - min_r.size) / diff_r.size;

			ps.rank.combined = (1.0 - ps.rank.geo) * geo_weight - ps.rank.size * size_weight;
			

			//std::cout << ps.creature << " : " << ps.rank;
		}
		//std::cout << "========================" << std::endl;
	}

	std::string info() const
	{
		return std::string();
	}

private:
	double geo_weight;
	double size_weight;

};

using SelectionTournamentSelector = TournamentSelector<RankedCreature<PrimitiveSelection, SelectionRank>>;
using SelectionIterationStopCriterion = NoFitnessIncreaseStopCriterion<RankedCreature<PrimitiveSelection, SelectionRank>, SelectionRank>;
using SelectionGA = GeneticAlgorithm<PrimitiveSelection, SelectionCreator, SelectionRanker, SelectionRank,
	SelectionTournamentSelector, SelectionIterationStopCriterion, SelectionPopMan>;
using NodeGA = GeneticAlgorithm<PrimitiveSelection, CSGNodeCreator, SelectionRanker, SelectionRank,
	SelectionTournamentSelector, SelectionIterationStopCriterion, SelectionPopMan>;


std::ostream& lmu::operator<<(std::ostream& out, const SelectionRank& r)
{
	std::cout << "combined: " << r.combined << " geo: " << r.geo << " size: " << r.size << std::endl;
	return out;
}

PrimitiveDecomposition lmu::decompose_primitives(const PrimitiveSet& primitives, const ModelSDF& model_sdf, double inside_t, double outside_t, double voxel_size)
{
	PrimitiveSet remaining_primitives, outside_primitives, inside_primitives;
	std::vector<CSGNode> outside_primitive_nodes;
	std::vector<CSGNode> inside_primitive_nodes;
	
	for (const auto& p : primitives)
	{
		switch (model_sdf.get_dh_type(p, inside_t, outside_t, voxel_size))
		{
		case DHType::INSIDE:
			inside_primitive_nodes.push_back(geometry(p.imFunc));
			inside_primitives.push_back(p);
			break;
		case DHType::OUTSIDE:
			outside_primitive_nodes.push_back(geometry(p.imFunc));
			outside_primitives.push_back(p);
			break;
		case DHType::NONE:
			remaining_primitives.push_back(p);
			break;
		}
	} 

	auto node = opNo();
	if (inside_primitive_nodes.empty() && outside_primitive_nodes.empty())
	{
		node = opNo();
	}
	else if (!inside_primitive_nodes.empty() && outside_primitive_nodes.empty())
	{
		node = inside_primitive_nodes.size() > 1 ? opUnion(inside_primitive_nodes) : inside_primitive_nodes[0];
	}
	else if (inside_primitive_nodes.empty() && !outside_primitive_nodes.empty())
	{
		node = opDiff({ opNo(),  outside_primitive_nodes.size() > 1 ? opUnion(outside_primitive_nodes) : outside_primitive_nodes[0] });
	}
	else
	{
		node = opDiff({ opUnion(inside_primitive_nodes), opUnion(outside_primitive_nodes) });
	}

	return PrimitiveDecomposition(node, remaining_primitives, inside_primitives, outside_primitives);
}

CSGNode lmu::integrate_node(const CSGNode& into, const PrimitiveSelection& s)
{
	auto res = into;
	std::vector<CSGNode> diff_prims;
	std::vector<CSGNode> union_prims;
	for (int i = 0; i < s.selection.size(); ++i)
	{
		if (s.selection[i].active)
		{
			if (s.selection[i].dh_type == DHType::OUTSIDE)
				diff_prims.push_back(geometry(s.prims->at(i).imFunc));
			else
				union_prims.push_back(geometry(s.prims->at(i).imFunc));
		}
	}

	switch (res.operationType())
	{
	case CSGNodeOperationType::Difference:

		if (res.childsCRef().size() == 2)
		{
			// Insert union nodes into left hand side. 
			if ((!union_prims.empty()))
			{
				if (res.childsRef()[0].operationType() == CSGNodeOperationType::Union)
				{
					res.childsRef()[0].childsRef().insert(res.childsRef()[0].childsRef().end(), union_prims.begin(), union_prims.end());
				}
				else if (res.childsRef()[0].operationType() == CSGNodeOperationType::Noop)
				{
					res.childsRef()[0] = union_prims.size() > 1 ? opUnion(union_prims) : union_prims[0];
				}
				else
				{
					auto n = opUnion({ res.childsRef()[0] });
					n.childsRef().insert(n.childsRef().end(), union_prims.begin(), union_prims.end());
					res.childsRef()[0] = n;
				}
			}

			// Insert difference nodes into right hand side.
			if ((!diff_prims.empty()))
			{
				if (res.childsRef()[1].operationType() == CSGNodeOperationType::Union)
				{
					res.childsRef()[1].childsRef().insert(res.childsRef()[1].childsRef().end(), diff_prims.begin(), diff_prims.end());
				}
				else
				{
					auto n = opUnion({ res.childsRef()[1] });
					n.childsRef().insert(n.childsRef().end(), diff_prims.begin(), diff_prims.end());
					res.childsRef()[1] = n;
				}
			}
		}
		else
		{
			std::cout << "Difference node is malformed." << std::endl;
		}

		break;

	case CSGNodeOperationType::Union:

		res.childsRef().insert(res.childsRef().end(), union_prims.begin(), union_prims.end());

		if (!diff_prims.empty())
		{
			res = opDiff({ res, opUnion(diff_prims) });
		}

		break;


	case CSGNodeOperationType::Noop:

		if (union_prims.empty() && !diff_prims.empty())
		{
			res = opDiff({ res, diff_prims.size() > 1 ? opUnion(diff_prims) : diff_prims[0] });
		}
		else if (!union_prims.empty() && diff_prims.empty())
		{		
			res = union_prims.size() > 1 ? opUnion(union_prims) : union_prims[0];
		}
		else if (!union_prims.empty() && !diff_prims.empty())
		{
			res = opDiff({ union_prims.size() > 1 ? opUnion(union_prims) : union_prims[0], diff_prims.size() > 1 ? opUnion(diff_prims) : diff_prims[0] });
		}

		break;

	default:
		std::cout << "Node configuration not supported." << std::endl;
		break;		
	}
	
	return res;
}

CSGNode lmu::integrate_node(const CSGNode& into, const CSGNode& node)
{
	if (into.operationType() == CSGNodeOperationType::Noop)
	{
		return node;
	} 
	else if (node.operationType() == CSGNodeOperationType::Noop)
	{
		return into;
	}
	else
	{
		if (into.operationType() == CSGNodeOperationType::Difference)
		{
			if (into.childsCRef()[0].operationType() == CSGNodeOperationType::Noop)
			{
				auto n = into;
				n.childsRef()[0] = node;
				return n;
			}
			else if (into.childsCRef()[0].operationType() == CSGNodeOperationType::Union)
			{
				auto n = into;
				n.childsRef()[0].childsRef().push_back(node);
				return n;
			}
			else 
			{
				auto n = into;
				n.childsRef()[0] = opUnion({node, n.childsRef()[0] });
				return n;
			}
		}
		else
		{
			return opUnion({ into, node });
		}
	}
}

CSGNode lmu::generate_csg_node(const PrimitiveDecomposition& decomposition, const std::shared_ptr<PrimitiveSetRanker>& primitive_ranker, const CSGNodeGenerationParams& params)
{
	int tournament_k = 2;
	int population_size = 150;
	double mut_prob = 0.4;
	double cross_prob = 0.4;
	double geo_weight = params.geo_weight;
	double size_weight = params.size_weight;

	SelectionTournamentSelector selector(2);
	SelectionIterationStopCriterion criterion(params.max_count, SelectionRank(0.00000001), params.max_iterations);

	auto primitives = decomposition.get_primitives(params.use_all_prims_for_ga);
	auto dh_types = decomposition.get_dh_types(params.use_all_prims_for_ga);
	auto start_node = params.use_all_prims_for_ga ? opNo() : decomposition.node;
	
	SelectionRanker ranker(primitive_ranker->model_sdf, start_node, params.creator_strategy);
	SelectionPopMan pop_man(geo_weight, size_weight);

	auto node = opNo();
	switch (params.creator_strategy)
	{
	case CreatorStrategy::SELECTION:
		{
			SelectionCreator selection_creator(PrimitiveSelection(&primitives, dh_types), primitive_ranker, params);
			SelectionGA ga;
			SelectionGA::Parameters ga_params(population_size, tournament_k, mut_prob, cross_prob, false, Schedule(), Schedule(), true);
			auto res = ga.run(ga_params, selector, selection_creator, ranker, criterion, pop_man);
			node = integrate_node(start_node, res.population[0].creature); 
			break;
		}
	case CreatorStrategy::NODE:
		{
			CSGNodeCreator node_creator(PrimitiveSelection(&primitives, dh_types), primitive_ranker, params);
			NodeGA ga;
			NodeGA::Parameters ga_params(population_size, tournament_k, mut_prob, cross_prob, false, Schedule(), Schedule(), true);
			auto res = ga.run(ga_params, selector, node_creator, ranker, criterion, pop_man);
			node = integrate_node(start_node, res.population[0].creature.node);
			break;
		}
	}
	writeNode(node, "node_node.gv");
		
	//node = lmu::remove_redundancies(node, 0.01, lmu::PointCloud());

	return node;
}

lmu::SelectionRank::SelectionRank(double v) :
	geo(v),
	size(v),
	combined(v)
{
}

lmu::SelectionRank::SelectionRank(double geo, double size, double combined) :
	geo(geo),
	size(size),
	combined(combined)
{
}

lmu::CSGNodeGenerationParams::CSGNodeGenerationParams()
{
}

lmu::CSGNodeGenerationParams::CSGNodeGenerationParams(double create_new_prob, double active_prob, bool use_prim_geo_scores_as_active_prob,
	double dh_type_prob, bool evolve_dh_type, bool use_all_prims_for_ga, int max_tree_depth, double subtree_prob, CreatorStrategy creator_strategy) :
	create_new_prob(create_new_prob),
	active_prob(active_prob),
	use_prim_geo_scores_as_active_prob(use_prim_geo_scores_as_active_prob),
	dh_type_prob(dh_type_prob),
	evolve_dh_type(evolve_dh_type),
	use_all_prims_for_ga(use_all_prims_for_ga),
	max_tree_depth(max_tree_depth),
	subtree_prob(subtree_prob),
	creator_strategy(creator_strategy)
{
}

lmu::PrimitiveDecomposition::PrimitiveDecomposition(const CSGNode& node, const PrimitiveSet& rem_prims, const PrimitiveSet& in_prims, const PrimitiveSet& out_prims) :
	node(node),
	remaining_primitives(rem_prims),
	inside_primitives(in_prims),
	outside_primitives(out_prims)
{
}

PrimitiveSet lmu::PrimitiveDecomposition::get_primitives(bool all) const
{	
	auto all_prims = remaining_primitives;

	if(all)
	{
		all_prims.insert(all_prims.end(), inside_primitives.begin(), inside_primitives.end());
		all_prims.insert(all_prims.end(), outside_primitives.begin(), outside_primitives.end());
	}
	
	return all_prims;
}

std::vector<DHType> lmu::PrimitiveDecomposition::get_dh_types(bool all) const
{
	std::vector<DHType> dh_types;

	std::transform(remaining_primitives.begin(), remaining_primitives.end(), std::back_inserter(dh_types),
		[](const auto& p) -> DHType { return DHType::NONE; });

	if (all)
	{
		std::transform(inside_primitives.begin(), inside_primitives.end(), std::back_inserter(dh_types),
			[](const auto& p) -> DHType { return DHType::INSIDE; });

		std::transform(outside_primitives.begin(), outside_primitives.end(), std::back_inserter(dh_types),
			[](const auto& p) -> DHType { return DHType::OUTSIDE; });		
	}
	
	return dh_types;
}
