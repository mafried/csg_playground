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
	prims(primitives)
{
	std::default_random_engine rnd_engine;
	std::random_device rnd_device;
	rnd_engine.seed(rnd_device());

	std::bernoulli_distribution d{};
	using parm_t = decltype(d)::param_type;

	
	for (const auto& p : *primitives)
	{
		bool active = d(rnd_engine, parm_t{ 0.5 });
		DHType dh_type = d(rnd_engine, parm_t{ 0.5 }) ? DHType::INSIDE : DHType::OUTSIDE;

		selection.push_back(SelectionValue(dh_type, active));
	}	
}

lmu::PrimitiveSelection::PrimitiveSelection(const PrimitiveSet* primitives, const ModelSDF& model_sdf, double t_inside, double t_outside) :
	prims(primitives)
{
	for (const auto& p : *prims)
	{
		auto da_type = model_sdf.get_dh_type(p, t_inside, t_outside);
		selection.push_back(SelectionValue(da_type, true));

		std::cout << "Primitive " << p.imFunc->name() << ": " << (int)da_type << std::endl;
	}
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
	
	return opDiff({ opUnion(union_prims), opUnion(diff_prims) });
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
	for (const auto& s : selection)
	{
		boost::hash_combine(seed, s.dh_type);
		boost::hash_combine(seed, s.active);
	}

	return seed;
}

std::ostream& lmu::operator<<(std::ostream& out, const PrimitiveSelection& ps)
{
	int i = 0;
	for (const auto& s : ps.selection)
	{
		out << ps.prims->at(i++).imFunc->name() << ": " << s.active << " " << (s.dh_type == DHType::INSIDE ? "IN" : s.dh_type == DHType::OUTSIDE ? "OUT" : "NONE") << " | ";
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

//////////////////////////// RANKER ////////////////////////////

struct SelectionRanker
{
	SelectionRanker(const std::shared_ptr<ModelSDF>& model_sdf, const CSGNode& start_node) :
		model_sdf(model_sdf),
		start_node(start_node)
	{
	}

	SelectionRank rank(const PrimitiveSelection& s) const
	{
		//static int counter = 0;
		//std::cout << "counter: " << counter << std::endl;
		//counter++;

		auto n = integrate_node(start_node, s);

		auto d = 0.0;

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


		auto sr = SelectionRank(d, (double)s.get_num_active(), 0.0);

		//std::cout << s << " SCORE: " << sr << std::endl;
		
		return sr;
	}

	std::string info() const
	{
		return std::string();
	}

private:

	std::shared_ptr<ModelSDF> model_sdf;
	CSGNode start_node;
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


std::ostream& lmu::operator<<(std::ostream& out, const SelectionRank& r)
{
	std::cout << "combined: " << r.combined << " geo: " << r.geo << " size: " << r.size << std::endl;
	return out;
}

PrimitiveDecomposition lmu::decompose_primitives(const PrimitiveSet& primitives, const ModelSDF& model_sdf, double inside_t, double outside_t, double voxel_size)
{
	PrimitiveSet remaining_primitives;
	std::vector<CSGNode> outside_primitives;
	std::vector<CSGNode> inside_primitives;
	
	for (const auto& p : primitives)
	{
		switch (model_sdf.get_dh_type(p, inside_t, outside_t, voxel_size))
		{
		case DHType::INSIDE:
			inside_primitives.push_back(geometry(p.imFunc));
			break;
		case DHType::OUTSIDE:
			outside_primitives.push_back(geometry(p.imFunc));
			break;
		case DHType::NONE:
			remaining_primitives.push_back(p);
			break;
		}
	} 

	auto node = opNo();
	auto left = inside_primitives.size() > 1 ? opUnion(inside_primitives) : inside_primitives.empty()? opNo() : inside_primitives[0];
	auto right = outside_primitives.size() > 1 ? opUnion(outside_primitives) : outside_primitives.empty() ? opNo() : outside_primitives[0];	
	
	if (right.operationType() == CSGNodeOperationType::Noop)
	{
		node = left;
	}
	else
	{
		node = opDiff({ left, right });
	}

	return PrimitiveDecomposition(node, remaining_primitives);
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

	if (res.type() == CSGNodeType::Operation)
	{
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

		default:
			std::cout << "Node configuration not supported." << std::endl;
			break;
		}
	}
	else 
	{
		if (union_prims.empty() && diff_prims.empty())
		{
			// nothing
		}
		else if (union_prims.empty())
		{
			res = opDiff({ res, opUnion(diff_prims) });
		}
		else if (diff_prims.empty())
		{
			res = opUnion({ res });
			res.childsRef().insert(res.childsRef().end(), union_prims.begin(), union_prims.end());
		}
	}

	return res;
}

CSGNode lmu::generate_csg_node(const PrimitiveSet& primitives, const CSGNode& start_node, const std::shared_ptr<PrimitiveSetRanker>& primitive_ranker, const CSGNodeGenerationParams& params)
{
	//if (primitives.empty())
	//	return lmu::opNo();
	//if (primitives.size() == 1)
	//	return lmu::geometry(primitives[0].imFunc);

	SelectionTournamentSelector selector(2);
	SelectionIterationStopCriterion criterion(1000, SelectionRank(0.00001), 1000);
	
	SelectionCreator creator(params.evolve_dh_type ? PrimitiveSelection(&primitives) : PrimitiveSelection(&primitives, *primitive_ranker->model_sdf, 0.9, 0.1), primitive_ranker, params);
	
	SelectionRanker ranker(primitive_ranker->model_sdf, start_node);
	SelectionPopMan pop_man(1, 0.0);

	SelectionGA ga;
	SelectionGA::Parameters ga_params(150, 2, 0.4, 0.4, false, Schedule(), Schedule(), true);

	auto res = ga.run(ga_params, selector, creator, ranker, criterion, pop_man);

	std::cout << "RESULT: " << res.population[0].creature << std::endl;
	
	auto node = integrate_node(start_node, res.population[0].creature);

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

lmu::CSGNodeGenerationParams::CSGNodeGenerationParams(double create_new_prob, double active_prob, bool use_prim_geo_scores_as_active_prob, double dh_type_prob, bool evolve_dh_type) :
	create_new_prob(create_new_prob),
	active_prob(active_prob),
	use_prim_geo_scores_as_active_prob(use_prim_geo_scores_as_active_prob),
	dh_type_prob(dh_type_prob),
	evolve_dh_type(evolve_dh_type)
{
}

lmu::PrimitiveDecomposition::PrimitiveDecomposition(const CSGNode & node, const PrimitiveSet & rem_prims) : 
	node(node),
	remaining_primitives(rem_prims)
{
}
