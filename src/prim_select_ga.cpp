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
	size_t s = 0;
	for (const auto& se : selection) 
	{
		s += se.hash(seed);
	}
	return s;
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
	SelectionCreator(const lmu::PrimitiveSelection& ps, const CSGNodeGenerationParams& params) :
		ps(ps),
		params(params)
	{
		_rndEngine.seed(_rndDevice());
	}

	lmu::PrimitiveSelection mutate(const lmu::PrimitiveSelection& s) const
	{
		static std::bernoulli_distribution d{};
		using parm_t = decltype(d)::param_type;

		static std::uniform_int_distribution<> du{};
		using parmu_t = decltype(du)::param_type;


		if (d(_rndEngine, parm_t{ params.create_new_prob }))
		{
			return create();
		}
		else
		{
			int num_selections = ps.selection.size();

			int s_idx = du(_rndEngine, parmu_t{ 0, num_selections - 1 });

			auto new_s = s;
			
			new_s.selection[s_idx].active = !new_s.selection[s_idx].active;
			if (params.evolve_dh_type)
			{
				new_s.selection[s_idx].dh_type = new_s.selection[s_idx].dh_type == DHType::INSIDE ? DHType::OUTSIDE : DHType::INSIDE;
			}
			
			return new_s;
		}
	}

	std::vector<PrimitiveSelection> crossover(const PrimitiveSelection& s1, const PrimitiveSelection& s2) const
	{
		int num_selections = ps.selection.size();

		auto new_s1 = s1;
		auto new_s2 = s2;

		static std::uniform_int_distribution<> du{};
		using parmu_t = decltype(du)::param_type;

		int x1 = du(_rndEngine, parmu_t{ 0, num_selections - 1 });
		int x2 = du(_rndEngine, parmu_t{ 0, num_selections - 1 });
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
			new_ps.selection[i].active = d(_rndEngine, parm_t{ params.active_prob });

			if (params.evolve_dh_type)
				new_ps.selection[i].dh_type = d(_rndEngine, parm_t{ params.dh_type_prob }) ? DHType::INSIDE : DHType::OUTSIDE;
		}

		return new_ps;
	}

	std::string info() const
	{
		return std::string();
	}

	private:

		lmu::PrimitiveSelection ps;
	
		CSGNodeGenerationParams params;

		mutable std::default_random_engine _rndEngine;
		mutable std::random_device _rndDevice;
};

//////////////////////////// RANKER ////////////////////////////

struct SelectionRanker
{
	SelectionRanker(const std::shared_ptr<ModelSDF>& model_sdf) :
		model_sdf(model_sdf)
	{
	}

	SelectionRank rank(const PrimitiveSelection& s) const
	{
		auto n = s.to_node();

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
		return SelectionRank(d, (double)s.get_num_active(), 0.0);
	}

	std::string info() const
	{
		return std::string();
	}

private:

	std::shared_ptr<ModelSDF> model_sdf;
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

CSGNode lmu::generate_csg_node(const PrimitiveSet& primitives, const std::shared_ptr<ModelSDF>& model_sdf, const CSGNodeGenerationParams& params)
{
	if (primitives.empty())
		return lmu::opNo();
	if (primitives.size() == 1)
		return lmu::geometry(primitives[0].imFunc);

	SelectionTournamentSelector selector(2);
	SelectionIterationStopCriterion criterion(500, SelectionRank(0.00001), 500);
	//SelectionCreator creator(params.evolve_dh_type ? PrimitiveSelection(&primitives) : PrimitiveSelection(&primitives, *model_sdf, 0.9, 0.1), params);
	SelectionCreator creator(PrimitiveSelection(&primitives, *model_sdf, 0.9, 0.1), params);

	SelectionRanker ranker(model_sdf);
	SelectionPopMan pop_man(1, 0.0);

	SelectionGA ga;
	SelectionGA::Parameters ga_params(50, 2, 0.4, 0.4, false, Schedule(), Schedule(), true);

	auto res = ga.run(ga_params, selector, creator, ranker, criterion, pop_man);

	auto node = res.population[0].creature.to_node();

	node = lmu::remove_redundancies(node, 0.01, lmu::PointCloud());

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

lmu::CSGNodeGenerationParams::CSGNodeGenerationParams(double create_new_prob, double active_prob, double dh_type_prob, bool evolve_dh_type) :
	create_new_prob(create_new_prob),
	active_prob(active_prob),
	dh_type_prob(dh_type_prob),
	evolve_dh_type(evolve_dh_type)
{
}
