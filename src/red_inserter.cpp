#include "red_inserter.h"

#include <random>
#include <algorithm>
#include "csgnode.h"
#include "csgnode_helper.h"
#include "optimizer_ga.h"

using namespace lmu;

std::ostream& lmu::operator <<(std::ostream& stream, const lmu::InserterType& it)
{
	std::string it_str;
	switch (it)
	{
	case InserterType::SubtreeCopy:
		it_str = "Subtree Copy";
		break;
	case InserterType::DoubleNegation:
		it_str = "Double Negation";
		break;
	case InserterType::Distributive:
		it_str = "Distributive";
		break;
	case InserterType::Absorption:
		it_str = "Absorption";
		break;
	case InserterType::GA:
		it_str = "GA";
		break;
	}

	stream << it_str;

	return stream;
}

Inserter lmu::inserter(InserterType type, double probability)
{
	switch (type)
	{
	case InserterType::SubtreeCopy:
		return Inserter(std::make_shared<SubtreeCopyInserter>(), probability);
	case InserterType::DoubleNegation:
		return Inserter(std::make_shared<DoubleNegationInserter>(), probability);
	case InserterType::Distributive:
		return Inserter(std::make_shared<DistributiveInserter>(), probability);
	case InserterType::Absorption:
		return Inserter(std::make_shared<AbsorptionInserter>(), probability);
	case InserterType::GA:
		return Inserter(std::make_shared<GAInserter>(), probability);
	}
}

CSGNode lmu::inflate_node(const CSGNode& node, int iterations, const std::vector<Inserter>& inserter)
{
	CSGNode inflated_node = node;

	std::vector<double> probs;
	std::transform(inserter.begin(), inserter.end(), std::back_inserter(probs), [](const Inserter& ins) {return ins.propability(); });
	
	std::default_random_engine generator;
	std::discrete_distribution<int> distribution(probs.begin(), probs.end());
	
	for (int i = 0; i < iterations; ++i)
	{
		int inserter_index = distribution(generator);
		int node_idx = std::uniform_int_distribution<int>(0, numNodes(inflated_node) - 1)(generator);

		CSGNode* sub_node = nodePtrAt(inflated_node, node_idx);

		bool worked = inserter[inserter_index].inflate(*sub_node);

		std::cout << "Applied " << inserter[inserter_index].type() << " at node idx " << node_idx << "." << std::endl;
		if (!worked)
			std::cout << "No effect." << std::endl;
	}

	return inflated_node;
}


bool lmu::SubtreeCopyInserter::inflate(CSGNode & node) const
{
	static std::default_random_engine generator;
	
	if (std::bernoulli_distribution(0.5)(generator))
	{
		node = opUnion({ node, node });
	}
	else
	{
		node = opInter({ node, node });
	}

	return true;
}

std::shared_ptr<IInserter> lmu::SubtreeCopyInserter::clone() const
{
	return std::make_shared<SubtreeCopyInserter>(*this);
}

InserterType lmu::SubtreeCopyInserter::type() const
{
	return InserterType::SubtreeCopy;
}


bool lmu::DoubleNegationInserter::inflate(CSGNode & node) const
{	
	node = opComp({ opComp({ node }) });

	return true;
}

std::shared_ptr<IInserter> lmu::DoubleNegationInserter::clone() const
{
	return std::make_shared<DoubleNegationInserter>(*this);
}

InserterType lmu::DoubleNegationInserter::type() const
{
	return InserterType::DoubleNegation;
}


bool lmu::DistributiveInserter::inflate(CSGNode& node) const
{
	static std::default_random_engine generator;
	const auto& childs = node.childsCRef();

	bool did_something = false;

	if (node.type() == CSGNodeType::Operation)
	{
		switch (node.operationType())
		{
		case CSGNodeOperationType::Intersection:
		{
			std::vector<int> union_indices;
			for (int i = 0; i < childs.size(); ++i)
				if (childs[i].type() == CSGNodeType::Operation && 
					childs[i].operationType() == CSGNodeOperationType::Union)
					union_indices.push_back(i);
				
			if (!union_indices.empty())
			{
				int union_idx = union_indices[std::uniform_int_distribution<int>(0, union_indices.size() - 1)(generator)];
				int other_idx = union_idx == 1 ? 0 : 1;
				
				if (childs[union_idx].childsCRef().size() >= 2)
				{
					node = opUnion({ opInter({childs[union_idx].childsCRef()[0], childs[other_idx]}),
						 opInter({ childs[union_idx].childsCRef()[1], childs[other_idx] }) });
					did_something = true;
				}
			}

			break;
		}
		case CSGNodeOperationType::Union:
		{
			std::vector<int> inter_indices;
			for (int i = 0; i < childs.size(); ++i)
				if (childs[i].type() == CSGNodeType::Operation &&
					childs[i].operationType() == CSGNodeOperationType::Intersection)
					inter_indices.push_back(i);

			if (!inter_indices.empty())
			{
				int inter_idx = inter_indices[std::uniform_int_distribution<int>(0, inter_indices.size() - 1)(generator)];
				int other_idx = inter_idx == 1 ? 0 : 1;

				if (childs[inter_idx].childsCRef().size() >= 2)
				{
					node = opInter({ opUnion({ childs[inter_idx].childsCRef()[0], childs[other_idx] }),
						opUnion({ childs[inter_idx].childsCRef()[1], childs[other_idx] }) });
					did_something = true;
				}
			}

			break;
		}
		}
	}

	return did_something;
}

std::shared_ptr<IInserter> lmu::DistributiveInserter::clone() const
{
	return std::make_shared<DistributiveInserter>(*this);
}

InserterType lmu::DistributiveInserter::type() const
{
	return InserterType::Distributive;
}

bool lmu::AbsorptionInserter::inflate(CSGNode& node) const
{
	static std::default_random_engine generator;
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;
	
	int node_idx = du(generator, parmu_t{ 0, numNodes(node) - 1 });
	CSGNode* sub_node = nodePtrAt(node, node_idx);
	
	if (std::bernoulli_distribution(0.5)(generator))
	{
		node = opUnion({ node, opInter({node, *sub_node }) });
	}
	else
	{
		node = opInter({ node, opUnion({ node, *sub_node }) });
	}

	return true;
}

std::shared_ptr<IInserter> lmu::AbsorptionInserter::clone() const
{
	return std::make_shared<AbsorptionInserter>(*this);
}

InserterType lmu::AbsorptionInserter::type() const
{
	return InserterType::Absorption;
}

bool lmu::GAInserter::inflate(CSGNode& node) const
{
	OptimizerGAParams params;

	params.ranker_params.geo_score_weight = 20.0;
	params.ranker_params.size_score_weight = -0.1; /*we want larger trees.*/
	params.ranker_params.prox_score_weight = -0.1; /*...and also trees with bad proxy score.*/
	
	params.ranker_params.gradient_step_size = 0.0001;
	params.ranker_params.position_tolerance = 0.1;
	params.ranker_params.sampling_params.errorSigma = 0.00000001;
	params.ranker_params.sampling_params.samplingStepSize = 0.1;
	params.ranker_params.sampling_params.maxDistance = 0.1;
	params.ranker_params.max_sampling_points = 250;
	params.ranker_params.geo_score_strat = GeoScoreStrategy::IN_OUT_SAMPLES;

	params.creator_params.create_new_prob = 0.3;
	params.creator_params.subtree_prob = 0.3;
	params.creator_params.initial_population_dist = { 0.1,0.8,0.1 };

	params.man_params.max_delta = 0.2;

	params.ga_params.crossover_rate = 0.4;
	params.ga_params.mutation_rate = 0.3;
	params.ga_params.in_parallel = true;
	params.ga_params.max_iterations = 10;
	params.ga_params.num_best_parents = 2;
	params.ga_params.population_size = 100;
	params.ga_params.tournament_k = 2;
	params.ga_params.use_caching = true;	

	
	auto res = optimize_with_ga(node, params, std::cout);

	double max_grow_percentage = 1.5;
	double min_grow_percentage = 1.2;

	for (const auto& n : res.pareto_nodes)
	{
		double percentage = (double)numNodes(n) / (double)numNodes(node);
		if (percentage >= min_grow_percentage && percentage <= max_grow_percentage)
		{
			node = n; 
			return true;
		}
	}

	return false;
}

std::shared_ptr<IInserter> lmu::GAInserter::clone() const
{
	return std::make_shared<GAInserter>(*this);
}

InserterType lmu::GAInserter::type() const
{
	return InserterType::GA;
}