#include "optimizer_ga.h"
#include "optimizer_red.h"
#include "evolution.h"
#include "pointcloud.h"
#include "csgnode_helper.h"

using namespace lmu;

struct CSGNodeCreator;
struct CSGNodeRanker;

using Rank = double;
using CSGNodeTournamentSelector = TournamentSelector<RankedCreature<CSGNode, Rank>>;
using CSGNodeIterationStopCriterion = IterationStopCriterion<RankedCreature<CSGNode, Rank>>;
using CSGNodeNoFitnessIncreaseStopCriterion = NoFitnessIncreaseStopCriterion<RankedCreature<CSGNode, Rank>, Rank>;
using CSGNodeGA = GeneticAlgorithm<CSGNode, CSGNodeCreator, CSGNodeRanker, Rank, CSGNodeTournamentSelector, CSGNodeIterationStopCriterion>;

//////////////////////////// RANKER ////////////////////////////

struct CSGNodeRanker
{
	CSGNodeRanker(const CSGNode& input_node, const RankerParams& params) :
		input_node(input_node),
		input_node_pc(lmu::farthestPointSampling(lmu::computePointCloud(input_node, params.sampling_params), params.max_sampling_points)),
		input_node_size(numNodes(input_node)),
		params(params)
	{
		input_node_geo_score = compute_geo_score(input_node);
	}

	Rank rank(const CSGNode& node) const
	{
		auto geo_score = compute_geo_score(node) / input_node_geo_score;

		auto prox_score = compute_local_proximity_score(node, params.sampling_params.samplingStepSize);

		auto size_score = 1.0 - ((double)numNodes(node) / (double)input_node_size);
		
		std::cout << "GEO: " << geo_score << " PROXIMITY:" << prox_score << " SIZE: " << size_score << std::endl;

		return Rank(
			params.geo_score_weight * geo_score + 
			params.prox_score_weight * prox_score + 
			params.size_score_weight * size_score);
	}

	std::string info() const
	{
		return std::string();
	}

private:

	double compute_geo_score(const CSGNode& node) const
	{
		double numCorrectSamples = 0.0;
		double numConsideredSamples = (double)input_node_pc.rows();

		for (int i = 0; i < input_node_pc.rows(); ++i)
		{

			Eigen::Matrix<double, 1, 6> pn = input_node_pc.row(i);
			Eigen::Vector3d sampleP = pn.leftCols(3);
			Eigen::Vector3d sampleN = pn.rightCols(3);

			Eigen::Vector4d sampleDistGradNode = node.signedDistanceAndGradient(sampleP, params.gradient_step_size);
			double sampleDistNode = sampleDistGradNode[0];
			Eigen::Vector3d sampleGradNode = sampleDistGradNode.bottomRows(3);

			double inc = std::abs(sampleDistNode) <= params.position_tolerance && sampleN.dot(sampleGradNode) >= 0.0;
			numCorrectSamples += inc;
		}

		return (numCorrectSamples / numConsideredSamples);
	}

	CSGNode input_node;
	int input_node_size;
	double input_node_geo_score;
	lmu::PointCloud input_node_pc;
	RankerParams params;
};

//////////////////////////// CREATOR ////////////////////////////

struct CSGNodeCreator
{
	CSGNodeCreator(const CSGNode& input_node, const CreatorParams& params) :
		primitives(lmu::allDistinctFunctions(input_node)),
		params(params),
		max_tree_depth(depth(input_node))
	{
	}

	CSGNode mutate(const CSGNode& node) const
	{
		static std::bernoulli_distribution d{};
		using parm_t = decltype(d)::param_type;

		static std::uniform_int_distribution<> du{};
		using parmu_t = decltype(du)::param_type;

		static std::uniform_real_distribution<double> dur(-0.1, 0.1);
		using parmur_t = decltype(dur)::param_type;

		if (d(_rndEngine, parm_t{ params.create_new_prob }))
		{
			return create(max_tree_depth);
		}
		else
		{
			int nodeIdx = du(_rndEngine, parmu_t{ 0, numNodes(node) - 1 });

			auto newNode = node;

			CSGNode* subNode = nodePtrAt(newNode, nodeIdx);

			create(*subNode, max_tree_depth, 0);

			return newNode;
		}
	}

	std::vector<CSGNode> crossover(const CSGNode& node1, const CSGNode& node2) const
	{
		if (!node1.isValid() || !node2.isValid())
			return std::vector<lmu::CSGNode> {node1, node2};

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

		return std::vector<lmu::CSGNode>
		{
			newNode1, newNode2
		};
	}

	CSGNode create() const
	{
		return create(max_tree_depth);
	}

	CSGNode create(int maxDepth) const
	{
		auto node = CSGNode::invalidNode;
		create(node, maxDepth, 0);
		return node;
	}

	std::string info() const
	{
		return std::string();
	}

private:
	
	void create(CSGNode& node, int maxDepth, int curDepth) const
	{
		static std::bernoulli_distribution db{};
		using parmb_t = decltype(db)::param_type;
		
		if (curDepth >= maxDepth)
		{
			node = create_rnd_primitive_node();
		}
		else
		{
			if (db(_rndEngine, parmb_t{ params.subtree_prob}))
			{
				node = create_rnd_operation_node();

				auto numAllowedChilds = node.numAllowedChilds();
				int numChilds = clamp(std::get<1>(numAllowedChilds), std::get<0>(numAllowedChilds), 2); //2 is the maximum number of childs allowed for create

				for (int i = 0; i < numChilds; ++i)
				{
					auto child = CSGNode::invalidNode;
					create(child, maxDepth, curDepth + 1);
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

		int funcIdx = du(_rndEngine, parmu_t{ 0, static_cast<int>(primitives.size() - 1) });
		return geometry(primitives[funcIdx]);
	}

	CSGNode create_rnd_operation_node() const
	{
		std::discrete_distribution<> d({ 1, 1, 1, 1 });
		int op = d(_rndEngine) + 1; //0 is OperationType::Unknown, 6 is OperationType::Invalid.

		return createOperation(static_cast<CSGNodeOperationType>(op));
	}
	
	CreatorParams params;
	int max_tree_depth;
	std::vector<ImplicitFunctionPtr> primitives;
	mutable std::default_random_engine _rndEngine;
	mutable std::random_device _rndDevice;
};

bool is_simple_case(const CSGNode& node)
{
	return node.childsCRef().empty();
}

CSGNode simply_optimize(const CSGNode& node)
{
	return node;
}

OptimizerGAResult lmu::optimize_with_ga(const CSGNode& node, const OptimizerGAParams& params, std::ostream& report_stream)
{	
	if (is_simple_case(node))
		return simply_optimize(node);

	CSGNodeRanker ranker(node, params.ranker_params);
	CSGNodeCreator creator(node, params.creator_params);
	CSGNodeIterationStopCriterion stop_criterion(params.ga_params.max_iterations);
	CSGNodeTournamentSelector t_selector(params.ga_params.tournament_k);

	CSGNodeGA::Parameters ga_params(params.ga_params.population_size, params.ga_params.num_best_parents,
		params.ga_params.mutation_rate, params.ga_params.crossover_rate, params.ga_params.in_parallel,
		Schedule(), Schedule(), params.ga_params.use_caching);

	CSGNodeGA ga;
	auto res = ga.run(ga_params, t_selector, creator, ranker, stop_criterion);

	auto opt_res = OptimizerGAResult(res.population[0].creature);

	res.statistics.save(report_stream, &opt_res.node);

	ranker.rank(opt_res.node);

	return opt_res;
}

void compute_local_proximity_score_rec(const CSGNode& node, double sampling_grid_size, double& score, EmptySetLookup& esLookup)
{
	auto n = node.childsCRef().size();

	if (score == invalid_proximity_score || n == 0)
	{
		return;
	}	
	else if (n > 2)
	{
		score = invalid_proximity_score;
		std::cerr << "Proximity score computation: No more than 2 children allowed." << std::endl;
		return;
	}
	else if (n == 1)
	{
		score += 1.0;
		compute_local_proximity_score_rec(node.childsCRef()[0], sampling_grid_size, score, esLookup);
	}
	else if (n == 2)
	{
		const auto& left = node.childsCRef()[0];
		const auto& right = node.childsCRef()[1];

		score += is_empty_set(opInter({ left, right }), sampling_grid_size, esLookup) ? 0.0 : 1.0;
		
		compute_local_proximity_score_rec(left, sampling_grid_size, score, esLookup);
		compute_local_proximity_score_rec(right, sampling_grid_size, score, esLookup);
	}
}

double lmu::compute_local_proximity_score(const CSGNode& node, double sampling_grid_size)
{
	EmptySetLookup esLookup;
	double score = 0.0;

	compute_local_proximity_score_rec(node, sampling_grid_size, score, esLookup);

	//std::cout << "NUM: " << (double)numNodes(node, true) << std::endl;
	//std::cout << "SCORE: " << score;

	return score == invalid_proximity_score ? score : score / (double)numNodes(node, true);
}
