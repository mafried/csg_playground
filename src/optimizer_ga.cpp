#include "optimizer_ga.h"
#include "optimizer_red.h"
#include "evolution.h"
#include "pointcloud.h"
#include "csgnode_helper.h"
#include "cit.h"

using namespace lmu;

struct CSGNodeCreator;
struct CSGNodeRanker;

//using Rank = double;
struct Rank
{
	Rank(double score, double geo_score, double prox_score, double size_score, double size_diff_score) :
		score(score),
		geo_score(geo_score),
		prox_score(prox_score),
		size_score(size_score),
		size_diff_score(size_diff_score)
	{
	}

	explicit Rank(double score = 0.0) :
		score(score),
		geo_score(score),
		prox_score(score),
		size_score(score),
		size_diff_score(score)

	{
	}
	
	double score;
	double geo_score; 
	double prox_score;
	double size_score;
	double size_diff_score;

	operator double() const { return score; }

	//friend inline bool operator< (const Rank& lhs, const Rank& rhs) { return lhs.score < rhs.score; }
	//friend inline bool operator> (const Rank& lhs, const Rank& rhs) { return rhs < lhs; }
	//friend inline bool operator<=(const Rank& lhs, const Rank& rhs) { return !(lhs > rhs); }
	//friend inline bool operator>=(const Rank& lhs, const Rank& rhs) { return !(lhs < rhs); }
	//friend inline bool operator==(const Rank& lhs, const Rank& rhs) { return lhs.score == rhs.score; }
	//friend inline bool operator!=(const Rank& lhs, const Rank& rhs) { return !(lhs == rhs); }
};

std::ostream& operator<<(std::ostream& out, const Rank& r)
{
	out << "score: " << r.score << " geo: " << r.geo_score << " prox: " << r.prox_score << " size: " << r.size_score << " size diff: " << r.size_diff_score;
	return out;
}

//////////////////////////// CREATOR ////////////////////////////

struct CSGNodeCreator
{
	CSGNodeCreator(const CSGNode& input_node, const std::vector<ImplicitFunctionPtr>& primitives, const CreatorParams& params) :
		input_node(input_node),
		primitives(primitives.empty() ? lmu::allDistinctFunctions(input_node) : primitives),
		params(params),
		max_tree_depth(depth(input_node))
	{
		_rndEngine.seed(_rndDevice());
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
		auto node = CSGNode::invalidNode;

		std::discrete_distribution<> d(params.initial_population_dist.begin(), params.initial_population_dist.end());
		switch (d(_rndEngine))
		{
		case 0:
			create(node, max_tree_depth, 0);
			//std::cout << "create" << std::endl;
			break;
		case 1:
			node = input_node;
			//std::cout << "take input node." << std::endl;
			break;
		case 2:
			node = mutate(input_node);
			//std::cout << "take mutated input node." << std::endl;
			break;
		default:
			std::cout << "Creator: No strategy for CSGNode creation available." << std::endl;
			break;
		}

		return node;
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
			if (db(_rndEngine, parmb_t{ params.subtree_prob }))
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
	CSGNode input_node;

	mutable std::default_random_engine _rndEngine;
	mutable std::random_device _rndDevice;
};

//////////////////////////// RANKER ////////////////////////////

struct CSGNodeRanker
{
	CSGNodeRanker(const CSGNode& input_node, const std::vector<ImplicitFunctionPtr>& primitives, 
		const RankerParams& params) :
		input_node(input_node),
		input_node_size(numNodes(input_node)),
		params(params)
	{
		switch (params.geo_score_strat)
		{
		case GeoScoreStrategy::SURFACE_SAMPLES:
			surface_pc = lmu::farthestPointSampling(
				lmu::computePointCloud(input_node, params.sampling_params), params.max_sampling_points);

			in_pc = empty_pc();
			out_pc = empty_pc();
			in_out_pc = empty_pc();
			break;
		case GeoScoreStrategy::IN_OUT_SAMPLES:
			surface_pc = empty_pc();

			auto out_cits = generate_cits(input_node, params.sampling_params.samplingStepSize,
				CITSGenerationOptions::OUTSIDE, primitives);
			out_pc = extract_points_from_cits(
				out_cits);

			auto in_cits = generate_cits(input_node, params.sampling_params.samplingStepSize,
				CITSGenerationOptions::INSIDE, primitives);
			in_pc = extract_points_from_cits(
				in_cits);

			toJSONFile(DNFtoCSGNode(out_cits.dnf), "outside_cits.json");
			toJSONFile(DNFtoCSGNode(in_cits.dnf), "inside_cits.json");

			//==================

			/*
			toJSONFile(DNFtoCSGNode(in_cits.dnf), "inside_cits.json");
			std::cout << "POINT CHECKS IN: " << std::endl;
			for (int i = 0; i < in_pc.rows(); ++i)
			{
				Eigen::Vector3d p = in_pc.row(i).leftCols(3).transpose();
				std::cout << DNFtoCSGNode(in_cits.dnf).signedDistance(p) << std::endl;
			}
			std::cout << "POINT CHECKS OUT: " << std::endl;
			for (int i = 0; i < out_pc.rows(); ++i)
			{
				Eigen::Vector3d p = out_pc.row(i).leftCols(3).transpose();
				std::cout << DNFtoCSGNode(out_cits.dnf).signedDistance(p) << std::endl;
			}
			*/
			//==================

			
			in_out_pc = mergePointClouds({ in_pc, out_pc });

			break;
		}

		input_node_geo_score = compute_geo_score(input_node);
	}

	Rank rank(const CSGNode& node) const
	{
		//std::cout << "INPUT GEO SCORE: " << input_node_geo_score << std::endl;

		auto geo_score = compute_geo_score(node) / input_node_geo_score;

		auto prox_score = params.prox_score_weight == 0.0 ? 0.0 : 
			compute_local_proximity_score(node, params.sampling_params.samplingStepSize, in_out_pc);

		auto size_score = (double)numNodes(node);

		auto size_diff_score = (double)input_node_size - size_score;
		
		//std::cout << "GEO: " << geo_score << " PROXIMITY:" << prox_score << " SIZE: " << size_score << std::endl;

		return Rank(0.0, geo_score, prox_score, size_score, size_diff_score);
	}

	std::string info() const
	{
		return std::string();
	}

	RankerParams params;

private:

	double compute_geo_score(const CSGNode& node) const
	{
		switch (params.geo_score_strat)
		{
		case GeoScoreStrategy::SURFACE_SAMPLES:
			return compute_geo_surface_score(node);
		case GeoScoreStrategy::IN_OUT_SAMPLES:
			return compute_geo_in_out_score(node);
		}
	}

	double compute_geo_in_out_score(const CSGNode& node) const
	{
		double numCorrectSamples = 0.0;
		double numConsideredSamples = (double)(in_pc.rows() + out_pc.rows());

		for (int i = 0; i < in_pc.rows(); ++i)
		{
			Eigen::Matrix<double, 1, 6> pn = in_pc.row(i);
			Eigen::Vector3d p = pn.leftCols(3);
			
			//std::cout << "IN: " << node.signedDistance(p) << std::endl;

			if (node.signedDistance(p) < 0.0)
			{
				numCorrectSamples++;
			}
		}

		for (int i = 0; i < out_pc.rows(); ++i)
		{
			Eigen::Matrix<double, 1, 6> pn = out_pc.row(i);
			Eigen::Vector3d p = pn.leftCols(3);

			//std::cout << "OUT: " << node.signedDistance(p) << std::endl;
			
			if (node.signedDistance(p) > 0.0)
			{
				numCorrectSamples++;
			}
		}

		return (numCorrectSamples / numConsideredSamples);
	}

	double compute_geo_surface_score(const CSGNode& node) const
	{
		double numCorrectSamples = 0.0;
		double numConsideredSamples = (double)surface_pc.rows();

		for (int i = 0; i < surface_pc.rows(); ++i)
		{

			Eigen::Matrix<double, 1, 6> pn = surface_pc.row(i);
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
	lmu::PointCloud surface_pc;
	lmu::PointCloud in_pc;
	lmu::PointCloud out_pc;
	lmu::PointCloud in_out_pc;

};

//////////////////////////// MANIPULATOR ////////////////////////////

struct CSGNodePopulationManipulator
{
	CSGNodePopulationManipulator(CSGNodeCreator* creator, CSGNodeRanker* ranker, double max_delta) :
		max_delta(max_delta),
		ranker(ranker),
		creator(creator),
		iteration(0)
	{
	}

	void manipulateBeforeRanking(std::vector<RankedCreature<CSGNode, Rank>>& population) const
	{
	}

	void manipulateAfterRanking(std::vector<RankedCreature<CSGNode, Rank>>& population) const
	{
		/*auto filtered_pop = population;

		for (auto& node : population)
		{
			if (1.0 - node.rank.geo_score > max_delta)
			{
				node.creature = creator->create();
				node.rank = ranker->rank(node.creature);
			}
		}*/

		// Normalize size. 
		double largest_size_diff = -std::numeric_limits<double>::max();
		double smallest_size_diff = std::numeric_limits<double>::max();
		for (const auto& node : population)
		{
			largest_size_diff = node.rank.size_diff_score > largest_size_diff ? node.rank.size_diff_score : largest_size_diff;
			smallest_size_diff = node.rank.size_diff_score < smallest_size_diff ? node.rank.size_diff_score : smallest_size_diff;
		}
		for (auto& node : population)
			node.rank.size_diff_score = (node.rank.size_diff_score - smallest_size_diff) / (largest_size_diff - smallest_size_diff);

		// Compute score.
		for (auto& node : population)
			node.rank.score =
			ranker->params.geo_score_weight * node.rank.geo_score +
			ranker->params.prox_score_weight * node.rank.prox_score +
			ranker->params.size_score_weight * node.rank.size_diff_score;

		// Add best nodes to pareto set.
		for (auto& node : population)
		{
			if (node.rank.geo_score == 1.0)//&& node.rank.size_score >= 0.0)
			{
				pareto_nodes.push_back(std::make_pair(iteration, node));
			}
		}
		
		iteration++;
	}

	CSGNode get_best()
	{
		CSGNode best = opNo(); 
		double best_score = 0.0;

		double largest_size = -std::numeric_limits<double>::max();
		double smallest_size = std::numeric_limits<double>::max();
		double largest_prox = -std::numeric_limits<double>::max();
		double smallest_prox = std::numeric_limits<double>::max();

		for (const auto& node : pareto_nodes)
		{
			largest_size = node.second.rank.size_score > largest_size ? node.second.rank.size_score : largest_size;
			smallest_size = node.second.rank.size_score < smallest_size ? node.second.rank.size_score : smallest_size;
	
			largest_prox = node.second.rank.prox_score > largest_prox ? node.second.rank.prox_score : largest_prox;
			smallest_prox = node.second.rank.prox_score < smallest_prox ? node.second.rank.prox_score : smallest_prox;
		}

		for (auto& node : pareto_nodes)
		{
			double size_score = (node.second.rank.size_score - smallest_size) / (largest_size - smallest_size);
			double prox_score = (node.second.rank.prox_score - smallest_prox) / (largest_prox - smallest_prox);

			double score = 
				ranker->params.geo_score_weight * node.second.rank.geo_score +
				ranker->params.prox_score_weight * prox_score -
				ranker->params.size_score_weight * size_score;

			if (score > best_score)
			{
				best = node.second.creature;
				best_score = score;
			}
		}

		return best;
	}

	void save_pareto(std::ostream& s)
	{
		s << "# Pareto" << std::endl;

		for (auto& node : pareto_nodes)
		{			
			s << node.first << " " << node.second.rank << std::endl;
		}
	}

	std::string info() const
	{
		return "CSGNode Population Manipulator";
	}

	mutable std::vector<std::pair<int, RankedCreature<CSGNode, Rank>>> pareto_nodes;

	double max_delta;
	CSGNodeCreator* creator;
	CSGNodeRanker* ranker;
	mutable int iteration;
};


struct CSGNodeIterationStopCriterion
{
	CSGNodeIterationStopCriterion(int maxCount, double delta, int maxIterations) :
		_maxCount(maxCount),
		_delta(delta),
		_maxIterations(maxIterations),
		_currentCount(0),
		_lastBestRank(0.0)
	{
	}

	bool shouldStop(std::vector<RankedCreature<CSGNode, Rank>>& population, int iterationCount)
	{
		std::cout << "Iteration " << iterationCount << std::endl;

		if (iterationCount >= _maxIterations)
			return true;

		if (population.empty())
			return true;

		Rank currentBestRank = population[0].rank;

		if( std::abs(currentBestRank.geo_score - _lastBestRank.geo_score) <= _delta &&
			std::abs(currentBestRank.size_score - _lastBestRank.size_score) <= _delta &&
			std::abs(currentBestRank.prox_score - _lastBestRank.prox_score) <= _delta)
		{
			//No change
			_currentCount++;
		}
		else
		{
			_currentCount = 0;
		}

		_lastBestRank = currentBestRank;

		return _currentCount >= _maxCount;
	}

	std::string info() const
	{
		std::stringstream ss;
		ss << "No Change Stop Criterion Selector (maxCount=" << _maxCount << ", delta=" << _delta << ", " << _maxIterations << ")";
		return ss.str();
	}

private:
	int _maxCount;
	int _currentCount;
	int _maxIterations;
	double _delta;
	Rank _lastBestRank;
};

using CSGNodeTournamentSelector = TournamentSelector<RankedCreature<CSGNode, Rank>>;
//using CSGNodeIterationStopCriterion = IterationStopCriterion<RankedCreature<CSGNode, Rank>>;
using CSGNodeNoFitnessIncreaseStopCriterion = NoFitnessIncreaseStopCriterion<RankedCreature<CSGNode, Rank>, Rank>;
using CSGNodeGA = GeneticAlgorithm<CSGNode, CSGNodeCreator, CSGNodeRanker, Rank, CSGNodeTournamentSelector, CSGNodeIterationStopCriterion, CSGNodePopulationManipulator>;

OptimizerGAResult lmu::optimize_with_ga(const CSGNode& node, const OptimizerGAParams& params, std::ostream& report_stream, const std::vector<ImplicitFunctionPtr>& primitives)
{	
	if (node.childsCRef().empty())
		return OptimizerGAResult(node, {});

	if (primitives.size() == 1)
		return OptimizerGAResult(geometry(primitives[0]), {});

	CSGNodeRanker ranker(node, primitives, params.ranker_params);
	CSGNodeCreator creator(node, primitives, params.creator_params);
	CSGNodeIterationStopCriterion stop_criterion(params.ga_params.max_count, params.ga_params.delta, params.ga_params.max_iterations);
	CSGNodeTournamentSelector t_selector(params.ga_params.tournament_k);
	CSGNodePopulationManipulator manipulator(&creator, &ranker, params.man_params.max_delta);


	//auto test_node = fromJSONFile("wrong_node.json");
	//std::cout << "RANK WRONG: " << ranker.rank(test_node) << std::endl;

	//int f;
	//std::cin >> f;

	CSGNodeGA::Parameters ga_params(params.ga_params.population_size, params.ga_params.num_best_parents,
		params.ga_params.mutation_rate, params.ga_params.crossover_rate, params.ga_params.in_parallel,
		Schedule(), Schedule(), params.ga_params.use_caching);

	CSGNodeGA ga;
	auto res = ga.run(ga_params, t_selector, creator, ranker, stop_criterion, manipulator);

	std::vector<CSGNode> pareto_nodes;
	std::transform(manipulator.pareto_nodes.begin(), manipulator.pareto_nodes.end(), std::back_inserter(pareto_nodes), 
		[](const auto& rpn) { return rpn.second.creature; });

	auto opt_res = OptimizerGAResult(manipulator.get_best(), pareto_nodes);
	
	res.statistics.save(report_stream, &opt_res.node);

	manipulator.save_pareto(report_stream);
	
	//toJSONFile(opt_res.node, "wrong_node.json");
	
	return opt_res;
}

void compute_local_proximity_score_rec(const CSGNode& node, double sampling_grid_size, double& score, 
	const lmu::PointCloud& sampling_points, EmptySetLookup& esLookup)
{
	auto n = node.childsCRef().size();

	if (score == invalid_proximity_score || n == 0)
	{
		return;
	}	
	else if (n > 2)
	{
		score = invalid_proximity_score;
		std::cout << "Proximity score computation: No more than 2 children allowed." << std::endl;
		return;
	}
	else if (n == 1)
	{
		score += 1.0;
		compute_local_proximity_score_rec(node.childsCRef()[0], sampling_grid_size, score, 
			sampling_points, esLookup);
	}
	else if (n == 2)
	{
		const auto& left = node.childsCRef()[0];
		const auto& right = node.childsCRef()[1];

		score += is_empty_set(opInter({ left, right }), sampling_grid_size, 
			sampling_points, esLookup) ? 0.0 : 1.0;
		
		compute_local_proximity_score_rec(left, sampling_grid_size, score, sampling_points, esLookup);
		compute_local_proximity_score_rec(right, sampling_grid_size, score, sampling_points, esLookup);
	}
}

double lmu::compute_local_proximity_score(const CSGNode& node, double sampling_grid_size, 
	const lmu::PointCloud& sampling_points)
{
	EmptySetLookup esLookup;
	double score = 0.0;

	compute_local_proximity_score_rec(node, sampling_grid_size, score, sampling_points, esLookup);

	//std::cout << "NUM: " << (double)numNodes(node, true) << std::endl;
	//std::cout << "SCORE: " << score;

	auto num_nodes = numNodes(node, true);
	if (num_nodes == 0)
		return 0.0;

	return score == invalid_proximity_score ? score : score / (double)num_nodes;
}

