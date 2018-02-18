#ifndef CSGNODE_EVO_H
#define CSGNODE_EVO_H

#include <vector>
#include <memory>

#include "csgnode.h"
#include "evolution.h"

#include <Eigen/Core>

namespace lmu
{
	struct ImplicitFunction;

	struct CSGNodeCreator
	{
		CSGNodeCreator(const std::vector<std::shared_ptr<ImplicitFunction>>& functions, double createNewRandomProb = 0.5, double subtreeProb = 0.7, int maxTreeDepth = 10, const lmu::Graph& connectionGraph = lmu::Graph());

		CSGNode mutate(const CSGNode& tree) const;
		std::vector<CSGNode> crossover(const CSGNode& tree1, const CSGNode& tree2) const;
		CSGNode create() const;
		CSGNode create(int maxDepth) const;

		std::string info() const;

	private:

		void create(CSGNode& node, int maxDepth, int curDepth) const;

		int getRndFuncIndex(const std::vector<int>& usedFuncIndices) const;

		double _createNewRandomProb;
		double _subtreeProb;
		int _maxTreeDepth;
		std::vector<std::shared_ptr<ImplicitFunction>> _functions;
		mutable std::default_random_engine _rndEngine;
		mutable std::random_device _rndDevice;

		lmu::Graph _connectionGraph;
	};

	struct CSGNodeRanker
	{
		CSGNodeRanker(double lambda, const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& functions, const lmu::Graph& connectionGraph = lmu::Graph());

		double rank(const CSGNode& node) const;
		std::string info() const;

		bool treeIsInvalid(const lmu::CSGNode& node) const;

	private:
		double _lambda;
		std::vector<std::shared_ptr<lmu::ImplicitFunction>> _functions;
		bool _earlyOutTest;
		lmu::Graph _connectionGraph;
	};

	using CSGNodeTournamentSelector = TournamentSelector<RankedCreature<CSGNode>>;

	using CSGNodeIterationStopCriterion = IterationStopCriterion<RankedCreature<CSGNode>>;
	using CSGNodeNoFitnessIncreaseStopCriterion = NoFitnessIncreaseStopCriterion<RankedCreature<CSGNode>>;

	using CSGNodeGA = GeneticAlgorithm<CSGNode, CSGNodeCreator, CSGNodeRanker, CSGNodeTournamentSelector, CSGNodeNoFitnessIncreaseStopCriterion>;

	CSGNode createCSGNodeWithGA(const std::vector<std::shared_ptr<ImplicitFunction>>& shapes, bool inParallel = false, const lmu::Graph& connectionGraph = Graph());

	using GeometryCliqueWithCSGNode = std::tuple<Clique, CSGNode>;

	enum class ParallelismOptions
	{
		NoParallelism = 0,
		PerCliqueParallelism = 1, 
		GAParallelism = 2
	};
	ParallelismOptions operator|(ParallelismOptions lhs, ParallelismOptions rhs);
	ParallelismOptions operator&(ParallelismOptions lhs, ParallelismOptions rhs);

	std::vector<GeometryCliqueWithCSGNode> computeNodesForCliques(std::vector<Clique> geometryCliques, const Graph& connectionGraph, ParallelismOptions po);

	using CSGNodeClique = std::vector<GeometryCliqueWithCSGNode>;

	CSGNode mergeCSGNodeCliqueSimple(CSGNodeClique& clique);
	
}

#endif