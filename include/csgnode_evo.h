#ifndef CSGNODE_EVO_H
#define CSGNODE_EVO_H

#include <vector>
#include <memory>

#include "csgnode.h"
#include "evolution.h"
#include "congraph.h"
#include "params.h"

#include <Eigen/Core>

namespace lmu
{
	struct ImplicitFunction;

	struct ParetoState
	{
		ParetoState();

		void update(const CSGNode& node, double geoScore); 
		CSGNode getBest() const;

	private: 
		std::mutex _mutex;
		std::vector<std::tuple<CSGNode, double>> _bestGeoScoreSizeScoreNodes;
		double _currentBestGeoScore;
	};

	struct CSGNodeRanker
	{
		CSGNodeRanker(double lambda, double epsilon, double alpha, double h, const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& functions, const lmu::Graph& connectionGraph = lmu::Graph(), std::shared_ptr<ParetoState> ps = nullptr);

		double rank(const CSGNode& node) const;
		double rank(const CSGNode& node, const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& functions, bool isCompleteModel = false) const;

		std::string info() const;

		bool treeIsInvalid(const lmu::CSGNode& node) const;

		int getNumSamplePoints() const;

	private:

		int getNumSamplePoints(const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& functions) const;
		double computeEpsilonScale();
		double _h;
		double _lambda;
		std::vector<std::shared_ptr<lmu::ImplicitFunction>> _functions;
		bool _earlyOutTest;
		lmu::Graph _connectionGraph;
		double _epsilonScale;
		double _epsilon;
		double _alpha;

		int _numSamplePoints; 

		std::shared_ptr<ParetoState> _paretoState;
	};

	using MappingFunction = std::function<double(double)>;

	struct CSGNodeCreator
	{
		CSGNodeCreator(const std::vector<std::shared_ptr<ImplicitFunction>>& functions, double createNewRandomProb, double subtreeProb, double simpleCrossoverProb, int maxTreeDepth, double initializeWithUnionOfAllFunctions, const lmu::CSGNodeRanker& ranker, const lmu::Graph& connectionGraph = lmu::Graph());

		CSGNode mutate(const CSGNode& tree) const;
		std::vector<CSGNode> crossover(const CSGNode& tree1, const CSGNode& tree2) const;
		CSGNode create(bool unions = true) const;
		CSGNode create(int maxDepth) const;
		std::string info() const;

	private:

		std::vector<CSGNode> simpleCrossover(const CSGNode& tree1, const CSGNode& tree2) const;
		std::vector<CSGNode> sharedPrimitiveCrossover(const CSGNode& tree1, const CSGNode& tree2) const;

		void create(CSGNode& node, int maxDepth, int curDepth) const;
		void createUnionTree(CSGNode& node, std::vector<ImplicitFunctionPtr>& funcs) const;

		int getRndFuncIndex(const std::vector<int>& usedFuncIndices) const;

		double _createNewRandomProb;
		double _subtreeProb;
		double _simpleCrossoverProb;
		double _initializeWithUnionOfAllFunctions;

		int _maxTreeDepth;
		std::vector<std::shared_ptr<ImplicitFunction>> _functions;
		mutable std::default_random_engine _rndEngine;
		mutable std::random_device _rndDevice;

		lmu::Graph _connectionGraph;
		lmu::CSGNodeRanker _ranker;
	};
	
	enum class CSGNodeOptimization
	{
		RANDOM, 
		TRAVERSE
	};

	CSGNodeOptimization optimizationTypeFromString(std::string type);

	struct CSGNodePopMan
	{	
		CSGNodePopMan(double optimizationProb, double preOptimizationProb, int maxFunctions, int nodeSelectionTries, int randomIterations, CSGNodeOptimization type, const lmu::CSGNodeRanker& ranker, const lmu::Graph& connectionGraph);

		void manipulateBeforeRanking(std::vector<RankedCreature<CSGNode>>& population) const;
		void manipulateAfterRanking(std::vector<RankedCreature<CSGNode>>& population) const;
		std::string info() const;

	private: 

		CSGNode getOptimizedTree(std::vector<ImplicitFunctionPtr> funcs) const;
		std::vector<ImplicitFunctionPtr> getSuitableFunctions(const std::vector<ImplicitFunctionPtr>& funcs) const;
		double _optimizationProb;
		double _preOptimizationProb;
		int _maxFunctions;
		int _nodeSelectionTries;
		lmu::CSGNodeRanker _ranker;
		lmu::Graph _connectionGraph;
		CSGNodeOptimization _type;
		int _randomIterations;
		mutable std::unordered_map<size_t, CSGNode> _nodeLookup;

		mutable std::default_random_engine _rndEngine;
		mutable std::random_device _rndDevice;
	};
	
	using CSGNodeTournamentSelector = TournamentSelector<RankedCreature<CSGNode>>;

	using CSGNodeIterationStopCriterion = IterationStopCriterion<RankedCreature<CSGNode>>;
	using CSGNodeNoFitnessIncreaseStopCriterion = NoFitnessIncreaseStopCriterion<RankedCreature<CSGNode>>;

	using CSGNodeGA = GeneticAlgorithm<CSGNode, CSGNodeCreator, CSGNodeRanker, CSGNodeTournamentSelector, CSGNodeNoFitnessIncreaseStopCriterion, CSGNodePopMan>;

	CSGNode createCSGNodeWithGA(const std::vector<std::shared_ptr<ImplicitFunction>>& shapes, const lmu::ParameterSet& p, const lmu::Graph& connectionGraph = Graph());

	using GeometryCliqueWithCSGNode = std::tuple<Clique, CSGNode>;

	enum class ParallelismOptions
	{
		NoParallelism = 0,
		PerCliqueParallelism = 1, 
		GAParallelism = 2
	};
	ParallelismOptions operator|(ParallelismOptions lhs, ParallelismOptions rhs);
	ParallelismOptions operator&(ParallelismOptions lhs, ParallelismOptions rhs);

	std::vector<GeometryCliqueWithCSGNode> computeNodesForCliques(const std::vector<Clique>& geometryCliques, const ParameterSet& params, ParallelismOptions po);

	using CSGNodeClique = std::vector<GeometryCliqueWithCSGNode>;

	CSGNode mergeCSGNodeCliqueSimple(CSGNodeClique& clique);
	void optimizeCSGNodeClique(CSGNodeClique& clique, float tolerance);
  
	double lambdaBasedOnPoints(const std::vector<lmu::ImplicitFunctionPtr>& shapes);
	
    CSGNode computeGAWithPartitions(const std::vector<Graph>& partitions, const lmu::ParameterSet& p);
}

#endif