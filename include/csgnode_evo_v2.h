#ifndef CSGNODE_EVO_V2_H
#define CSGNODE_EVO_V2_H

#include <vector>
#include <memory>
#include <unordered_map>

#include "csgnode.h"
#include "evolution.h"
#include "csgnode_evo.h"

#include <Eigen/Core>

namespace lmu
{
	using IFBudget = std::unordered_map<lmu::ImplicitFunctionPtr, int>;

	std::ostream& operator<<(std::ostream& os, const IFBudget& b);
		
	struct CSGNodeCreatorV2
	{
		CSGNodeCreatorV2(double createNewRandomProb, double subtreeProb, const lmu::Graph& connectionGraph);

		CSGNode mutate(const CSGNode& tree) const;
		std::vector<CSGNode> crossover(const CSGNode& tree1, const CSGNode& tree2) const;
		CSGNode create() const;
		
		std::string info() const;

	private:
		CSGNode create(IFBudget& budget) const;
		
		double _createNewRandomProb;
		double _subtreeProb;
		lmu::Graph _connectionGraph;
		IFBudget _ifBudget;

		mutable std::default_random_engine _rndEngine;
	};

	struct CSGNodeRankerV2
	{
		CSGNodeRankerV2(const lmu::Graph& g, double sizeWeight, double h);

		double rank(const CSGNode& node) const;
		std::string info() const;

	private: 
		lmu::Graph _connectionGraph;
		IFBudget _ifBudget;
		double _sizeWeight;
		double _h;
	};

	using CSGNodeGAV2 = GeneticAlgorithm<CSGNode, CSGNodeCreatorV2, CSGNodeRankerV2, CSGNodeTournamentSelector, CSGNodeNoFitnessIncreaseStopCriterion>;

	CSGNode createCSGNodeWithGAV2(const lmu::Graph& connectionGraph, bool inParallel = false, const std::string& statsFile = std::string("stats.dat"));

}

#endif