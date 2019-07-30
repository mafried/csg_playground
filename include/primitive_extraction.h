#ifndef PRIMITIVE_EXTRACTION_H
#define PRIMITIVE_EXTRACTION_H

#include "primitives.h"
#include "evolution.h"

#include <vector>

namespace lmu
{
	using PrimitiveSetRank = double;

	struct PrimitiveSetCreator
	{
		PrimitiveSetCreator(const ManifoldSet& ms, double intraCrossProb, const std::vector<double>& mutationDistribution,
			int maxMutationIterations, int maxCrossoverIterations, int maxPrimitiveSetSize, double angleEpsilon);

		int getRandomPrimitiveIdx(const PrimitiveSet & ps) const;

		PrimitiveSet mutate(const PrimitiveSet& ps) const;
		std::vector<PrimitiveSet> crossover(const PrimitiveSet& ps1, const PrimitiveSet& ps2) const;
		PrimitiveSet create() const;
		std::string info() const;

	private:

		enum class MutationType
		{
			NEW,
			REPLACE,
			MODIFY,
			REMOVE,
			ADD
		};

		ManifoldPtr getManifold(ManifoldType type, const Eigen::Vector3d& direction, const ManifoldSet& alreadyUsed, double angleEpsilon, bool ignoreDirection = false) const;
		ManifoldPtr getPerpendicularPlane(const std::vector<ManifoldPtr>& planes, const ManifoldSet& alreadyUsed, double angleEpsilon) const;
		ManifoldPtr getParallelPlane(const ManifoldPtr& plane, const ManifoldSet& alreadyUsed, double angleEpsilon) const;

		std::unordered_set<ManifoldType> getAvailableManifoldTypes(const ManifoldSet& ms) const;
		PrimitiveType getRandomPrimitiveType() const;

		Primitive createPrimitive() const;
		Primitive mutatePrimitive(const Primitive& p, double angleEpsilon) const;

		ManifoldSet ms;
		std::unordered_set<ManifoldType> availableManifoldTypes;
		double intraCrossProb;

		std::vector<double> mutationDistribution;

		int maxMutationIterations;
		int maxCrossoverIterations;
		int maxPrimitiveSetSize;
		double angleEpsilon;

		mutable std::default_random_engine rndEngine;
		mutable std::random_device rndDevice;
	};

	struct PrimitiveSetRanker
	{
		PrimitiveSetRanker(const PointCloud& pc, const ManifoldSet& ms, const PrimitiveSet& staticPrims, double distanceEpsilon, int maxPrimitiveSetSize);

		PrimitiveSetRank rank(const PrimitiveSet& ps) const;
		std::string info() const;

		PrimitiveSet bestPrimitiveSet() const;

	private: 
		mutable PrimitiveSetRank bestRank;
		mutable PrimitiveSet bestPrimitives;

		PrimitiveSet staticPrimitives;

		PointCloud pc;
		ManifoldSet ms;
		double distanceEpsilon;
		int maxPrimitiveSetSize;

		double getCompleteUseScore(const ManifoldSet& ms, const PrimitiveSet& ps) const;
	};

	struct GAResult
	{
		PrimitiveSet primitives; 
		ManifoldSet manifolds; 
	};

	lmu::ManifoldSet generateGhostPlanes(const PointCloud& pc, const lmu::ManifoldSet& ms, double distanceThreshold, double angleThreshold);

	GAResult extractPrimitivesWithGA(const RansacResult& ransacResult);

	Primitive createBoxPrimitive(const ManifoldSet& planes);
	lmu::Primitive createSpherePrimitive(const ManifoldPtr& m);
	Primitive createCylinderPrimitive(const ManifoldPtr& m, ManifoldSet& planes);

	using PrimitiveSetTournamentSelector = TournamentSelector<RankedCreature<PrimitiveSet, PrimitiveSetRank>>;

	using PrimitiveSetIterationStopCriterion = NoFitnessIncreaseStopCriterion<RankedCreature<PrimitiveSet, PrimitiveSetRank>>;
		//IterationStopCriterion<RankedCreature<PrimitiveSet, PrimitiveSetRank>>;
	
	using PrimitiveSetGA = GeneticAlgorithm<PrimitiveSet, PrimitiveSetCreator, PrimitiveSetRanker, PrimitiveSetRank, 
		PrimitiveSetTournamentSelector, PrimitiveSetIterationStopCriterion>;
	
	PrimitiveSet extractCylindersFromCurvedManifolds(const ManifoldSet& manifolds, bool estimateHeight);

	std::tuple<double, Eigen::Vector3d> estimateCylinderHeightAndPosFromPointCloud(const Manifold& m);
	ManifoldPtr estimateSecondCylinderPlaneFromPointCloud(const Manifold& m, const Manifold& firstPlane);

}

#endif 