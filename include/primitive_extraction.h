#ifndef PRIMITIVE_EXTRACTION_H
#define PRIMITIVE_EXTRACTION_H

#include "primitives.h"
#include "evolution.h"

#include <vector>

namespace lmu
{


	struct AreaScore
	{
		AreaScore() :
			AreaScore(0.0, 0.0)
		{
		}

		AreaScore(double a, double pa) :
			area(a),
			point_area(pa)
		{
		}

		AreaScore operator += (const AreaScore& a)
		{
			area += a.area;
			point_area += a.point_area;
			return *this;
		}

		bool operator == (const AreaScore& a) const
		{
			return area == a.area && point_area == a.point_area;
		}

		bool operator != (const AreaScore& a) const
		{
			return !operator==(a);
		}
		static const AreaScore Invalid;

		double point_area;
		double area;
	};

	struct GeometryScore
	{
		GeometryScore(int cp, int vp) :
			checked_points(cp),
			valid_points(vp)
		{
		}

		int checked_points;
		int valid_points;
	};


	//using PrimitiveSetRank = double;
	struct PrimitiveSetRank
	{
		PrimitiveSetRank(double geo, double total_area, double relative_area, double size, double combined, 
			const std::vector<AreaScore>& per_primitive_area_scores = std::vector<AreaScore>()) :
			geo(geo),
			total_area(total_area),
			relative_area(relative_area),
			size(size),
			combined(combined),
			per_primitive_area_scores(per_primitive_area_scores)
		{
		}

		PrimitiveSetRank() :
			PrimitiveSetRank(0.0)
		{
		}

		explicit PrimitiveSetRank(double v) :
			PrimitiveSetRank(v, v, v, v, v)
		{
		}

		static const PrimitiveSetRank Invalid;

		double geo;
		double size;
		double total_area;
		double relative_area;
		double combined;

		std::vector<AreaScore> per_primitive_area_scores;

		friend inline bool operator< (const PrimitiveSetRank& lhs, const PrimitiveSetRank& rhs) { return lhs.combined < rhs.combined; }
		friend inline bool operator> (const PrimitiveSetRank& lhs, const PrimitiveSetRank& rhs) { return rhs < lhs; }
		friend inline bool operator<=(const PrimitiveSetRank& lhs, const PrimitiveSetRank& rhs) { return !(lhs > rhs); }
		friend inline bool operator>=(const PrimitiveSetRank& lhs, const PrimitiveSetRank& rhs) { return !(lhs < rhs); }
		friend inline bool operator==(const PrimitiveSetRank& lhs, const PrimitiveSetRank& rhs) { return lhs.combined == rhs.combined; }
		friend inline bool operator!=(const PrimitiveSetRank& lhs, const PrimitiveSetRank& rhs) { return !(lhs == rhs); }

		PrimitiveSetRank& operator+=(const PrimitiveSetRank& rhs)
		{
			geo += rhs.geo;
			size += rhs.size;
			total_area += rhs.total_area;
			relative_area += rhs.relative_area;
			combined += rhs.combined;

			return *this;
		}

		PrimitiveSetRank& operator-=(const PrimitiveSetRank& rhs)
		{
			geo -= rhs.geo;
			size -= rhs.size;
			total_area -= rhs.total_area;
			relative_area -= rhs.relative_area;
			combined -= rhs.combined;

			return *this;
		}

		friend PrimitiveSetRank operator+(PrimitiveSetRank lhs, const PrimitiveSetRank& rhs)
		{
			lhs += rhs;
			return lhs;
		}

		friend PrimitiveSetRank operator-(PrimitiveSetRank lhs, const PrimitiveSetRank& rhs)
		{
			lhs -= rhs;
			return lhs;
		}
	};

	std::ostream& operator<<(std::ostream& out, const PrimitiveSetRank& r);

	struct PrimitiveSetCreator
	{
		PrimitiveSetCreator(const ManifoldSet& ms, double intraCrossProb, const std::vector<double>& mutationDistribution,
			int maxMutationIterations, int maxCrossoverIterations, int maxPrimitiveSetSize, double angleEpsilon, 
			double minDistanceBetweenParallelPlanes);

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

		ManifoldPtr getManifold(ManifoldType type, const Eigen::Vector3d& direction, const ManifoldSet& alreadyUsed, 
			double angleEpsilon, bool ignoreDirection = false, 
			const Eigen::Vector3d& point = Eigen::Vector3d(0,0,0), double minimumPointDistance = 0.0) const;

		ManifoldPtr getPerpendicularPlane(const std::vector<ManifoldPtr>& planes, const ManifoldSet& alreadyUsed, double angleEpsilon) const;
		ManifoldPtr getParallelPlane(const ManifoldPtr& plane, const ManifoldSet& alreadyUsed, double angleEpsilon, double minDistanceToPlanePoint) const;

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
		double minDistanceBetweenParallelPlanes;

		mutable std::default_random_engine rndEngine;
		mutable std::random_device rndDevice;
	};

	struct PrimitiveSetCreatorBasedOnPrimitiveSet
	{
		PrimitiveSetCreatorBasedOnPrimitiveSet(const PrimitiveSet& primitives, const std::vector<double>& mutationDistribution,
			int maxMutationIterations, int maxCrossoverIterations);

		PrimitiveSet mutate(const PrimitiveSet& ps) const;
		std::vector<PrimitiveSet> crossover(const PrimitiveSet& ps1, const PrimitiveSet& ps2) const;
		PrimitiveSet create() const;
		std::string info() const;

		int getRandomPrimitiveIdx(const PrimitiveSet & ps) const;

	private:

		enum class MutationType
		{
			NEW,
			REPLACE,
			REMOVE,
			ADD
		};
		std::vector<double> mutationDistribution;
		int maxMutationIterations;
		int maxCrossoverIterations;
		PrimitiveSet primitives;

		mutable std::default_random_engine rndEngine;
		mutable std::random_device rndDevice;
	};

	struct PrimitiveSetRanker
	{
		PrimitiveSetRanker(const PointCloud& pc, const ManifoldSet& ms, const PrimitiveSet& staticPrims, double distanceEpsilon, int maxPrimitiveSetSize, 
			double surface_area);

		PrimitiveSetRank rank(const PrimitiveSet& ps) const;

		std::string info() const;

		AreaScore getAreaScore(const Primitive& p, int& cache_hits) const;
		GeometryScore getGeometryScore(const PrimitiveSet& ps) const;

		double getCompleteUseScore(const ManifoldSet& ms, const PrimitiveSet& ps) const;

	private: 
		double surface_area;

		PrimitiveSet staticPrimitives;

		PointCloud pc;
		ManifoldSet ms;
		double distanceEpsilon;
		int maxPrimitiveSetSize;

		mutable std::unordered_map<size_t, AreaScore> primitiveAreaScoreLookup;
		mutable std::mutex lookupMutex;
	};
	

	struct GAResult
	{
		PrimitiveSet primitives; 
		ManifoldSet manifolds; 
	};

	struct PrimitiveSetPopMan
	{
		PrimitiveSetPopMan(const PrimitiveSetRanker& ranker, double geoWeight, double areaWeight, double sizeWeight);

		void manipulateBeforeRanking(std::vector<RankedCreature<PrimitiveSet, PrimitiveSetRank>>& population) const;
		void manipulateAfterRanking(std::vector<RankedCreature<PrimitiveSet, PrimitiveSetRank>>& population) const;
		std::string info() const;
		
		double geoWeight;
		double areaWeight;
		double sizeWeight;

		const PrimitiveSetRanker* ranker;
	};

	lmu::ManifoldSet generateGhostPlanes(const PointCloud& pc, const lmu::ManifoldSet& ms, double distanceThreshold, double angleThreshold);

	GAResult extractPrimitivesWithGA(const RansacResult& ransacResult);

	Primitive createBoxPrimitive(const ManifoldSet& planes);
	lmu::Primitive createSpherePrimitive(const ManifoldPtr& m);
	Primitive createCylinderPrimitive(const ManifoldPtr& m, ManifoldSet& planes);

	using PrimitiveSetTournamentSelector = TournamentSelector<RankedCreature<PrimitiveSet, PrimitiveSetRank>>;

	using PrimitiveSetIterationStopCriterion = NoFitnessIncreaseStopCriterion<RankedCreature<PrimitiveSet, PrimitiveSetRank>, PrimitiveSetRank>;
		//IterationStopCriterion<RankedCreature<PrimitiveSet, PrimitiveSetRank>>;
	
	using PrimitiveSetGA = GeneticAlgorithm<PrimitiveSet, PrimitiveSetCreator, PrimitiveSetRanker, PrimitiveSetRank, 
		PrimitiveSetTournamentSelector, PrimitiveSetIterationStopCriterion, PrimitiveSetPopMan>;

	using PrimitiveSetGABasedOnPrimitiveSet = GeneticAlgorithm<PrimitiveSet, PrimitiveSetCreatorBasedOnPrimitiveSet, PrimitiveSetRanker, PrimitiveSetRank,
		PrimitiveSetTournamentSelector, PrimitiveSetIterationStopCriterion, PrimitiveSetPopMan>;
	
	PrimitiveSet extractCylindersFromCurvedManifolds(const ManifoldSet& manifolds, bool estimateHeight);

	double estimateCylinderHeightFromPointCloud(const Manifold& m);
	ManifoldPtr estimateSecondCylinderPlaneFromPointCloud(const Manifold& m, const Manifold& firstPlane);

}

#endif 