#ifndef PRIMITIVE_EXTRACTION_H
#define PRIMITIVE_EXTRACTION_H

#include "primitives.h"
#include "evolution.h"
#include "pc_structure.h"

#include <vector>

typedef struct _object PyObject;

namespace lmu
{
	struct PrimitiveSetRank
	{
		PrimitiveSetRank(double geo, double per_prim_geo_sum, double size, double combined,
			const std::vector<double>& per_primitive_geo_scores = std::vector<double>()) :
			geo(geo),
			per_prim_geo_sum(per_prim_geo_sum),
			size(size),
			combined(combined),
			per_primitive_geo_scores(per_primitive_geo_scores)
		{
		}

		PrimitiveSetRank() :
			PrimitiveSetRank(0.0)
		{
		}

		explicit PrimitiveSetRank(double v) :
			PrimitiveSetRank(v, v, v, v)
		{
		}

		static const PrimitiveSetRank Invalid;

		double geo;
		double per_prim_geo_sum;
		double size;
		double combined;
		std::vector<double> per_primitive_geo_scores;

		double per_primitive_mean_score;
		double size_unnormalized;
		double geo_unnormalized;

		void capture_score_stats();

		operator double() const { return combined; }
		
		friend inline bool operator< (const PrimitiveSetRank& lhs, const PrimitiveSetRank& rhs) { return lhs.combined < rhs.combined || lhs.combined == rhs.combined && lhs.size < rhs.size; }
		friend inline bool operator> (const PrimitiveSetRank& lhs, const PrimitiveSetRank& rhs) { return lhs.combined > rhs.combined || lhs.combined == rhs.combined && lhs.size > rhs.size; }
		//friend inline bool operator<=(const PrimitiveSetRank& lhs, const PrimitiveSetRank& rhs) { return !(lhs > rhs); }
		//friend inline bool operator>=(const PrimitiveSetRank& lhs, const PrimitiveSetRank& rhs) { return !(lhs < rhs); }
		//friend inline bool operator==(const PrimitiveSetRank& lhs, const PrimitiveSetRank& rhs) { return lhs.combined == rhs.combined; }
		//friend inline bool operator!=(const PrimitiveSetRank& lhs, const PrimitiveSetRank& rhs) { return !(lhs == rhs); }

		PrimitiveSetRank& operator+=(const PrimitiveSetRank& rhs)
		{
			geo += rhs.geo;
			per_prim_geo_sum += rhs.per_prim_geo_sum;
			size += rhs.size;
			combined += rhs.combined;

			return *this;
		}

		PrimitiveSetRank& operator-=(const PrimitiveSetRank& rhs)
		{
			geo -= rhs.geo;
			per_prim_geo_sum -= rhs.per_prim_geo_sum;
			size -= rhs.size;
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
		PrimitiveSetCreator(const PlaneGraph& plane_graph, double intraCrossProb, const std::vector<double>& mutationDistribution,
			int maxMutationIterations, int maxCrossoverIterations, int maxPrimitiveSetSize, double angleEpsilon, 
			double minDistanceBetweenParallelPlanes, double polytope_prob, int min_polytope_planes, int max_polytope_planes,
			const Eigen::Vector3d& polytope_center, const ManifoldSet& fixed_planes);

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
			const Eigen::Vector3d& point = Eigen::Vector3d(0,0,0), double minimumPointDistance = 0.0, bool ignorePointDistance = false) const;

		ManifoldPtr getNeighborPlane(const ManifoldPtr& plane, const ManifoldSet& already_used) const;

		ManifoldPtr getPerpendicularPlane(const std::vector<ManifoldPtr>& planes, const ManifoldSet& alreadyUsed, double angleEpsilon) const;
		ManifoldPtr getParallelPlane(const ManifoldPtr& plane, const ManifoldSet& alreadyUsed, double angleEpsilon, double minDistanceToPlanePoint) const;

		std::unordered_set<ManifoldType> getAvailableManifoldTypes(const ManifoldSet& ms) const;
		PrimitiveType getRandomPrimitiveType() const;

		Primitive createPrimitive() const;
		Primitive mutatePrimitive(const Primitive& p, double angleEpsilon) const;

		PlaneGraph plane_graph;
		double intraCrossProb;

		int min_polytope_planes;
		int max_polytope_planes;
		double polytope_prob;

		std::vector<double> mutationDistribution;

		int maxMutationIterations;
		int maxCrossoverIterations;
		int maxPrimitiveSetSize;
		double angleEpsilon;
		double minDistanceBetweenParallelPlanes;

		mutable std::default_random_engine rndEngine;
		mutable std::random_device rndDevice;

		Eigen::Vector3d polytope_center;
		ManifoldSet fixed_planes;
	};
	
	struct PrimitiveSetRanker;
	struct GAResult
	{
		PrimitiveSet polytopes; 
		PlaneGraph plane_graph; 
		std::shared_ptr<PrimitiveSetRanker> ranker;
	};

	struct SDFValue
	{
		static const float max_distance;

		SDFValue();
		SDFValue(float d, Eigen::Vector3f& n);

		float d;
		Eigen::Vector3f n;
	};

	enum class DHType
	{
		NONE,
		INSIDE,
		OUTSIDE
	};
	std::ostream& operator<<(std::ostream& out, const DHType& t);


	struct ModelSDF
	{
		ModelSDF(const PointCloud& pc, double voxel_size, std::ofstream& s);

		~ModelSDF();

		void recreate_from_mesh(const Mesh& m);

		double distance(const Eigen::Vector3d& p) const;
		SDFValue sdf_value(const Eigen::Vector3d& p) const;

		Mesh to_mesh() const;
		PointCloud to_pc() const;

		DHType get_dh_type(const Primitive &p, double t_inside, double t_outside, double voxel_size, std::vector<Eigen::Matrix<double, 1, 6>>& points, bool debug = false) const;

		Eigen::Vector3i grid_size;
		Eigen::Vector3d origin;
		double voxel_size;
		Mesh surface_mesh;
		SDFValue* data;

	private:
		
		Eigen::Vector3d size;

		int n;

		igl::AABB<Eigen::MatrixXd, 3> tree;
		Eigen::MatrixXd fn, vn, en; //note that _vn is the same as mesh's _normals. TODO
		Eigen::MatrixXi e;
		Eigen::VectorXi emap;
	};

	struct PrimitiveSetRanker
	{
		PrimitiveSetRanker(const PointCloud& pc,
			double distanceEpsilon, int maxPrimitiveSetSize, double cell_size, bool allow_cube_cutout, const std::shared_ptr<ModelSDF>& model_sdf,
			double geo_weight, double per_prim_geo_weight, double size_weight);

		PrimitiveSetRank rank(const PrimitiveSet& ps, bool debug = false) const;

		std::vector<double> get_per_prim_geo_score(const PrimitiveSet& ps, std::vector<Eigen::Matrix<double, 1, 6>>& points, bool debug = false) const;

		std::string info() const;

		std::shared_ptr<ModelSDF> model_sdf;
		
	private:

		double get_geo_score(const PrimitiveSet& ps) const;
	
		PointCloud pc;
		double distanceEpsilon;
		double cell_size;
		int maxPrimitiveSetSize;
		bool allow_cube_cutout;
		double geo_weight;	
		double per_prim_geo_weight;
		double size_weight;
	};

	struct PrimitiveSetPopMan
	{
		PrimitiveSetPopMan(const PrimitiveSetRanker& ranker, int maxPrimitiveSetSize,
			double geoWeight, double perPrimGeoWeight, double sizeWeight,
			int num_elite_injections);

		void manipulateBeforeRanking(std::vector<RankedCreature<PrimitiveSet, PrimitiveSetRank>>& population) const;
		void manipulateAfterRanking(std::vector<RankedCreature<PrimitiveSet, PrimitiveSetRank>>& population) const;
		std::string info() const;

		double geoWeight;
		double perPrimGeoWeight;
		double sizeWeight;
		int num_elite_injections;
		int maxPrimitiveSetSize;

		const PrimitiveSetRanker* ranker;
	};

	using PrimitiveSetTournamentSelector = TournamentSelector<RankedCreature<PrimitiveSet, PrimitiveSetRank>>;
	using PrimitiveSetIterationStopCriterion = NoFitnessIncreaseStopCriterion<RankedCreature<PrimitiveSet, PrimitiveSetRank>, PrimitiveSetRank>;
	//IterationStopCriterion<RankedCreature<PrimitiveSet, PrimitiveSetRank>>;
	using PrimitiveSetGA = GeneticAlgorithm<PrimitiveSet, PrimitiveSetCreator, PrimitiveSetRanker, PrimitiveSetRank,
		PrimitiveSetTournamentSelector, PrimitiveSetIterationStopCriterion, PrimitiveSetPopMan>;

	struct PrimitiveGaParams
	{
		double size_weight;// = 0.1;
		double geo_weight;// = 0.0;
		double per_prim_geo_weight;// = 1.0;//0.1;

		int maxPrimitiveSetSize;// = 75;
		double polytope_prob; // = 0.0;
		int min_polytope_planes;
		int max_polytope_planes;

		double ranker_voxel_size;// = 0.05;
		double sdf_voxel_size;
		double max_dist;// = 0.05;
		double allow_cube_cutout;// = true;

		int max_iterations; //30
		int max_count; //30

		double similarity_filter_epsilon; //0.0
		double filter_threshold; //0.01

		bool similarity_filter_similarity_only;
		double similarity_filter_perfectness_t;
		double similarity_filter_voxel_size;

		int num_geo_score_samples;
		int num_elite_injections;
	};

	PrimitiveSet extractNonPlanarPrimitives(const ManifoldSet& manifolds);

	GAResult extractPolytopePrimitivesWithGA(const PlaneGraph& plane_graph, const std::shared_ptr<ModelSDF>& model_sdf, const PrimitiveGaParams& params, std::ostream& stream);

	Primitive createBoxPrimitive(const ManifoldSet& planes);
	Primitive createPolytopePrimitive(const ManifoldSet& planes, const Eigen::Vector3d& polytope_center);

	lmu::Primitive createSpherePrimitive(const ManifoldPtr& m);
	Primitive createCylinderPrimitive(const ManifoldPtr& m, ManifoldSet& planes);

	double estimateCylinderHeightFromPointCloud(const Manifold& m);
	ManifoldPtr estimateSecondCylinderPlaneFromPointCloud(const Manifold& m, const Manifold& firstPlane);

	struct ThresholdOutlierDetector
	{
		ThresholdOutlierDetector(double threshold);

		PrimitiveSet remove_outliers(const PrimitiveSet& ps, const PrimitiveSetRanker& ranker) const;

	private:
		double threshold;
	};
	
	struct OutlierDetector
	{
		OutlierDetector(const std::string& python_module_path);
		~OutlierDetector();

		PrimitiveSet remove_outliers(const PrimitiveSet& ps, const PrimitiveSetRank& psr) const;

	private:
		PyObject *od_method_name, *od_module, *od_dict, *od_method;		
	};

	struct SimilarityFilter
	{
		SimilarityFilter(double epsilon, double voxel_size, bool similarity_only, double perfectness_t);

		PrimitiveSet filter(const PrimitiveSet& ps, const PrimitiveSetRanker& ranker);

	private:

		double epsilon;
		double voxel_size;
		bool similarity_only;
		double perfectness_t;
		
	};

	struct CapOptimizer
	{
		CapOptimizer(double cap_plane_adjustment_max_dist);

		CSGNode optimize_caps(const PrimitiveSet& ps, const CSGNode& inp_node);

	private: 
		double cap_plane_adjustment_max_dist;
	};
}

#endif 