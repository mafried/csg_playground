#ifndef OPTIMIZER_GA_H
#define OPTIMIZER_GA_H

#include "csgnode.h"

namespace lmu 
{
	enum class GeoScoreStrategy
	{
		SURFACE_SAMPLES,
		IN_OUT_SAMPLES
	};

	struct RankerParams
	{
		CSGNodeSamplingParams sampling_params;

		double gradient_step_size;
		double position_tolerance;

		double geo_score_weight; 
		double prox_score_weight;
		double size_score_weight; 

		int max_sampling_points;
			
		GeoScoreStrategy geo_score_strat;
	};

	struct CreatorParams
	{		
		double create_new_prob;
		double subtree_prob;
		std::vector<double> initial_population_dist;
	};

	struct ManipulatorParams
	{
		double max_delta;
	};

	struct GAParams
	{
		int population_size;
		int num_best_parents;
		double mutation_rate;
		double crossover_rate;
		bool in_parallel;
		bool use_caching;
		int tournament_k; 
		int max_iterations;
	};

	struct OptimizerGAParams
	{
		RankerParams ranker_params;
		CreatorParams creator_params;
		ManipulatorParams man_params;
		GAParams ga_params;
	};

	struct OptimizerGAResult
	{
		OptimizerGAResult(const CSGNode& node) : 
			node(node)
		{
		}

		CSGNode node; 
	};

	OptimizerGAResult optimize_with_ga(const CSGNode& node, const OptimizerGAParams& params, std::ostream& report_stream, const std::vector<ImplicitFunctionPtr>& primitives = {});

	const double invalid_proximity_score = -1.0;

	double compute_local_proximity_score(const CSGNode& node, double sampling_grid_size, 
		const lmu::PointCloud& sampling_points);
}

#endif
