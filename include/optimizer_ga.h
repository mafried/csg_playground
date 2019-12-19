#ifndef OPTIMIZER_GA_H
#define OPTIMIZER_GA_H

#include "csgnode.h"

namespace lmu 
{
	struct RankerParams
	{
		CSGNodeSamplingParams sampling_params;

		double gradient_step_size;
		double position_tolerance;

		double geo_score_weight; 
		double size_score_weight; 

		int max_sampling_points;
	};

	struct CreatorParams
	{		
		double create_new_prob;
		double subtree_prob;
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

	OptimizerGAResult optimize_with_ga(const CSGNode& node, const OptimizerGAParams& params, std::ostream& report_stream);
}

#endif
