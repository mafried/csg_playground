#ifndef PIPELINE_RUNNER_H
#define PIPELINE_RUNNER_H

#include "params.h"
#include "optimizer_ga.h"
#include "optimizer_clustering.h"
#include "cit.h"

namespace lmu
{
	struct PipelineParams
	{
		std::string optimizer; 
		std::string tree_file;
		double sampling_grid_size;
		bool save_meshes;	

		bool use_decomposition;
		bool use_redundancy_removal;

		bool use_cit_points_for_redundancy_removal;
		bool use_cit_points_for_decomposition;
	};

	struct SamplingParams
	{
		bool use_cit_points_for_pi_extraction;
		double sampling_grid_size;
		std::string python_interpreter_path;
	};
	
	struct CITSampling
	{
		CITSampling() :
			in(lmu::empty_pc()), out(lmu::empty_pc()), in_out(lmu::empty_pc())
		{
		}

		lmu::PointCloud in;
		lmu::PointCloud out;
		lmu::PointCloud in_out;

		lmu::CITSets in_sets;
		lmu::CITSets out_sets;

		bool empty() const
		{
			return in.rows() == 0;
		}
	};

	struct PipelineRunner
	{
		PipelineRunner(const std::string& input_config, const std::string& output_folder);

		int run();

	private:

		CSGNode load(const PipelineParams& pp);

		CSGNode optimize(const CSGNode& node, const PrimitiveCluster& prims,
			const PipelineParams& pp, std::ofstream& opt_out, std::ofstream& timings);

		PipelineParams read_pipeline_params(const ParameterSet& params);
		SamplingParams read_opt_sampling_params(const ParameterSet& params);
		OptimizerGAParams read_opt_ga_params(const ParameterSet & params);

		ParameterSet params;
		std::string output_folder;
	};
}

#endif