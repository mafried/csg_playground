#ifndef POLYTOPE_EXTRACTION_H
#define POLYTOPE_EXTRACTION_H

#include "primitive_extraction.h"

namespace lmu
{
	struct IntermediateConvexCluster
	{
		std::set<lmu::ManifoldPtr> planes;
		std::vector<Eigen::Matrix<double, 1, 6>> points;
	};

	struct ConvexCluster
	{
		ConvexCluster(const IntermediateConvexCluster& icc, bool rem_outliers);

		lmu::ManifoldSet planes;
		PointCloud pc;

		Eigen::Vector3d compute_center(const lmu::ModelSDF& msdf) const;

		void remove_outliers();
	};

	//std::vector<ConvexCluster> get_convex_clusters(const lmu::PointCloud& pc, const std::vector<int>& pc_to_plane_idx, 
	//	const lmu::ManifoldSet& planes, const std::string& python_script, double am_clustering_param);

	std::vector<ConvexCluster> get_convex_clusters_without_planes(const std::string& cluster_file, bool remove_outliers);

	std::vector<ConvexCluster> get_convex_clusters(PlaneGraph& pg, const std::string& python_script, double am_clustering_param);

	PrimitiveSet generate_polytopes(const std::vector<ConvexCluster>& convex_clusters, const PlaneGraph& plane_graph,
		const lmu::PrimitiveGaParams& params, std::ofstream& s);

	PrimitiveSet merge_polytopes(const lmu::PrimitiveSet& ps, double am_quality_threshold);

}

#endif 