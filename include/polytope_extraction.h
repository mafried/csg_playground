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
		ConvexCluster(const IntermediateConvexCluster& icc);

		lmu::ManifoldSet planes;
		PointCloud pc;
	};

	std::vector<ConvexCluster> get_convex_clusters(lmu::PlaneGraph& pg, double max_point_dist, const std::string& python_script);

	PrimitiveSet generate_polytopes(const std::vector<ConvexCluster>& convex_clusters, const PlaneGraph& plane_graph,
		const lmu::PrimitiveGaParams& params, std::ofstream& s);
}

#endif 