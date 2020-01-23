#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <math.h>
#include <unordered_map>
#include "pointcloud.h"

namespace lmu 
{
	class CSGNode;

	using EmptySetLookup = ::std::unordered_map<size_t, bool>;

	CSGNode to_binary_tree(const CSGNode& node);

	CSGNode remove_redundancies(const CSGNode& node, double sampling_grid_size,
		const lmu::PointCloud& sampling_points);

	CSGNode transform_to_diffs(const CSGNode& node);
	
	bool is_empty_set(const CSGNode& n, double sampling_grid_size, const lmu::PointCloud& sampling_points, 
		EmptySetLookup& esLookup);
}

#endif
