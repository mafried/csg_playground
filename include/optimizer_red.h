#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <math.h>
#include <unordered_map>

namespace lmu 
{
	class CSGNode;

	using EmptySetLookup = ::std::unordered_map<size_t, bool>;

	CSGNode to_binary_tree(const CSGNode& node);

	CSGNode remove_redundancies(const CSGNode& node, double sampling_grid_size);

	CSGNode transform_to_diffs(const CSGNode& node);
	
	bool is_empty_set(const CSGNode& n, double sampling_grid_size, EmptySetLookup& esLookup);
}

#endif
