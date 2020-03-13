#ifndef OPTIMIZER_QA_H
#define OPTIMIZER_QA_H

#include "csgnode.h"
#include "cit.h"

namespace lmu 
{
	CSGNode optimize_with_qa(const CSGNode& n, double sampling_grid_size, const std::vector<ImplicitFunctionPtr>& primitives,
		const PythonInterpreter& interpreter);
}

#endif
