#ifndef CSGNODE_HELPER_H
#define CSGNODE_HELPER_H

#include <vector>
#include <memory>

#include "csgnode.h"

namespace lmu
{


	CSGNode geometry(ImplicitFunctionPtr function);
	
	template<class T, class... Types>
	CSGNode geo(Types&&... args)
	{	// make a shared_ptr
		return geometry(std::make_shared<T>(std::forward<Types>(args)...));
	}

	using Union = UnionOperation;
	using Intersection = IntersectionOperation;
	using Difference = DifferenceOperation;


	template<class T>
	CSGNode op(const std::vector<CSGNode>& childs = {})
	{	// make a shared_ptr
		return CSGNode(std::make_shared<T>("", childs));
	}
	
	CSGNode opUnion(const std::vector<CSGNode>& childs = {});
	CSGNode opDiff(const std::vector<CSGNode>& childs = {});
	CSGNode opInter(const std::vector<CSGNode>& childs = {});
	CSGNode opComp(const std::vector<CSGNode>& childs = {});
}

#endif