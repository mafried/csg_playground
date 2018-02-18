#include "csgnode_helper.h"

using namespace lmu;

CSGNode lmu::geometry(ImplicitFunctionPtr function)
{
	return CSGNode(std::make_shared<CSGNodeGeometry>(function));
}
CSGNode lmu::opUnion(const std::vector<CSGNode>& childs)
{
	return CSGNode(std::make_shared<UnionOperation>("", childs));
}
CSGNode lmu::opDiff(const std::vector<CSGNode>& childs)
{
	return CSGNode(std::make_shared<DifferenceOperation>("", childs));
}
CSGNode lmu::opInter(const std::vector<CSGNode>& childs)
{
	return CSGNode(std::make_shared<IntersectionOperation>("", childs));
}