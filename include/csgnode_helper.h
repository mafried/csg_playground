#ifndef CSGNODE_HELPER_H
#define CSGNODE_HELPER_H

#include <vector>
#include <memory>

#include "json.hpp"
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
	CSGNode opNo(const std::vector<CSGNode>& childs = {});

	using json = nlohmann::json;

	CSGNode fromJSON(const json& json);
	CSGNode fromJSONFile(const std::string& file);

	json toJSON(const CSGNode& node);
	void toJSONFile(const CSGNode& node, const std::string& file);


	lmu::AABB aabb_from_node(const lmu::CSGNode& n);
	lmu::AABB aabb_from_primitives(const std::vector<lmu::ImplicitFunctionPtr>& prims);

	bool _is_empty_set(const lmu::CSGNode& n, double sampling_grid_size,
		const Eigen::Vector3d& min, const Eigen::Vector3d& max);

	bool _is_in(const lmu::ImplicitFunctionPtr& primitive, const lmu::CSGNode& n, double sampling_grid_size,
		const Eigen::Vector3d& min, const Eigen::Vector3d& max);

}

#endif