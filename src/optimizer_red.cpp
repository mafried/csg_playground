#include "optimizer_red.h"
#include "csgnode.h"
#include "csgnode_helper.h"

bool lmu::is_empty_set(const CSGNode& n, double sampling_grid_size, EmptySetLookup& esLookup)
{
	// Check if lookup contains value for node already.
	//size_t node_hash = n.hash(0);
	//auto it = esLookup.find(node_hash);
	//if (it != esLookup.end())
	//	return it->second;

	lmu::AABB aabb = aabb_from_node(n);
	Eigen::Vector3d min = aabb.c - aabb.s; 
	Eigen::Vector3d max = aabb.c + aabb.s;

	bool is_empty = _is_empty_set(n, sampling_grid_size, min, max);

	// Store in lookup table.
	//esLookup[node_hash] = is_empty;

	return is_empty; 
}

bool do_not_overlap(const lmu::CSGNode& n1, const lmu::CSGNode& n2, double sampling_grid_size, lmu::EmptySetLookup& esLookup)
{
	return is_empty_set(lmu::opInter({ n1,n2 }), sampling_grid_size, esLookup);
}

bool are_same(const lmu::CSGNode& n1, const lmu::CSGNode& n2, double sampling_grid_size, lmu::EmptySetLookup& esLookup)
{
	return is_empty_set(lmu::opDiff({ n1,n2 }), sampling_grid_size, esLookup);
}

bool has_empty_marker(const lmu::CSGNode& n)
{
	return n.operationType() == lmu::CSGNodeOperationType::Noop && n.name() == "0";
}

bool has_all_marker(const lmu::CSGNode& n)
{
	return n.operationType() == lmu::CSGNodeOperationType::Noop && n.name() == "1";
}

bool is_valid_op(const lmu::CSGNode& n)
{
	return n.isValid() && n.type() == lmu::CSGNodeType::Operation && n.operationType() != lmu::CSGNodeOperationType::Noop &&
		n.childsCRef().size() >= std::get<0>(n.numAllowedChilds()) && n.childsCRef().size() <= std::get<1>(n.numAllowedChilds());
}

bool process_node(lmu::CSGNode& n, double sampling_grid_size, lmu::EmptySetLookup& esLookup)
{
	static auto const empty_set = lmu::CSGNode(std::make_shared<lmu::NoOperation>("0"));
	static auto const all = lmu::CSGNode(std::make_shared<lmu::NoOperation>("1"));

	if (!is_valid_op(n)) return false;

	const auto& op1 = n.childsCRef()[0];
	bool something_has_changed = true;

	switch (n.operationType())
	{
	// Intersection 
	case lmu::CSGNodeOperationType::Intersection:
	{
		const auto& op2 = n.childsCRef()[1];

		if (has_empty_marker(op1) || has_empty_marker(op2)) n = empty_set;

		else if (has_all_marker(op1)) n = op2;

		else if (has_all_marker(op2)) n = op1;

		else if (are_same(op1, op2, sampling_grid_size, esLookup)) n = op1;

		else if (do_not_overlap(op1, op2, sampling_grid_size, esLookup)) n = empty_set;

		else something_has_changed = false;

		break;
	}
	// Union 
	case lmu::CSGNodeOperationType::Union:
	{
		const auto& op2 = n.childsCRef()[1];

		if (has_empty_marker(op1) && has_empty_marker(op2)) n = empty_set;

		else if (has_empty_marker(op1)) n = op2; 

		else if (has_empty_marker(op2)) n = op1;

		else if (has_all_marker(op1)) n = all;

		else if (has_all_marker(op2)) n = all;

		else if (are_same(op1, op2, sampling_grid_size, esLookup)) n = op1;

		else something_has_changed = false;

		break;
	}
	// Difference 
	case lmu::CSGNodeOperationType::Difference:
	{
		const auto& op2 = n.childsCRef()[1];

		if (are_same(op1, op2, sampling_grid_size, esLookup)) n = empty_set;

		else if (has_empty_marker(op1)) n = empty_set;

		else if (has_empty_marker(op2)) n = op1;

		else if (has_all_marker(op1)) n = lmu::opComp({ op2 });

		else if (has_all_marker(op2)) n = empty_set;
		
		else something_has_changed = false;

		break;
	}
	// Complement
	case lmu::CSGNodeOperationType::Complement:
	{
		if (op1.operationType() == lmu::CSGNodeOperationType::Complement)
		{
			n = op1.childsCRef()[0];
		}
		else 
			something_has_changed = false;
	}
	default: 
		something_has_changed = false;
	}

	return something_has_changed;
}

bool process_node_rec(lmu::CSGNode& n, double sampling_grid_size, lmu::EmptySetLookup& esLookup)
{
	if (process_node(n, sampling_grid_size, esLookup))
	{
		return true;
	}
	else
	{
		bool something_has_changed = false;
		for (auto& child : n.childsRef())
		{
			something_has_changed = process_node_rec(child, sampling_grid_size, esLookup);
		}
		return something_has_changed;
	}	
}

lmu::CSGNode lmu::remove_redundancies(const CSGNode& node, double sampling_grid_size)
{
	auto opt_node = node;
	bool something_has_changed = true;
	EmptySetLookup esLookup;

	while (something_has_changed)
	{
		something_has_changed = process_node_rec(opt_node, sampling_grid_size, esLookup);
	}
	return opt_node;
}
