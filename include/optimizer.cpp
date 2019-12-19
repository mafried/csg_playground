#include "optimizer.h"
#include "csgnode.h"


bool does_not_overlap(const lmu::CSGNode& n1, const lmu::CSGNode& n2)
{
	return true;
}

bool is_same(const lmu::CSGNode& n1, const lmu::CSGNode& n2)
{
	return true;
}

bool is_op(const lmu::CSGNode& n)
{
	return n.type() == lmu::CSGNodeType::Operation; 
}

void process_node(lmu::CSGNode& n) 
{
	static const no_op = lmu::optimizeCSGNode

	if (!is_op(n)) return; 

	const auto& op1 = n.childs()[0];

	switch (n.operationType())
	{
	case lmu::CSGNodeOperationType::Intersection:

		const auto& op2 = n.childs()[1];

		// Both operands are the same => replace with one operand.
		if (is_same(op1, op2)) n = op1;

		// Both operands' volumes do not intersect => replace with empty set.
		else if (does_not_overlap(op1, op2)) n = 

		break;
	}
}

lmu::CSGNode lmu::remove_redundancies(const CSGNode& node, double sampling_grid_size)
{
	auto opt_node = node;

	while (true)
	{

	}

	return opt_node;
}
