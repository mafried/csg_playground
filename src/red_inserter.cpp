#include "red_inserter.h"

#include <random>
#include <algorithm>
#include "csgnode.h"
#include "csgnode_helper.h"

using namespace lmu;

std::ostream& lmu::operator <<(std::ostream& stream, const lmu::InserterType& it)
{
	std::string it_str;
	switch (it)
	{
	case InserterType::SubtreeCopy:
		it_str = "Subtree Copy";
		break;
	case InserterType::DoubleNegation:
		it_str = "Double Negation";
		break;
	case InserterType::Distributive:
		it_str = "Distributive";
		break;


	}

	stream << it_str;

	return stream;
}

Inserter lmu::inserter(InserterType type, double probability)
{
	switch (type)
	{
	case InserterType::SubtreeCopy:
		return Inserter(std::make_shared<SubtreeCopyInserter>(), probability);
	case InserterType::DoubleNegation:
		return Inserter(std::make_shared<DoubleNegationInserter>(), probability);
	case InserterType::Distributive:
		return Inserter(std::make_shared<DistributiveInserter>(), probability);

	}
}

CSGNode lmu::inflate_node(const CSGNode& node, int iterations, const std::vector<Inserter>& inserter)
{
	CSGNode inflated_node = node;


	std::vector<double> probs;
	std::transform(inserter.begin(), inserter.end(), std::back_inserter(probs), [](const Inserter& ins) {return ins.propability(); });
	
	std::default_random_engine generator;
	std::discrete_distribution<int> distribution(probs.begin(), probs.end());
	
	for (int i = 0; i < iterations; ++i)
	{
		int inserter_index = distribution(generator);
		int node_idx = std::uniform_int_distribution<int>(0, numNodes(inflated_node) - 1)(generator);

		CSGNode* sub_node = nodePtrAt(inflated_node, node_idx);

		bool worked = inserter[inserter_index].inflate(*sub_node);

		std::cout << "Applied " << inserter[inserter_index].type() << " at node idx " << node_idx << "." << std::endl;
		if (!worked)
			std::cout << "No effect." << std::endl;
	}

	return inflated_node;
}


bool lmu::SubtreeCopyInserter::inflate(CSGNode & node) const
{
	static std::default_random_engine generator;
	
	if (std::bernoulli_distribution(0.5)(generator))
	{
		node = opUnion({ node, node });
	}
	else
	{
		node = opInter({ node, node });
	}

	return true;
}

std::shared_ptr<IInserter> lmu::SubtreeCopyInserter::clone() const
{
	return std::make_shared<SubtreeCopyInserter>(*this);
}

InserterType lmu::SubtreeCopyInserter::type() const
{
	return InserterType::SubtreeCopy;
}


bool lmu::DoubleNegationInserter::inflate(CSGNode & node) const
{	
	node = opComp({ opComp({ node }) });

	return true;
}

std::shared_ptr<IInserter> lmu::DoubleNegationInserter::clone() const
{
	return std::make_shared<DoubleNegationInserter>(*this);
}

InserterType lmu::DoubleNegationInserter::type() const
{
	return InserterType::DoubleNegation;
}


bool lmu::DistributiveInserter::inflate(CSGNode& node) const
{
	static std::default_random_engine generator;
	const auto& childs = node.childsCRef();

	bool did_something = false;

	if (node.type() == CSGNodeType::Operation)
	{
		switch (node.operationType())
		{
		case CSGNodeOperationType::Intersection:
		{
			std::vector<int> union_indices;
			for (int i = 0; i < childs.size(); ++i)
				if (childs[i].type() == CSGNodeType::Operation && 
					childs[i].operationType() == CSGNodeOperationType::Union)
					union_indices.push_back(i);
				
			if (!union_indices.empty())
			{
				int union_idx = union_indices[std::uniform_int_distribution<int>(0, union_indices.size() - 1)(generator)];
				int other_idx = union_idx == 1 ? 0 : 1;
				
				if (childs[union_idx].childsCRef().size() >= 2)
				{
					node = opUnion({ opInter({childs[union_idx].childsCRef()[0], childs[other_idx]}),
						 opInter({ childs[union_idx].childsCRef()[1], childs[other_idx] }) });
					did_something = true;
				}
			}

			break;
		}
		case CSGNodeOperationType::Union:
		{
			std::vector<int> inter_indices;
			for (int i = 0; i < childs.size(); ++i)
				if (childs[i].type() == CSGNodeType::Operation &&
					childs[i].operationType() == CSGNodeOperationType::Intersection)
					inter_indices.push_back(i);

			if (!inter_indices.empty())
			{
				int inter_idx = inter_indices[std::uniform_int_distribution<int>(0, inter_indices.size() - 1)(generator)];
				int other_idx = inter_idx == 1 ? 0 : 1;

				if (childs[inter_idx].childsCRef().size() >= 2)
				{
					node = opInter({ opUnion({ childs[inter_idx].childsCRef()[0], childs[other_idx] }),
						opUnion({ childs[inter_idx].childsCRef()[1], childs[other_idx] }) });
					did_something = true;
				}
			}

			break;
		}
		}
	}

	return did_something;
}

std::shared_ptr<IInserter> lmu::DistributiveInserter::clone() const
{
	return std::make_shared<DistributiveInserter>(*this);
}

InserterType lmu::DistributiveInserter::type() const
{
	return InserterType::Distributive;
}
