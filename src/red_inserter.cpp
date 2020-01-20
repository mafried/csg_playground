#include "red_inserter.h"

#include <random>
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

		inserter[inserter_index].inflate(*sub_node);
	}

	return inflated_node;
}


void lmu::SubtreeCopyInserter::inflate(CSGNode & node) const
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
}

std::shared_ptr<IInserter> lmu::SubtreeCopyInserter::clone() const
{
	return std::make_shared<SubtreeCopyInserter>(*this);
}

InserterType lmu::SubtreeCopyInserter::type() const
{
	return InserterType::SubtreeCopy;
}


void lmu::DoubleNegationInserter::inflate(CSGNode & node) const
{	
	node = opComp({ opComp({ node }) });
}

std::shared_ptr<IInserter> lmu::DoubleNegationInserter::clone() const
{
	return std::make_shared<DoubleNegationInserter>(*this);
}

InserterType lmu::DoubleNegationInserter::type() const
{
	return InserterType::DoubleNegation;
}
