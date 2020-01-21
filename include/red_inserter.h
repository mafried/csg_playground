#ifndef RED_INSERTER_H
#define RED_INSERTER_H

#include <memory>
#include <vector>

namespace lmu 
{
	class CSGNode;

	enum class InserterType
	{
		SubtreeCopy,
		DoubleNegation,
		Distributive
	};
	std::ostream& operator <<(std::ostream& stream, const InserterType& it);
	
	struct IInserter
	{
		virtual bool inflate(CSGNode& node) const = 0;
		virtual std::shared_ptr<IInserter> clone() const = 0;
		virtual InserterType type() const = 0;
	};

	struct Inserter : IInserter 
	{
		Inserter(const std::shared_ptr<IInserter>& inserter, double probability) :
			inserter_ptr(inserter),
			probability(probability)
		{
		}

		Inserter(const Inserter& inserter) :
			inserter_ptr(inserter.clone()),
			probability(inserter.probability)
		{			
		}

		Inserter& operator = (const Inserter& other)
		{
			if (this != &other)
			{
				inserter_ptr = other.clone();
				probability = other.probability;
			}

			return *this;
		}

		bool inflate(CSGNode& node) const override
		{
			return inserter_ptr->inflate(node);
		}
		
		std::shared_ptr<IInserter> clone() const override 
		{
			return inserter_ptr->clone();
		}	

		InserterType type() const override
		{
			return inserter_ptr->type();
		}

		double propability() const
		{
			return probability;
		}

	private:


		std::shared_ptr<IInserter> inserter_ptr;
		double probability;

	};

	struct SubtreeCopyInserter : IInserter
	{
		virtual bool inflate(CSGNode& node) const override;
		virtual std::shared_ptr<IInserter> clone() const override;
		virtual InserterType type() const override;
	};

	struct DoubleNegationInserter : IInserter
	{
		virtual bool inflate(CSGNode& node) const override;
		virtual std::shared_ptr<IInserter> clone() const override;
		virtual InserterType type() const override;
	};

	struct DistributiveInserter : IInserter
	{
		virtual bool inflate(CSGNode& node) const override;
		virtual std::shared_ptr<IInserter> clone() const override;
		virtual InserterType type() const override;
	};

	Inserter inserter(InserterType type, double probability);
	
	CSGNode inflate_node(const CSGNode& node, int iterations, const std::vector<Inserter>& inserter);
}

#endif
