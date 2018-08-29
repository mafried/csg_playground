#ifndef DNF_H
#define DNF_H

#include <vector>
//#include <boost/dynamic_bitset.hpp>
#include <boost/container/vector.hpp>

#include "csgnode.h"

namespace lmu
{
	struct Clause
	{
		Clause(int size) : 
			literals(size, false),
			negated(size, false)
		{
		}

		Clause()
		{
		}

		void clearAll()
		{
			literals = boost::container::vector<bool>(literals.size(), false);
			negated = boost::container::vector<bool>(negated.size(), false);
		}

		size_t size() const
		{
			return literals.size();
		}

		boost::container::vector<bool> literals; 
		boost::container::vector<bool> negated;
	};

	extern Eigen::MatrixXd g_testPoints;
	extern Clause g_clause;


	struct DNF
	{
		std::vector<Clause> clauses; 		
		std::vector<ImplicitFunctionPtr> functions;
	};

	struct SampleParams
	{
		double h;
	};

	std::ostream& operator <<(std::ostream& stream, const Clause& c);

	struct Graph;
	
	CSGNode DNFtoCSGNode(const DNF& dnf);
	CSGNode clauseToCSGNode(const Clause& clause, const std::vector<ImplicitFunctionPtr>& functions);
	
	std::tuple<Clause, double, double> scoreClause(const Clause& clause, const std::vector<ImplicitFunctionPtr>& functions, const std::unordered_map<lmu::ImplicitFunctionPtr, std::tuple<double, double>> outlierTestValues, const lmu::Graph& conGraph, const SampleParams& params);
	
	std::unordered_map<lmu::ImplicitFunctionPtr, std::tuple<double, double>> computeOutlierTestValues(const std::vector<lmu::ImplicitFunctionPtr>& functions, double h);


	DNF computeShapiro(const std::vector<ImplicitFunctionPtr>& functions, bool usePrimeImplicantOptimization, const lmu::Graph& conGraph, const SampleParams& params);
	DNF mergeDNFs(const std::vector<DNF>& dnfs);

	std::string espressoExpression(const DNF& dnf);
}

#endif