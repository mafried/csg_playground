#ifndef DNF_H
#define DNF_H

#include <vector>
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

		double signedDistance(const Eigen::Vector3d& p, const std::vector<ImplicitFunctionPtr>& functions) const
		{
			double res = -std::numeric_limits<double>::max();
			for (int i = 0; i < size(); ++i)
			{
				auto childRes = literals[i] ? (negated[i] ? -1.0 : 1.0) * functions[i]->signedDistance(p) : -std::numeric_limits<double>::max();
				res = childRes > res ? childRes : res;
			}

			return res;
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

		double signedDistance(const Eigen::Vector3d& p) const
		{
			double res = std::numeric_limits<double>::max();
			for (const auto& clause : clauses)
			{		
				auto childRes = clause.signedDistance(p, functions);
				res = childRes < res ? childRes : res;
			}

			return res;
		}
	};

	struct SampleParams
	{
		double h;
		double distThreshold;
		double angleThreshold;
	};

	std::ostream& operator <<(std::ostream& stream, const Clause& c);
	
	struct Graph;
	
	CSGNode DNFtoCSGNode(const DNF& dnf);
	CSGNode clauseToCSGNode(const Clause& clause, const std::vector<ImplicitFunctionPtr>& functions);
	
	std::tuple<Clause, double, double> scoreClause(const Clause& clause, const std::vector<ImplicitFunctionPtr>& functions, 
		int numClauseFunctions, const std::unordered_map<lmu::ImplicitFunctionPtr, std::tuple<double, double>>& outlierTestValues, 
		const lmu::Graph& conGraph, const SampleParams& params);
	
	std::unordered_map<lmu::ImplicitFunctionPtr, std::tuple<double, double>> computeOutlierTestValues(const std::vector<lmu::ImplicitFunctionPtr>& functions, double h);

	DNF computeShapiro(const std::vector<ImplicitFunctionPtr>& functions, bool usePrimeImplicantOptimization, const lmu::Graph& conGraph, const SampleParams& params);
	DNF mergeDNFs(const std::vector<DNF>& dnfs);

	std::string espressoExpression(const DNF& dnf);

	CSGNode computeShapiroWithPartitions(const std::vector<Graph>& partitions, const SampleParams& params);

	std::vector<Graph> getUnionPartitionsByPrimeImplicants(const Graph& graph, const SampleParams& params);

	std::vector<Graph> getUnionPartitionsByPrimeImplicantsWithPruning(const Graph& graph, const SampleParams& params);

	std::vector<Graph> getUnionPartitionsByArticulationPoints(const Graph& graph);

	void print_clause(std::ostream & stream, const lmu::Clause & c, const std::vector<lmu::ImplicitFunctionPtr>& functions, bool printNonSetLiterals);

}

bool operator==(const lmu::Clause& lhs, const lmu::Clause& rhs);

#endif