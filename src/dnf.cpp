#include "dnf.h"
#include "csgnode_helper.h"
#include "curvature.h"
#include "congraph.h"

#include <algorithm>
#include <unordered_map>
#include <Eigen/Core>

#include <boost/math/special_functions/erf.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/copy.hpp>

#include "statistics.h"
#include "helper.h"

Eigen::MatrixXd lmu::g_testPoints;
lmu::Clause lmu::g_clause;


lmu::CSGNode lmu::DNFtoCSGNode(const DNF& dnf)
{
	CSGNode node = opUnion();

	for (const auto& clause : dnf.clauses)
		node.addChild(clauseToCSGNode(clause, dnf.functions));

	return node;
}

lmu::CSGNode lmu::clauseToCSGNode(const Clause& clause, const std::vector<ImplicitFunctionPtr>& functions)
{
	CSGNode node = opInter();

	for (int i = 0; i < functions.size(); ++i)
	{
		if (!clause.literals[i])
			continue;

		if (clause.negated[i])
			node.addChild(opComp({ geometry(functions[i]) }));
		else
			node.addChild(geometry(functions[i]));
	}	

	return node.childsCRef().size() == 1 ? node.childsCRef()[0] : node;
}

std::ostream& lmu::operator <<(std::ostream& stream, const lmu::Clause& c)
{
	for (int i = 0; i < c.literals.size(); ++i)
	{
		if (c.negated[i])
			stream << "!";

		stream << c.literals[i];
	}
	return stream;
}

void print (std::ostream& stream, const lmu::Clause& c, const std::vector<lmu::ImplicitFunctionPtr>& functions, bool printNonSetLiterals)
{
	for (int i = 0; i < c.literals.size(); ++i)
	{
		if (c.negated[i])
			stream << "!";

		if(printNonSetLiterals || c.literals[i])
			stream << functions[i]->name();
	}
}

inline double median(std::vector<double> v)
{
	std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
	return v[v.size() / 2];
}

//https://www.mathworks.com/help/matlab/ref/isoutlier.html
std::tuple<double,double> scaled3MADAndMedian(const lmu::ImplicitFunctionPtr& func, double h)
{	
	lmu::CSGNode node = lmu::geometry(func);

	std::vector<double> values(func->pointsCRef().rows());

	for (int j = 0; j < func->pointsCRef().rows(); ++j)
	{

		Eigen::Matrix<double, 1, 6> pn = func->pointsCRef().row(j);

		Eigen::Vector3d p = pn.leftCols(3);
		Eigen::Vector3d n = pn.rightCols(3);
			
		lmu::Curvature c = curvature(p, node, h);

		values[j] = std::sqrt(c.k1 * c.k1 + c.k2 * c.k2);
	}

	double med = median(values);
	std::transform(values.begin(), values.end(), values.begin(), [med](double v) -> double { return std::abs(v - med); });

	const double c = -1.0 / (std::sqrt(2.0)*boost::math::erfc_inv(3.0 / 2.0));
	
	return std::make_tuple(c * median(values) * 3.0, med);
}

std::unordered_map<lmu::ImplicitFunctionPtr, std::tuple<double, double>> lmu::computeOutlierTestValues(const std::vector<lmu::ImplicitFunctionPtr>& functions, double h)
{
	std::unordered_map<lmu::ImplicitFunctionPtr, std::tuple<double, double>> map;

	std::cout << "----------------------------" << std::endl;
	std::cout << "Deviation from flatness outliers: " << std::endl;
	for (const auto& func : functions)
	{
		map[func] = scaled3MADAndMedian(func, h);
		std::cout << func->name() << ": " << std::get<0>(map[func]) << " Mean: " << std::get<1>(map[func]) << std::endl;
	}
	std::cout << "----------------------------" << std::endl;


	return map;
}

double getInOutThreshold(const std::vector<double>& qualityValues)
{
	const int k = 2;

	//K-MEANS
	auto res = lmu::k_means(qualityValues, k , 300);
	std::cout << "Means: " << std::endl;
	for (auto m : res.means)
		std::cout << m << std::endl;

	std::vector<double> min( k, std::numeric_limits<double>::max());
	size_t i = 0;
	for (auto t : res.assignments)
	{
		min[t] = qualityValues[i] < min[t] ? qualityValues[i] : min[t];
		i++;
	}

	std::sort(min.begin(), min.end());

	for (auto m : min)
		if (m <= 1.0)
			return m;

	return -1.0;

	/*
	//JENKS BREAKS
	ValueCountPairContainer sortedUniqueValueCounts;
	GetValueCountPairs(sortedUniqueValueCounts, &qualityValues[0], qualityValues.size());

	std::cout << "Finding Jenks ClassBreaks..." << std::endl;
	LimitsContainer resultingbreaksArray;
	ClassifyJenksFisherFromValueCountPairs(resultingbreaksArray, k, sortedUniqueValueCounts);

	std::cout << "Breaks: " << std::endl;
	for (const auto& b : resultingbreaksArray)
		std::cout << b << std::endl;

	return resultingbreaksArray[1];*/
}

std::vector<std::tuple<lmu::Clause, size_t>> getValidClauses(const std::vector<std::tuple<lmu::Clause, double, double>>& clauseQualityPairs)
{
	std::vector<double> distQualityValues;
	std::vector<double> angleQualityValues;

	std::transform(clauseQualityPairs.begin(), clauseQualityPairs.end(), std::back_inserter(distQualityValues),
		[](auto p) { return std::get<1>(p); });

	std::transform(clauseQualityPairs.begin(), clauseQualityPairs.end(), std::back_inserter(angleQualityValues),
		[](auto p) { return std::get<2>(p); });


	double distT = 0.6; //getInOutThreshold(distQualityValues);
	double angleT = 0.6; // getInOutThreshold(angleQualityValues);
	

	std::cout << "DIST T: " << distT << " ANGLE T: " << angleT;
	
	std::vector<std::tuple<lmu::Clause, size_t>> validClauses;

	for (size_t i = 0; i < clauseQualityPairs.size(); ++i)
	{
		if (std::get<1>(clauseQualityPairs[i]) >= distT && std::get<2>(clauseQualityPairs[i]) >= angleT)
			validClauses.push_back(std::make_tuple(std::get<0>(clauseQualityPairs[i]), i));
	}

	return validClauses; 
}

bool isConnected(const lmu::Clause& c, const std::vector<lmu::ImplicitFunctionPtr>& functions, const lmu::Graph& g, const lmu::ImplicitFunctionPtr& func)
{
	for (int i = 0; i < c.literals.size(); ++i)
		if (c.literals[i] && func != functions[i] && lmu::areConnected(g, func, functions[i]))
			return true;

	return false;
}

std::tuple<lmu::Clause, double, double> lmu::scoreClause(const Clause& clause, const std::vector<ImplicitFunctionPtr>& functions, int numClauseFunctions,
	const std::unordered_map<lmu::ImplicitFunctionPtr, std::tuple<double, double>>& outlierTestValues, const lmu::Graph& conGraph, const SampleParams& params)
{
	lmu::CSGNode node = clauseToCSGNode(clause, functions);

	int totalNumCorrectSamples = 0;
	int totalNumConsideredSamples = 0;

	const double smallestDelta = 0.000000001;

	std::vector<Eigen::Matrix<double, 1, 2>> consideredPoints;

	double correctSamplesPointCheck = std::numeric_limits<double>::max();

	std::cout << "Clause: ";
	print(std::cout, clause, functions, false);
	std::cout << std::endl;

	int numConsideredFunctions = 0;

	//Point position violation check.
	for (int i = 0; i < functions.size(); ++i)
	{	
		int outsideSamples = 0;
		int insideSamples = 0;
		int numCorrectSamples = 0;
		int numConsideredSamples = 0;

		//function should not be a literal of the clause to test.
		if (i < numClauseFunctions && clause.literals[i])
			continue;

		//If current primitive is not connected with primitives in clause, continue.
		if (!isConnected(clause, functions, conGraph, functions[i]))
			continue;

		//If pruned vertices should not be considered, test for it.
		//if (ignorePrunedVertices && lmu::wasPruned(conGraph, functions[i]))
		//	continue;

		numConsideredFunctions++;

		lmu::ImplicitFunctionPtr currentFunc = functions[i];		
	
		//Test if points of are inside the volume (if so => wrong node).
		for (int j = 0; j < currentFunc->pointsCRef().rows(); ++j)
		{
			Eigen::Matrix<double, 1,6> pn = currentFunc->pointsCRef().row(j);

			Eigen::Vector3d sampleP = pn.leftCols(3);
			Eigen::Vector3d sampleN = pn.rightCols(3);

			Eigen::Vector4d sampleDistGradFunction = currentFunc->signedDistanceAndGradient(sampleP);
			double sampleDistFunction = sampleDistGradFunction[0];
			Eigen::Vector3d sampleGradFunction = sampleDistGradFunction.bottomRows(3);			
	
			Eigen::Vector4d sampleDistGradNode = node.signedDistanceAndGradient(sampleP, params.h);
			double sampleDistNode = sampleDistGradNode[0];
			Eigen::Vector3d sampleGradNode = sampleDistGradNode.bottomRows(3);
			
			numConsideredSamples++;

			//If points are inside the node's volume, consider them as indicator of a wrong node.
			if (sampleDistNode < -smallestDelta)
			{	
			}
			else
			{
				numCorrectSamples++;
			}
		}
				
		totalNumConsideredSamples += numConsideredSamples;
		totalNumCorrectSamples += numCorrectSamples;


		double score = numConsideredSamples == 0 ? 1.0 : (double)numCorrectSamples / (double)numConsideredSamples;
		std::cout << currentFunc->name() << ": " << score << std::endl;
		correctSamplesPointCheck = score < correctSamplesPointCheck ? score : correctSamplesPointCheck;		
	}

	//If no function was considered then all functions are literals.
	//In that case, point check is pointless.
	if (numConsideredFunctions == 0)
		correctSamplesPointCheck = 1.0;

	totalNumConsideredSamples = 0;
	totalNumCorrectSamples = 0;

	std::cout << "-" << std::endl;

	double correctSamplesAngleCheck = std::numeric_limits<double>::max();

	//Angle violation check.
	for (int i = 0; i < functions.size(); ++i)
	{
		int numCorrectSamples = 0;
		int numConsideredSamples = 0;
		
		//function must be a literal of the clause to test.
		if (i >= numClauseFunctions || !clause.literals[i])
			continue;

		lmu::ImplicitFunctionPtr currentFunc = functions[i];
		std::tuple<double, double> outlierTestValue = outlierTestValues.at(currentFunc);

		for (int j = 0; j < currentFunc->pointsCRef().rows(); ++j)
		{
			Eigen::Matrix<double, 1, 6> pn = currentFunc->pointsCRef().row(j);

			Eigen::Vector3d sampleP = pn.leftCols(3);
			Eigen::Vector3d sampleN = pn.rightCols(3);

			Eigen::Vector4d sampleDistGradFunction = currentFunc->signedDistanceAndGradient(sampleP);
			double sampleDistFunction = sampleDistGradFunction[0];

			Eigen::Vector3d sampleGradFunction = sampleDistGradFunction.bottomRows(3);

			Eigen::Vector4d sampleDistGradNode = node.signedDistanceAndGradient(sampleP, params.h);
			double sampleDistNode = sampleDistGradNode[0];
			Eigen::Vector3d sampleGradNode = sampleDistGradNode.bottomRows(3);

			//Do not consider points that are far away from the node's surface.
			if (std::abs(sampleDistNode - sampleDistFunction) > smallestDelta)
			{
				continue;
			}
			else
			{
			}

			//Normals close to edges tend to be brittle. 
			//We try to filter normals that are located close to curvature outliers (== edges).
			Curvature c = curvature(sampleP, geometry(currentFunc), params.h);
			double deviationFromFlatness = std::sqrt(c.k1 * c.k1 + c.k2 * c.k2);
			double median = std::get<1>(outlierTestValue);
			double maxDelta = std::get<0>(outlierTestValue);
			if (std::abs(deviationFromFlatness - median) > maxDelta)
			{	
				//std::cout << deviationFromFlatness << " ";

				//Eigen::Matrix<double, 1, 6> m;
				//m << sampleP.transpose(), sampleN.transpose();
				//g_testPoints.conservativeResize(g_testPoints.rows() + 1, 6);
				//g_testPoints.row(g_testPoints.rows() - 1) = m;

				continue;
			}
			else
			{
			}

			numConsideredSamples++;

			//Check if normals point in the correct direction.
			if (sampleGradNode.dot(sampleN) <= 0.0)
			{
				continue;
			}
			else
			{
			}

			numCorrectSamples++;
		}

		double score = numConsideredSamples == 0 ? 1.0 : (double)numCorrectSamples / (double)numConsideredSamples;
		std::cout << currentFunc->name() << ": " << score  << std::endl;
		correctSamplesAngleCheck = score < correctSamplesAngleCheck ? score: correctSamplesAngleCheck;

		totalNumConsideredSamples += numConsideredSamples;
		totalNumCorrectSamples += numCorrectSamples;
	}

	if (totalNumConsideredSamples == 0)
		correctSamplesAngleCheck = 0.0;

	std::cout << "---------------------------------" << std::endl;
		
	return std::make_tuple(clause, correctSamplesPointCheck, correctSamplesAngleCheck);
}

std::vector<std::tuple<lmu::Clause, double, double>>  permutateAllPossibleFPs(lmu::Clause clause /*copy necessary*/, lmu::DNF& dnf, const std::unordered_map<lmu::ImplicitFunctionPtr, std::tuple<double,double>> outlierTestValues, 
	const lmu::Graph& conGraph, const lmu::SampleParams& params, int& iterationCounter)
{	
	std::vector<std::tuple<lmu::Clause, double, double>> clauses;

	std::sort(clause.negated.begin(), clause.negated.end());
	do {							
		clauses.push_back(lmu::scoreClause(clause, dnf.functions, dnf.functions.size(), outlierTestValues, conGraph, params));

		iterationCounter++;		
		std::cout << "Ready: " << (double)iterationCounter / std::pow(2, clause.negated.size()) * 100.0 << "%" << std::endl;

	} while (std::next_permutation(clause.negated.begin(), clause.negated.end()));

	return clauses;
}

std::tuple<lmu::DNF, std::vector<lmu::ImplicitFunctionPtr>> identifyPrimeImplicants(const std::vector<lmu::ImplicitFunctionPtr>& functions, 
	const std::unordered_map<lmu::ImplicitFunctionPtr, std::tuple<double, double>> outlierTestValues, const lmu::Graph& conGraph,
	const lmu::SampleParams& params)
{
	//Check which primitive is completely inside the geometry.
	std::vector<std::tuple<lmu::Clause, double, double>> clauses;
	for (int i = 0; i < functions.size(); ++i)
	{
		//Defining a clause representing a single primitive.
		lmu::Clause clause(functions.size());
		clause.literals[i] = true;

		//print(std::cout, clause, functions, false);
		//std::cout << std::endl;
		
		auto c = lmu::scoreClause(clause, functions, functions.size(), outlierTestValues, conGraph, params);
			clauses.push_back(c);			
	}

	//Create a DNF that contains a clause for each primitive.
	lmu::DNF dnf; 
	std::vector<int> functionDeleteMarker(functions.size(),0);
	auto validClauses = getValidClauses(clauses);
	int i = 0; //New index.
	for (const auto& validClause :validClauses)
	{
		lmu::Clause clause(validClauses.size());
			
		clause.literals[i] = true;

		//Index in vector that contains all clauses (valid and non-valid).
		auto idx = std::get<1>(validClause);
		dnf.functions.push_back(functions[idx]); //Add function to the dnf's functions
		
		functionDeleteMarker[idx] = 1;
	
		dnf.clauses.push_back(clause);

		i++;
	}

	//Create set of non PI functions.
	std::vector<lmu::ImplicitFunctionPtr> nonPIs;
	for (int i = 0; i < functionDeleteMarker.size(); ++i)
	{
		if (functionDeleteMarker[i] != 1)
			nonPIs.push_back(functions[i]);
	}
		
	return std::make_tuple(dnf, nonPIs);
}

lmu::DNF lmu::computeShapiro(const std::vector<ImplicitFunctionPtr>& functions, bool usePrimeImplicantOptimization, const lmu::Graph& conGraph, const SampleParams& params)
{
	DNF primeImplicantsDNF;
	DNF dnf;

	auto outlierTestValues = computeOutlierTestValues(functions, params.h);
	
	if (usePrimeImplicantOptimization)
	{
		auto res = identifyPrimeImplicants(functions, outlierTestValues, conGraph, params);
		primeImplicantsDNF = std::get<0>(res);
		dnf.functions = std::get<1>(res);		
	}
	else
	{
		dnf.functions = functions;
	}

	Clause clause(dnf.functions.size());
	std::fill(clause.literals.begin(), clause.literals.end(), true);
		
	std::cout << "Do Shapiro..." << std::endl;
	//return primeImplicantsDNF;
	
	int iterationCounter = 0;

	std::vector<std::tuple<lmu::Clause, double, double>> clauses;

	for (int i = 0; i <= dnf.functions.size(); i++)
	{	
		auto newClauses = permutateAllPossibleFPs(clause, dnf, outlierTestValues, conGraph, params, iterationCounter);
		clauses.insert(clauses.end(), newClauses.begin(), newClauses.end());
				
		if(i < dnf.functions.size())
			clause.negated[i] = true;
	}

	//Check for validity of all found clauses
	for (const auto& validClause : getValidClauses(clauses))
		dnf.clauses.push_back(std::get<0>(validClause));

	std::cout << "Done Shapiro." << std::endl;

	return lmu::mergeDNFs({ primeImplicantsDNF, dnf });
}

lmu::DNF lmu::mergeDNFs(const std::vector<DNF>& dnfs)
{
	lmu::DNF mergedDNF;
	for (const auto& dnf : dnfs)
	{
		int oldClauseSize = mergedDNF.functions.size();
		int newClauseSize = oldClauseSize + dnf.functions.size();

		if (oldClauseSize == newClauseSize)
			continue;

		//Resize existing clauses.
		for (auto& clause : mergedDNF.clauses)
		{
			clause.literals.resize(newClauseSize, false);
			clause.negated.resize(newClauseSize, false);
		}

		//Add modified new clauses.
		for (auto& clause : dnf.clauses)
		{
			lmu::Clause newClause(oldClauseSize);			
			newClause.literals.insert(newClause.literals.end(), clause.literals.begin(), clause.literals.end());
			newClause.negated.insert(newClause.negated.end(), clause.negated.begin(), clause.negated.end());

			mergedDNF.clauses.push_back(newClause);
		}
		
		mergedDNF.functions.insert(mergedDNF.functions.end(), dnf.functions.begin(), dnf.functions.end());
	}
		
	return mergedDNF;
}

std::string lmu::espressoExpression(const DNF& dnf)
{
	std::stringstream ss;

	std::stringstream literalsS;
	for (const auto& func : dnf.functions)
	{
		literalsS << func->name() << ",";
	}
	std::string literals = literalsS.str();
	literals = literals.substr(0, literals.size() - 1);

	ss << literals << "= map(exprvar, '" << literals << "'.split(','))" << std::endl;

	ss << "expr = ";

	bool firstClause = true;

	for (const auto& clause : dnf.clauses)
	{
		
		if (!firstClause)
		{
			ss << "| ";
		}
		else
		{
			firstClause = false;
		}

		bool firstLiteral = true;

		for (int i = 0; i < clause.size(); ++i)
		{

			if (clause.literals[i])
			{
				if (!firstLiteral)
				{
					ss << "& ";
				}
				else
				{
					firstLiteral = false;
				}

				if (clause.negated[i])
					ss << "~";

				ss << dnf.functions[i]->name() << " ";
			}
		}
	}

	ss << std::endl;
	ss << "dnf = expr.to_dnf()";
	
	return ss.str();
}

lmu::CSGNode lmu::computeShapiroWithPartitions(const std::vector<Graph>& partitions, const SampleParams& params)
{
	lmu::CSGNode res = lmu::op<Union>();

	for (const auto& p : partitions)
	{
		std::cout << "component functions: " << std::endl;
		for (const auto& f : lmu::getImplicitFunctions(p))
		{
			std::cout << "f: " << f->name() << std::endl;
		}
		std::cout << "----" << std::endl;

		auto dnf = lmu::computeShapiro(lmu::getImplicitFunctions(p), true, p, params);
		for (auto& const clause : dnf.clauses)
		{
			res.addChild(lmu::clauseToCSGNode(clause, dnf.functions));
		}
	}

	return res;
}


std::vector<lmu::Graph> lmu::getUnionPartitionsByPrimeImplicants(const lmu::Graph& graph, const SampleParams& params)
{
	auto functions = lmu::getImplicitFunctions(graph);
	
	auto outlierTestValues = computeOutlierTestValues(functions, params.h);

	std::cout << "PARTITION BY PIs" << std::endl;

	//Get prime implicants.
	auto res = identifyPrimeImplicants(functions, outlierTestValues, graph, params);
	auto primeImplicants = std::get<0>(res).functions;

	std::cout << "Found " << primeImplicants.size() << " prime implicants." << std::endl;
	for (const auto& f : primeImplicants)
	{
		std::cout << f->name() << std::endl;
	}

	//Remove prime implicants from graph.	
	struct Predicate 
	{
		bool operator()(GraphStructure::edge_descriptor) const 
		{ 
			return true; 
		}
		bool operator()(GraphStructure::vertex_descriptor vd) const 
		{ 
			//std::cout << g->structure[vd]->name() << ": Pruned: " << wasPruned(*g, g->structure[vd]) << " PI: " << (std::find(pis->begin(), pis->end(), g->structure[vd]) != pis->end())  << std::endl;

			return std::find(pis->begin(), pis->end(), g->structure[vd]) == pis->end(); 
		}

		const Graph* g;
		const std::vector<lmu::ImplicitFunctionPtr>* pis;		

	} predicate{ &graph, &primeImplicants};

	boost::filtered_graph<GraphStructure, Predicate, Predicate> fg(graph.structure, predicate, predicate);
	
	lmu::Graph newGraph;
	boost::copy_graph(fg, newGraph.structure);
	lmu::recreateVertexLookup(newGraph);

	//std::cout << "New graph created." << std::endl;
	//lmu::writeConnectionGraph("connectionGraph.dot", newGraph);

	//Get connected components. 
	auto partitions = lmu::getConnectedComponents(newGraph);

	//Add prime implicants as nodes 
	for (const auto& pi : primeImplicants)
	{
		lmu::Graph piGraph;
		lmu::addVertex(piGraph, pi);

		partitions.push_back(piGraph);
	}

	return partitions;
}

std::vector<lmu::Graph> lmu::getUnionPartitionsByPrimeImplicantsWithPruning(const Graph& graph, const SampleParams& params)
{
	if (numVertices(graph) <= 2)
		return { graph };
		
	//Prune graph.
	auto prunedGraph = lmu::pruneGraph(graph);
	//lmu::writeConnectionGraph("pgraph.dot", prunedGraph);
	//=> Tested, works.
	
	if (numVertices(prunedGraph) < 2)
	{
		//TODO;
	}

	//Get components separated by prime implicants.
	std::vector<lmu::Graph> partitions;

	auto components = getUnionPartitionsByPrimeImplicants(prunedGraph, params);
	//auto components = std::get<0>(componentsAndPIs);

	for (const auto& c : components)
	{
		//Only search for bridges if component has more than 2 vertices.
		if (numVertices(c) > 2)
		{
			auto bridgeComponents = getBridgeSeparatedConnectedComponents(c);
			partitions.insert(partitions.end(), bridgeComponents.begin(), bridgeComponents.end());
		}
		else if(numVertices(c) == 2)
		{
			partitions.push_back(c);
		}
		else if(numVertices(c) == 1) //components of size 1 are prime implicants. 
		{
			//Create graphs for PIs + pruned vertices.
			auto piGraph = getGraphWithPrunedVertices(graph, lmu::getImplicitFunctions(c).front());
			partitions.push_back(piGraph);
		}
	}

	return partitions;
}

std::vector<lmu::Graph> lmu::getUnionPartitionsByArticulationPoints(const Graph& graph)
{
	return getArticulationPointSeparatedConnectedComponents(graph);
}



//////////////////////////////////////////////////////////////////////
// Old optimization code
//////////////////////////////////////////////////////////////////////

bool isIn(const lmu::Clause& clause, const std::vector<lmu::ImplicitFunctionPtr>& functions, int numClauseFunctions, const std::unordered_map<lmu::ImplicitFunctionPtr,
	std::tuple<double, double>>& outlierTestValues, const lmu::Graph& conGraph, const lmu::SampleParams& params)
{
	//std::tuple<lmu::Clause, double, double> 
	auto clauseQuality = lmu::scoreClause(clause, functions, numClauseFunctions, outlierTestValues, conGraph, params);
	
	const double distT = 0.9;
	const double angleT = 0.9; 

	return (std::get<1>(clauseQuality) >= distT && std::get<2>(clauseQuality) >= angleT);
}


void computeClauseForPivotalFunction(const lmu::Clause& pivotal, int pivotalLiteral, int currentLiteral, const std::vector<lmu::ImplicitFunctionPtr>& functions, int numClauseFunctions,
	const std::unordered_map<lmu::ImplicitFunctionPtr, std::tuple<double, double>>& outlierTestValues, const lmu::SampleParams& params, std::vector<lmu::Clause>& resultClauses)
{
	if (isIn(pivotal, functions, numClauseFunctions, outlierTestValues, lmu::Graph() /*currently not used for overlap calculation*/, params))
	{
		resultClauses.push_back(pivotal);
		return;
	}

	std::vector<lmu::ImplicitFunctionPtr> candidateFunctions;

	for (int i = 0; i < numClauseFunctions; ++i)
	{
		if (i == pivotalLiteral)
			continue; 

		candidateFunctions.push_back(functions[i]);
		candidateFunctions.push_back(functions[i]);
	}

	for (size_t k = 1; k <= candidateFunctions.size(); ++k)
	{
		do
		{


		

		} while (lmu::next_combination(candidateFunctions.begin(), candidateFunctions.begin() + k, candidateFunctions.end()));

		//Remove all functions 
	}
	
	
	/*for (int i = currentLiteral; i < numClauseFunctions; ++i)
	{
		if (i == pivotalLiteral)
			continue;

		auto c = pivotal;

		//TODO: Check if non negated pivotal was already considered => lookup 

		c.literals[i] = true;

		computeClauseForPivotalFunction(c, pivotalLiteral, i+1, functions, numClauseFunctions, outlierTestValues, params, resultClauses);

		c.negated[i] = true; 

		computeClauseForPivotalFunction(c, pivotalLiteral, i+1, functions, numClauseFunctions, outlierTestValues, params, resultClauses);

		//TODO: Think about lookup for avoiding double cover of regions e.g. ABC | AB instead of AB
	}*/
}

std::vector<lmu::Clause> computeClauseForPivotalFunction(int pivotalLiteral, const std::vector<lmu::ImplicitFunctionPtr>& functions, int numClauseFunctions,
	const std::unordered_map<lmu::ImplicitFunctionPtr, std::tuple<double, double>>& outlierTestValues, const lmu::SampleParams& params)
{
	lmu::Clause pivotal(numClauseFunctions);
	pivotal.literals[pivotalLiteral] = true;
	
	std::vector<lmu::Clause> resultClauses;

	computeClauseForPivotalFunction(pivotal, pivotalLiteral, 0, functions, numClauseFunctions, outlierTestValues, params, resultClauses);

	return resultClauses;
}

std::vector<std::tuple<lmu::Clique, int>> getAllCliquesContainingFunction(const lmu::ImplicitFunctionPtr& func, const std::vector<lmu::Clique>& allCliques)
{
	std::vector<std::tuple<lmu::Clique, int>> cliques;

	for (const auto& clique : allCliques)
	{
		auto it = std::find(clique.functions.begin(), clique.functions.end(), func);
		if(it != clique.functions.end())
			cliques.push_back(std::make_tuple(clique, it - clique.functions.begin()));
	}

	return cliques; 
}

std::unordered_set<lmu::ImplicitFunctionPtr> getAllIntersectingFunctions(const std::vector<std::tuple<lmu::Clique, int>>& cliques)
{
	std::unordered_set<lmu::ImplicitFunctionPtr> res; 

	for (const auto& clique : cliques)	
		res.insert(std::get<0>(clique).functions.begin(), std::get<0>(clique).functions.end());
		
	return res;
}

//Horrible 
std::vector<lmu::ImplicitFunctionPtr> getAllCliqueFunctionsWithAppendedRest(const std::vector<lmu::ImplicitFunctionPtr>& cliqueFunctions, const std::unordered_set<lmu::ImplicitFunctionPtr>& intersectingFunctions)
{
	std::vector<lmu::ImplicitFunctionPtr> res; 

	res.insert(res.end(), cliqueFunctions.begin(), cliqueFunctions.end());

	for (const auto& intersectingFunction : intersectingFunctions)
	{
		if (std::find(cliqueFunctions.begin(), cliqueFunctions.end(), intersectingFunction) == cliqueFunctions.end())
			res.push_back(intersectingFunction);
	}

	return res;
}

lmu::CSGNode lmu::computeCSGNode(const std::vector<lmu::ImplicitFunctionPtr>& functions, const lmu::Graph& conGraph, const lmu::SampleParams& params)
{
	lmu::CSGNode node = lmu::opUnion();
	
	auto allCliques = getCliques(conGraph);

	auto outlierTestValues = computeOutlierTestValues(functions, params.h);

	for (const auto& f : functions)
	{
		std::cout << "Compute for " << f->name() << std::endl;

		auto cliquesForFunc = getAllCliquesContainingFunction(f, allCliques);

		auto intersectingFunctions = getAllIntersectingFunctions(cliquesForFunc);

		for (const auto& cliqueForFunc : cliquesForFunc)
		{		
			auto cliqueFunctions = getAllCliqueFunctionsWithAppendedRest(std::get<0>(cliqueForFunc).functions, intersectingFunctions);

			int numClauseFunctions = std::get<0>(cliqueForFunc).functions.size();

			auto clauses = computeClauseForPivotalFunction(std::get<1>(cliqueForFunc), cliqueFunctions, numClauseFunctions, outlierTestValues, params);

			for (const auto& clause : clauses)
			{
				node.addChild(lmu::clauseToCSGNode(clause, cliqueFunctions));
			}
		}
	}

	return node;
}
