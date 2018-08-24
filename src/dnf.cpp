#include "dnf.h"
#include "csgnode_helper.h"
#include "curvature.h"

#include <algorithm>
#include <unordered_map>
#include <Eigen/Core>

#include <boost/math/special_functions/erf.hpp>


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
double scaledMAD(const lmu::ImplicitFunctionPtr& func)
{	
	lmu::CSGNode node = lmu::geometry(func);

	std::vector<double> values(func->pointsCRef().rows());

	for (int j = 0; j < func->pointsCRef().rows(); ++j)
	{

		Eigen::Matrix<double, 1, 6> pn = func->pointsCRef().row(j);

		Eigen::Vector3d p = pn.leftCols(3);
		Eigen::Vector3d n = pn.rightCols(3);

		double h = 0.01;

		lmu::Curvature c = curvature(p, node, h);

		values[j] = std::sqrt(c.k1 * c.k1 + c.k2 * c.k2);
	}

	double med = median(values);
	std::transform(values.begin(), values.end(), values.begin(), [med](double v) -> double { return std::abs(v - med); });

	const double c = -1.0 / (std::sqrt(2.0)*boost::math::erfc_inv(3.0 / 2.0));
	
	return c * median(values);	
}

std::unordered_map<lmu::ImplicitFunctionPtr, double> lmu::computeOutlierTestValues(const std::vector<lmu::ImplicitFunctionPtr>& functions)
{
	std::unordered_map<lmu::ImplicitFunctionPtr, double> map;

	for (const auto& func : functions)
		map[func] = scaledMAD(func) * 3.0;

	return map;
}


bool lmu::isIn(const Clause& clause, const std::vector<ImplicitFunctionPtr>& functions, const std::unordered_map<lmu::ImplicitFunctionPtr, double> outlierTestValues, const SampleParams& params)
{
	lmu::CSGNode node = clauseToCSGNode(clause, functions);

	int numCorrectSamples = 0;
	int numTotalSamples = 0;
	int numConsideredSamples = 0;

	double h = 0.0001;

	for (int i = 0; i < functions.size(); ++i)
	{	
		//if (!clause.literals[i])
		//	continue;

		lmu::ImplicitFunctionPtr func = functions[i];

		double outlierTestValue = outlierTestValues.at(func);

		for (int j = 0; j < func->pointsCRef().rows(); ++j)
		{
			numTotalSamples++;

			Eigen::Matrix<double, 1,6> pn = func->pointsCRef().row(j);

			Eigen::Vector3d sampleP = pn.leftCols(3);
			Eigen::Vector3d sampleN = pn.rightCols(3);

			Eigen::Vector4d sampleDistGradFunction = func->signedDistanceAndGradient(sampleP);
			double sampleDistFunction = sampleDistGradFunction[0];
			Eigen::Vector3d sampleGradFunction = sampleDistGradFunction.bottomRows(3);			
			
			//Move sample position back on the function's implied surface.			
			sampleP = sampleP - sampleGradFunction.cwiseProduct(Eigen::Vector3d(sampleDistFunction, sampleDistFunction, sampleDistFunction));

			Eigen::Vector4d sampleDistGradNode = node.signedDistanceAndGradient(sampleP, h);
			double sampleDistNode = sampleDistGradNode[0];
			Eigen::Vector3d sampleGradNode = sampleDistGradNode.bottomRows(3);
			
			//if(abs(func->signedDistanceAndGradient(sampleP)[0]) > h*h / 2.0)
			//	std::cout << sampleDistFunction << " " << func->signedDistanceAndGradient(sampleP)[0] << std::endl;


			if (sampleDistNode > h*h / 2.0)
			{
				continue;
			}

			numConsideredSamples++; 

			if (sampleDistNode < -(h*h) / 2.0) //TODO
			{
				//std::cout << "0: " << sampleDistNode << std::endl;
				continue;
			}
			
			if (sampleGradNode.dot(sampleN) <= 0.0)
			{
				//std::cout << "1: " << sampleDistNode << std::endl;
				continue;
			}


			//std::cout << sampleGradNode.dot(sampleGradFunction) << std::endl;

			numCorrectSamples++;
			//if (std::abs(fp[0] - distToFunc) > 0.0001)
			//	continue;

			//std::cout << fp[0] << std::endl;

			//if (std::abs(fp[0]) > params.maxDistDelta)
			//	continue;
			
			//double h = 0.01;
			//Curvature c = curvature(p, node, h);
			//double deviationFromFlatness = std::sqrt(c.k1 * c.k1 + c.k2 * c.k2);

			//if (deviationFromFlatness - outlierTestValue > 0.0001)
			//	continue;

			//Eigen::Vector3d g(fp[1], fp[2], fp[3]);
						
			//bool isInSameDirection = g.dot(n) > 0.0;

			//numSameDir += isInSameDirection;			
			//numOtherDir += !isInSameDirection;	
		}
	}

	//if (numSameDir + numOtherDir == 0)
	//	return false;

	//double consideredSamples = (double)(numSameDir + numOtherDir) / (double)totalNumSamples;
	double correctSamples = numConsideredSamples == 0.0 ? 0.0 : (double)numCorrectSamples / (double)(numConsideredSamples);

	std::cout << "Clause: ";
	print(std::cout, clause, functions, false);
	std::cout << std::endl;
	//std::cout << "Considered Samples: " << consideredSamples << std::endl;
	std::cout << "Correct Samples: " << correctSamples << std::endl;
	
	return correctSamples >= params.requiredCorrectSamples; //&&
		//consideredSamples >= params.requiredConsideredSamples;
}

bool lmu::isPrime(const ImplicitFunctionPtr& testFunc, const std::vector<ImplicitFunctionPtr>& functions, const std::unordered_map<lmu::ImplicitFunctionPtr, double> outlierTestValues, const SampleParams& params)
{
	std::cout << testFunc->name() << std::endl;

	int numConsideredSamples = 0;
	int numCorrectSamples = 0;

	for (int i = 0; i < functions.size(); ++i)
	{
		if (functions[i] == testFunc)
			continue;

		double outlierTestValue = outlierTestValues.at(testFunc);

		for (int j = 0; j < functions[i]->pointsCRef().rows(); ++j)
		{			
			Eigen::Matrix<double, 1, 6> pn = functions[i]->pointsCRef().row(j);

			Eigen::Vector3d p = pn.leftCols(3);
			Eigen::Vector3d n = pn.rightCols(3);

			Eigen::Vector4d fp = testFunc->signedDistanceAndGradient(p);

			if (fp[0] > 0.1)
				continue;

			numConsideredSamples++;

			if (fp[0] > 0.0)
				numCorrectSamples++;
		}
	}

	double correctSamples = (double)numCorrectSamples / (double)numConsideredSamples;
		
	std::cout << "Correct Samples: " << correctSamples << std::endl;

	return correctSamples >= params.requiredCorrectSamples;		
}

void permutate(lmu::Clause clause /*copy necessary*/, lmu::DNF& dnf, const std::unordered_map<lmu::ImplicitFunctionPtr, double> outlierTestValues, const lmu::SampleParams& params, int& iterationCounter)
{			
	std::sort(clause.negated.begin(), clause.negated.end());

	do {
							
		if (lmu::isIn(clause, dnf.functions, outlierTestValues, params))
			dnf.clauses.push_back(clause);				

		iterationCounter++;
		
		std::cout << "Ready: " << (double)iterationCounter / std::pow(2, clause.negated.size()) * 100.0 << "%" << std::endl;

	} while (std::next_permutation(clause.negated.begin(), clause.negated.end()));
}

std::tuple<lmu::DNF, std::vector<lmu::ImplicitFunctionPtr>> identifyPrimeImplicants(const std::vector<lmu::ImplicitFunctionPtr>& functions, const std::unordered_map<lmu::ImplicitFunctionPtr, double> outlierTestValues, const lmu::SampleParams& params)
{
	std::vector<lmu::ImplicitFunctionPtr> restFuncs;
	std::vector<lmu::ImplicitFunctionPtr> dnfFuncs;

	//Check which primitive is completely inside the geometry.
	for (int i = 0; i < functions.size(); ++i)
	{
		//Defining a clause representing a single primitive.
		lmu::Clause clause(functions.size());
		clause.literals[i] = true;
		print(std::cout, clause, functions, false);
		 
		if (/*lmu::isPrime(functions[i], functions, outlierTestValues, params)*/ lmu::isIn(clause, functions, outlierTestValues, params))
		{
			dnfFuncs.push_back(functions[i]);
			std::cout << "Prime implicant: " << functions[i]->name() << std::endl;
		}
		else
			restFuncs.push_back(functions[i]);
	}

	//Create a DNF that contains a clause for each primitive.
	lmu::DNF dnf; 	
	for (int i = 0; i < dnfFuncs.size(); ++i)
	{
		lmu::Clause clause(dnfFuncs.size());
		clause.literals[i] = true;
		dnf.clauses.push_back(clause);
	}
	dnf.functions = dnfFuncs;

	return std::make_tuple(dnf, restFuncs);
}

lmu::DNF lmu::computeShapiro(const std::vector<ImplicitFunctionPtr>& functions, bool usePrimeImplicantOptimization, const SampleParams& params)
{
	DNF primeImplicantsDNF;
	DNF dnf;

	auto outlierTestValues = computeOutlierTestValues(functions);
	
	if (usePrimeImplicantOptimization)
	{
		auto res = identifyPrimeImplicants(functions, outlierTestValues, params);
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

	for (int i = 0; i <= dnf.functions.size(); i++)
	{	
		permutate(clause, dnf, outlierTestValues, params, iterationCounter);
		
		if(i < dnf.functions.size())
			clause.negated[i] = true;
	}

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
