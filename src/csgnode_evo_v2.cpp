#include <numeric>
#include "../include/csgnode_evo_v2.h"
#include "../include/csgnode_helper.h"

// =========================================================================================
// Free Functions 
// =========================================================================================

std::ostream& lmu::operator<<(std::ostream& os, const IFBudget& b)
{
	for (const auto& item : b)
		os << item.first->name() << ": " << item.second << std::endl;

	return os;
}

lmu::IFBudget getBudget(const lmu::Graph& g)
{
	lmu::IFBudget budget;
	auto prunedGraph = g;//lmu::pruneGraph(g);
	auto funcs = getImplicitFunctions(g);

	//Initialize budgets with 0.
	for (const auto& func : funcs)
		budget[func] = 0;

	//heuristic for function budget based on cliques.
	auto cliques = lmu::getCliques(prunedGraph);
	for (const auto& clique : cliques)
	{
		for (const auto& func : clique.functions)
		{
			int budgetForFunc = 0;
			switch (clique.functions.size())
			{
			case 1:
				budgetForFunc = 1;
				break;
			case 2:
				budgetForFunc = 2;
				break;
			case 3:
				budgetForFunc = 3;
				break;
			case 4:
				budgetForFunc = 4;
				break;
			case 5:
				budgetForFunc = 5;
				break;
			default:
				std::cerr << "Cannot estimate budget for Function. Not implemented for clique size " << clique.functions.size() << "." << std::endl;
			}

			budget[func] += budgetForFunc;
		}
	}

	//add pruned functions with a budget of 1.
	for (const auto& func : funcs)
	{
		if (budget[func] == 0)
			budget[func] = 1;
	}

	return budget;
}

int numFuncsInBudget(const lmu::IFBudget& budget)
{
	return std::accumulate(std::begin(budget), std::end(budget), 0, [](int value, const lmu::IFBudget::value_type& p) { return value + p.second; });
}

void getUsedBudget(lmu::IFBudget& budget, const lmu::CSGNode& node)
{
	if (node.type() == lmu::CSGNodeType::Geometry)
	{
		auto it = budget.find(node.function());

		if (it == budget.end())
			budget[node.function()] = 1;
		else
			it->second++;
	}
	else
	{
		for (const auto& child : node.childsCRef())
			getUsedBudget(budget, child);
	}
}

lmu::IFBudget getUsedBudget(const lmu::CSGNode& node)
{
	lmu::IFBudget usedBudget;

	getUsedBudget(usedBudget, node);

	return usedBudget;
}

lmu::ImplicitFunctionPtr useIF(lmu::IFBudget& budget, const lmu::ImplicitFunctionPtr& func)
{
	auto it = budget.find(func);

	if (it == budget.end())
		return nullptr;

	if (it->second <= 0)
		return nullptr;

	it->second = it->second - 1;

	return func;
}

lmu::ImplicitFunctionPtr getRandomIF(lmu::IFBudget& budget, std::default_random_engine& rndEngine)
{
	std::vector<double> probs(budget.size());
	std::vector<lmu::ImplicitFunctionPtr> funcs(budget.size());

	transform(budget.begin(), budget.end(), probs.begin(), [](auto pair) {return pair.second; });
	transform(budget.begin(), budget.end(), funcs.begin(), [](auto pair) {return pair.first; });

	std::discrete_distribution<> d(probs.begin(), probs.end());

	int funcIdx = d(rndEngine);

	return useIF(budget, funcs[funcIdx]);
}

// =========================================================================================
// Types 
// =========================================================================================

lmu::CSGNodeRankerV2::CSGNodeRankerV2(const lmu::Graph& g, double sizeWeight, double h) : 
	_connectionGraph(g),
	_sizeWeight(sizeWeight),
	_h(h),
	_ifBudget(getBudget(g))
{
}

double lmu::CSGNodeRankerV2::rank(const CSGNode& node) const
{
	if (!node.isValid())
	{		
		return 0.0;
	}

	auto funcs = lmu::getImplicitFunctions(_connectionGraph);
	int numCorrectSamples = 0;
	int numConsideredSamples = 0;
	const double smallestDelta = 0.0001;
	
	for (const auto& func : funcs)
	{
		for (int i = 0; i < func->pointsCRef().rows(); ++i)
		{
			Eigen::Matrix<double, 1, 6> pn = func->pointsCRef().row(i);

			Eigen::Vector3d sampleP = pn.leftCols(3);
			Eigen::Vector3d sampleN = pn.rightCols(3);

			Eigen::Vector4d sampleDistGradNode = node.signedDistanceAndGradient(sampleP, _h);
			double sampleDistNode = sampleDistGradNode[0];
			Eigen::Vector3d sampleGradNode = sampleDistGradNode.bottomRows(3);

			numConsideredSamples++;

			if (std::abs(sampleDistNode) <= smallestDelta && sampleGradNode.dot(sampleN) > 0.0)
			{
				numCorrectSamples++;
			}		
			else
			{
				//std::cout << sampleDistNode << std::endl;
			}
		}
	}

	double geometryScore = (double)numCorrectSamples / (double)numConsideredSamples; 

	int numUsed = numFuncsInBudget(getUsedBudget(node));
	int numAvailable = numFuncsInBudget(_ifBudget);

	double sizeScore = (double)numUsed / (double)numAvailable;

	return geometryScore - _sizeWeight * sizeScore;
}

std::string lmu::CSGNodeRankerV2::info() const
{
	return std::string();
}

lmu::CSGNodeCreatorV2::CSGNodeCreatorV2(double createNewRandomProb, double subtreeProb, const lmu::Graph& graph) : 
	_createNewRandomProb(createNewRandomProb),
	_subtreeProb(subtreeProb),
	_ifBudget(getBudget(graph))
{
}

lmu::CSGNode lmu::CSGNodeCreatorV2::mutate(const CSGNode& node) const
{
	if (!node.isValid())
		return node;

	static std::bernoulli_distribution d{};
	using parm_t = decltype(d)::param_type;

	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	CSGNode mutatedNode = CSGNode::invalidNode;

	//_createNewRandomProb (my_0) 
	if (d(_rndEngine, parm_t{ _createNewRandomProb }))
	{
		mutatedNode = create();
	}
	else
	{
		int nodeIdx = du(_rndEngine, parmu_t{ 0, numNodes(node) - 1 });

		auto mutatedNode = node;

		CSGNode* subNode = nodePtrAt(mutatedNode, nodeIdx);

		*subNode = create(getUsedBudget(*subNode));
	}

	return mutatedNode.isValid() ? mutatedNode : node;
}

std::vector<lmu::CSGNode> lmu::CSGNodeCreatorV2::crossover(const CSGNode& node1, const CSGNode& node2) const
{
	if (!node1.isValid() || !node2.isValid())
		return std::vector<lmu::CSGNode> {node1, node2};

	int numNodes1 = numNodes(node1);
	int numNodes2 = numNodes(node2);

	auto newNode1 = node1;
	auto newNode2 = node2;

	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	int nodeIdx1 = du(_rndEngine, parmu_t{ 0, numNodes1 - 1 });
	int nodeIdx2 = du(_rndEngine, parmu_t{ 0, numNodes2 - 1 });

	CSGNode* subNode1 = nodePtrAt(newNode1, nodeIdx1);
	CSGNode* subNode2 = nodePtrAt(newNode2, nodeIdx2);

	std::swap(*subNode1, *subNode2);

	return std::vector<lmu::CSGNode>
	{
		newNode1, newNode2
	};
}

lmu::CSGNode lmu::CSGNodeCreatorV2::create() const
{
	auto budget = _ifBudget;

	return create(budget);
}

lmu::CSGNode lmu::CSGNodeCreatorV2::create(IFBudget& budget) const
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	static std::bernoulli_distribution db{};
	using parmb_t = decltype(db)::param_type;

	// Check for cases that don't need an operation.
	// Note: It would be possible to choose a complement operation with a budget of only 1 func left. 
	//       But since we currently do not support complement in this method, we do not consider that case.
	//       In addition, union and intersection could also deal with only 1 operand, but its pointless.
	switch (numFuncsInBudget(budget))
	{
	case 0:
		return CSGNode::invalidNode;
	case 1:
		return geometry(budget.begin()->first);
	}
	
	// Create operation node.
	// 0 is OperationType::Unknown, 4 is OperationType::Complement, 5 is OperationType::Invalid.
	auto node = createOperation(static_cast<CSGNodeOperationType>(du(_rndEngine, parmu_t{1, 3})));

	//std::cout << "Node of type: " << operationTypeToString(node.operationType()) << std::endl;

	// Create operands
	auto numAllowedChilds = node.numAllowedChilds();
	int minChilds = 2;// std::get<0>(numAllowedChilds);
	int maxChilds = std::get<1>(numAllowedChilds);
	int numChilds = clamp(2, minChilds, maxChilds); //2 is the maximum number of childs allowed.
	for (int i = 0; i < numChilds; i++)
	{
		if (db(_rndEngine, parmb_t{ _subtreeProb }))
		{
			auto child = create(budget);
			if (child.isValid()) //valid operand?
			{
				node.addChild(child);
				continue;
			}
		}
		
		auto func = getRandomIF(budget, _rndEngine);
		if (func)
		{		
			node.addChild(geometry(func));			
		}
		else //budget exceeded
		{				
			if (i > minChilds - 1) //do we have already enough operands?
			{		
				break;
			}
			else // return invalid node.
			{
				return CSGNode::invalidNode;
			}
		}	
	}

	return node;
}

std::string lmu::CSGNodeCreatorV2::info() const
{
	return std::string();
}

lmu::CSGNode lmu::createCSGNodeWithGAV2(const lmu::Graph& connectionGraph, bool inParallel, const std::string& statsFile)
{
	lmu::CSGNodeGAV2 ga;
	lmu::CSGNodeGAV2::Parameters p(150, 2, 0.7, 0.3, inParallel);
	lmu::CSGNodeTournamentSelector s(2, true);
	lmu::CSGNodeNoFitnessIncreaseStopCriterion isc(100, 0.001, 100);
	lmu::CSGNodeRankerV2 r(connectionGraph, 0.3, 0.01);
	lmu::CSGNodeCreatorV2 c(0.3, 0.7, connectionGraph);

	auto res = ga.run(p, s, c, r, isc);

	res.statistics.save(statsFile, &res.population[0].creature);
	return res.population[0].creature;
}


