#include <numeric>
#include "../include/csgnode_evo_v2.h"
#include "../include/csgnode_helper.h"
#include "../include/dnf.h"

// =========================================================================================
// Types 
// =========================================================================================

lmu::CSGNodeRankerV2::CSGNodeRankerV2(const lmu::Graph& g, double sizeWeight, double h) : 
	_connectionGraph(g),
	_sizeWeight(sizeWeight),
	_h(h),
	_ifBudget(IFBudget(g))
{
}

double lmu::CSGNodeRankerV2::rank(const CSGNode& node) const
{
	if (!node.isValid())
		return 0.0;
	
	double geometryScore = computeGeometryScore(node, lmu::getImplicitFunctions(_connectionGraph));

	int numRest = IFBudget(node, _ifBudget).numFuncs();
	int numAvailable = _ifBudget.numFuncs();

	double sizeScore = (double)(numAvailable-numRest) / (double)numAvailable;

	return geometryScore - _sizeWeight * sizeScore;
}

std::string lmu::CSGNodeRankerV2::info() const
{
	return "Size weight: " + std::to_string(_sizeWeight);
}

double lmu::CSGNodeRankerV2::computeGeometryScore(const CSGNode & node, const std::vector<ImplicitFunctionPtr>& funcs) const
{
	if (!node.isValid())
		return 0.0;

	//auto funcs = lmu::getImplicitFunctions(_connectionGraph);
	double numCorrectSamples = 0;
	double numConsideredSamples = 0;
	const double smallestDelta = 0.0001;

	double totalNumSamples = 0;
	for (const auto& func : funcs)
		totalNumSamples += func->pointsCRef().rows();

	for (const auto& func : funcs)
	{
		double sampleFactor = 1.0;//totalNumSamples / func->pointsCRef().rows();

		for (int i = 0; i < func->pointsCRef().rows(); ++i)
		{
			Eigen::Matrix<double, 1, 6> pn = func->pointsCRef().row(i);

			Eigen::Vector3d sampleP = pn.leftCols(3);
			Eigen::Vector3d sampleN = pn.rightCols(3);

			Eigen::Vector4d sampleDistGradNode = node.signedDistanceAndGradient(sampleP, _h);
			double sampleDistNode = sampleDistGradNode[0];
			Eigen::Vector3d sampleGradNode = sampleDistGradNode.bottomRows(3);

			numConsideredSamples += (1.0 * sampleFactor);

			if (std::abs(sampleDistNode) <= smallestDelta && sampleGradNode.dot(sampleN) > 0.0)
			{
				numCorrectSamples += (1.0 * sampleFactor);
			}
			else
			{
				//std::cout << sampleDistNode << std::endl;
			}
		}
	}

	return numCorrectSamples / numConsideredSamples;
}

lmu::CSGNodeCreatorV2::CSGNodeCreatorV2(double createNewRandomProb, double subtreeProb, const lmu::Graph& graph) : 
	_createNewRandomProb(createNewRandomProb),
	_subtreeProb(subtreeProb),
	_ifBudget(IFBudget(graph)),
	_rndEngine(lmu::rndEngine())
{
	std::cout << "CREATOR BUDGET: " << _ifBudget << std::endl;
}

lmu::CSGNode createWithShapiro(lmu::IFBudget& budget, std::default_random_engine& rndEngine)
{
	static std::bernoulli_distribution d{};
	using parm_t = decltype(d)::param_type;

	//Collect primitives 
	lmu::ImplicitFunctionPtr func = nullptr;
	std::vector<lmu::ImplicitFunctionPtr> funcs; 
	do 
	{
		func = budget.getRandomIF();
		if (!func)
			break; 

		funcs.push_back(func);
	} while (d(rndEngine, parm_t{ 0.5 }));

	auto g = createConnectionGraph(funcs);
	auto cc = getConnectedComponents(g);
	size_t ccSize = 0;
	const lmu::Graph* cg = nullptr;
	for (const auto& c : cc)
	{
		if (numVertices(c) > ccSize)
		{
			cg = &c; 
			ccSize = numVertices(c);
		}
	}

	//std::cout << "GRAPH " << (cg == nullptr) << " " << (funcs.size()) << " " << cc.size() << std::endl;

	funcs = lmu::getImplicitFunctions(*cg);

	if (funcs.size() == 1)
		return lmu::geometry(funcs[0]);

	auto dnf = lmu::computeShapiro(funcs, true, *cg, { 0.001 });

	return lmu::DNFtoCSGNode(dnf);
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
				
		*subNode = createOperation(CSGNodeOperationType::Identity);
				
		if (d(_rndEngine, parm_t{ 1.0 }))
		{
		  IFBudget ifbud(mutatedNode, _ifBudget);
		  //*subNode = create(IFBudget(mutatedNode, _ifBudget));
		  *subNode = create(ifbud);
		}
		else 		
		{
		  IFBudget ifbud(mutatedNode, _ifBudget);
		  //replaceIFs(ifbud, *subNode);
		  *subNode = createWithShapiro(ifbud, _rndEngine);
		}
	}

	return mutatedNode;
}

/*std::vector<lmu::CSGNode> lmu::CSGNodeCreatorV2::crossover(const CSGNode& node1, const CSGNode& node2) const
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
}*/

std::vector<lmu::CSGNode> lmu::CSGNodeCreatorV2::crossover(const CSGNode& node1, const CSGNode& node2) const
{
	if (!node1.isValid() || !node2.isValid())
		return std::vector<lmu::CSGNode> {node1, node2};

	lmu::CSGNodeRankerV2 r(_connectionGraph, 0.1, 0.01);
	
	auto newNode1 = node1;
	auto newNode2 = node2;

	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;
	int nodeIdx1 = du(_rndEngine, parmu_t{ 0, numNodes(node1) - 1 });
	
	CSGNode* subNode1 = nodePtrAt(newNode1, nodeIdx1);
	auto subNode1Funcs = lmu::allDistinctFunctions(*subNode1);
	
	CSGNode* subNode2 = findSmallestSubgraphWithImplicitFunctions(newNode2, subNode1Funcs);
	
	if(!subNode2)
		return std::vector<lmu::CSGNode> {node1, node2};

	auto subNode2Funcs = lmu::allDistinctFunctions(*subNode2);

	auto funcs = subNode1Funcs.size() > subNode2Funcs.size() ? subNode1Funcs : subNode2Funcs;

	double score1 = r.computeGeometryScore(*subNode1, funcs);
	double score2 = r.computeGeometryScore(*subNode2, funcs);

	std::cout << "CROSSOVER" << std::endl;
	std::cout << serializeNode(*subNode1) << "     " << serializeNode(*subNode2) << std::endl;
	std::cout << score1 << "     " << score2 << std::endl;
	if (score1 > score2)
		*subNode2 = *subNode1;
	else if (score1 < score2)
		*subNode1 = *subNode2;
	
	return std::vector<lmu::CSGNode>{ newNode1, newNode2};
}

lmu::CSGNode lmu::CSGNodeCreatorV2::create() const
{
	auto budget = _ifBudget;

	return create(budget);
}

void lmu::CSGNodeCreatorV2::replaceIFs(IFBudget& budget, CSGNode& node) const
{
	if (node.type() == CSGNodeType::Geometry)
	{
		auto f = budget.exchangeIF(node.function());
		node.setFunction(f);
	}
	else
	{
		for (auto& child : node.childsRef())
			replaceIFs(budget, child);
	}
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
	//switch (budget.numFuncs())
	//{
	//case 0:
	//	return CSGNode::invalidNode;
	//case 1:
	//	return geometry(budget.useFirstIF());
	//}
	
	// Create operation node.
	// 0 is OperationType::Unknown, 4 is OperationType::Complement, 5 is OperationType::Invalid.
	auto node = createOperation(static_cast<CSGNodeOperationType>(du(_rndEngine, parmu_t{1, 3})));
		
	// Create operands
	auto numAllowedChilds = node.numAllowedChilds();
	int minChilds = 2;// std::get<0>(numAllowedChilds);
	int maxChilds = std::get<1>(numAllowedChilds);
	int numChilds = clamp(2, minChilds, maxChilds); //2 is the maximum number of childs allowed.

	if (minChilds > budget.numFuncs())
	{
		return geometry(budget.getRandomIF(false));
	}
	
	for (int i = 0; i < numChilds; i++)
	{
		//In case budget is exhausted and enough childs are already available, stop child creation.
		if (budget.numFuncs() <= 0 && i > minChilds - 1) 
			break;		

		if (db(_rndEngine, parmb_t{ _subtreeProb }))
		{
			auto child = create(budget);			
			node.addChild(child);
						
		}
		else
		{
			auto func = budget.getRandomIF(false);
			node.addChild(geometry(func));
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
	lmu::CSGNodeTournamentSelector s(2, true);
	lmu::CSGNodeNoFitnessIncreaseStopCriterion isc(300, 0.001, 300);
	lmu::CSGNodeCreatorV2 c(0.5, 0.7, connectionGraph);

	// New Ranker
	lmu::CSGNodeGAV2 ga;
	lmu::CSGNodeGAV2::Parameters p(150, 2, 0.7, 0.7, true);
	lmu::CSGNodeRankerV2 r(connectionGraph, 0.2, 0.01);
	
	auto res = ga.run(p, s, c, r, isc);

	res.statistics.save(statsFile, &res.population[0].creature);
	return res.population[0].creature;
}

void lmu::IFBudgetPerIF::getRestBudget(const lmu::CSGNode& node, IFBudgetPerIF& budget)
{
	if (node.type() == lmu::CSGNodeType::Geometry)
	{
		auto it = budget._budget.find(node.function());

		if (it != budget._budget.end())
			it->second--;
	}
	else
	{
		for (const auto& child : node.childsCRef())
			getRestBudget(child, budget);
	}
}

lmu::IFBudgetPerIF::IFBudgetPerIF(const lmu::CSGNode& node, const IFBudgetPerIF& budget) : 
	_rndEngine(lmu::rndEngine())
{
	auto b = budget;
	getRestBudget(node, b);
	_budget = b._budget;
	_totalBudget = std::accumulate(std::begin(_budget), std::end(_budget), 0,
			[](int value, const std::unordered_map<lmu::ImplicitFunctionPtr, int>::value_type& p) { return value + p.second; });
}

lmu::IFBudgetPerIF::IFBudgetPerIF(const lmu::Graph& g) : 
	_rndEngine(lmu::rndEngine())
{
	auto prunedGraph = g;//lmu::pruneGraph(g);
	auto funcs = getImplicitFunctions(g);

	//Initialize budgets with 0.
	for (const auto& func : funcs)
		_budget[func] = 0;

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

			_budget[func] += budgetForFunc;
		}
	}

	//add pruned functions with a budget of 1.
	for (const auto& func : funcs)
	{
		if (_budget[func] == 0)
			_budget[func] = 1;
	}

	_totalBudget = std::accumulate(std::begin(_budget), std::end(_budget), 0,
		[](int value, const std::unordered_map<lmu::ImplicitFunctionPtr, int>::value_type& p) { return value + p.second; });
}

int lmu::IFBudgetPerIF::numFuncs() const
{
	return _totalBudget;
}

lmu::ImplicitFunctionPtr lmu::IFBudgetPerIF::useIF(const lmu::ImplicitFunctionPtr & func)
{
	auto it = _budget.find(func);

	if (it == _budget.end())
		return nullptr;

	//if (it->second <= 0)
	//	return nullptr;

	//it->second = it->second - 1;

	_totalBudget = _totalBudget > 0 ? _totalBudget - 1 : 0;

	return func;
}

lmu::ImplicitFunctionPtr lmu::IFBudgetPerIF::getRandomIF(bool uniform)
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	std::vector<lmu::ImplicitFunctionPtr> funcs(_budget.size());
	std::transform(_budget.begin(), _budget.end(), funcs.begin(), [](auto pair) {return pair.first; });

	int funcIdx = 0;
	if (uniform)
	{
		funcIdx = du(_rndEngine, parmu_t{ 0, (int)funcs.size() - 1 });
	}
	else
	{
		std::vector<double> probs(_budget.size());
		std::transform(_budget.begin(), _budget.end(), probs.begin(), [](auto pair) {return pair.second; });
		std::discrete_distribution<> d(probs.begin(), probs.end());
		funcIdx = d(_rndEngine);
	}

	return useIF(funcs[funcIdx]);
}

lmu::ImplicitFunctionPtr lmu::IFBudgetPerIF::useFirstIF()
{
	return useIF(_budget.begin()->first);
}

lmu::ImplicitFunctionPtr lmu::IFBudgetPerIF::exchangeIF(const lmu::ImplicitFunctionPtr& func)
{
	_totalBudget++;
	return getRandomIF();
}

void lmu::IFBudgetPerIF::freeIF(const lmu::ImplicitFunctionPtr & func)
{
	auto it = _budget.find(func);

	if (it != _budget.end())
		_totalBudget++;
}


std::ostream& lmu::operator<<(std::ostream& os, const IFBudgetPerIF& b)
{
	for (const auto& item : b._budget)
		os << item.first->name() << ": " << item.second << std::endl;

	return os;
}

lmu::CSGNode lmu::computeGAWithPartitionsV2(const std::vector<Graph>& partitions,
	bool inParallel, const std::string& statsFile)
{
	lmu::CSGNode res = lmu::op<Union>();

	//for (const auto& pi: get<1>(partition)) {
	//  res.addChild(lmu::geometry(pi));
	//}

	for (const auto& p : partitions)
	{			
		lmu::CSGNode ga = lmu::createCSGNodeWithGAV2(p, inParallel, statsFile);

		if (partitions.size() == 1)
			return ga;

		res.addChild(ga);
	}

	return res;
}
