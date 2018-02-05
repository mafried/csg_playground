#include "evolution.h"

#define _USE_MATH_DEFINES
#include <math.h>

lmu::CSGTreeRanker::CSGTreeRanker(double lambda, const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& functions, const lmu::Graph& connectionGraph) :
	_lambda(lambda),
	_functions(functions),
	_earlyOutTest(!connectionGraph.m_vertices.empty()),
	_connectionGraph(connectionGraph)
{
}

double lmu::CSGTreeRanker::rank(const lmu::CSGTree& tree) const
{	
	const double alpha = M_PI / 18.0;
	const double epsilon = 0.01;

	double geometryScore = tree.computeGeometryScore(epsilon, alpha, _functions);

	double score = geometryScore -_lambda * (tree.sizeWithFunctions());

	std::cout << "EARLY: " << _earlyOutTest << std::endl;

	//if (_earlyOutTest && treeIsInvalid(tree))
	//	score = 0.0;
	
	//std::cout << "lambda: " << _lambda << std::endl;
	std::cout << "geometry score: " << geometryScore << std::endl;
	//std::cout << "size score: " << tree.sizeWithFunctions() << std::endl;
	//std::cout << "score: " << score << std::endl;

	//Important check. If not done, trees with a poor geometry score (lower than size penalty term)
	//Have a negative score which always outperforms more decent scores.
	if (score < 0)
		score = 0;
	
	return 1.0 / (1.0 + score);
}

std::string lmu::CSGTreeRanker::info() const
{
	std::stringstream ss;
	ss << "CSGTree Ranker (lambda: " << _lambda << ", early out test: " << _earlyOutTest << ")";
	return ss.str();
}

bool funcsConnect(const std::vector< std::shared_ptr<lmu::ImplicitFunction>>& funcs, const std::shared_ptr<lmu::ImplicitFunction>& func, const lmu::Graph& connectionGraph)
{
	for (auto& f : funcs)
	{
		auto v1 = connectionGraph.vertexLookup.at(f);
		auto v2 = connectionGraph.vertexLookup.at(func);

		if (boost::edge(v1, v2, connectionGraph).second)
			return true;

	}
	return false;
}

bool lmu::CSGTreeRanker::treeIsInvalid(const lmu::CSGTree & tree) const
{
	if (tree.childs.empty())
	{
		if (tree.functions.size() == 2)
			return tree.functions[0] == tree.functions[1];
		else
			return true;
	}
	else if (tree.childs.size() == 1 && tree.functions.size() == 1)
	{
		auto funcs = tree.childs[0].functionsRecursively();
		auto f = tree.functions[0];

		if (std::find(funcs.begin(), funcs.end(), f) != funcs.end() || funcsConnect(funcs, f, _connectionGraph))		
			return treeIsInvalid(tree.childs[0]);		
	}
	else if (tree.childs.size() == 2 && tree.functions.empty())
	{
		auto funcs = tree.childs[1].functionsRecursively();

		for (auto& f : tree.childs[0].functionsRecursively())
		{			
			if (std::find(funcs.begin(), funcs.begin(), f) != funcs.end() || funcsConnect(funcs, f, _connectionGraph))
				return treeIsInvalid(tree.childs[0]) || treeIsInvalid(tree.childs[1]);
		}
	}
	
	return true;
}

lmu::CSGTreeCreator::CSGTreeCreator(const std::vector<std::shared_ptr<ImplicitFunction>>& functions, double createNewRandomProb, double subtreeProb, int maxTreeDepth, const lmu::Graph& connectionGraph):
	_functions(functions),
	_createNewRandomProb(createNewRandomProb),
	_subtreeProb(subtreeProb),
	_maxTreeDepth(maxTreeDepth)
{
	_rndEngine.seed(_rndDevice());
}

lmu::CSGTree lmu::CSGTreeCreator::mutate(const lmu::CSGTree& tree) const 
{
	static std::bernoulli_distribution d{};
	using parm_t = decltype(d)::param_type;

	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	//_createNewRandomProb (my_0) 
	if (d(_rndEngine, parm_t{ _createNewRandomProb }))
		return create(_maxTreeDepth);

	int nodeIdx = du(_rndEngine, parmu_t{ 0, tree.numNodes() - 1 });

	std::cout << "Mutation at " << nodeIdx << std::endl;

	auto newTree = tree;
	lmu::CSGTree* subtree = newTree.node(nodeIdx);

	assert(subtree);

	*subtree = create(_maxTreeDepth - newTree.nodeDepth(nodeIdx));
	
	std::cout << "    old tree depth: " << tree.depth() << " new tree depth: " << newTree.depth() << " subtree depth: " << subtree->depth() << std::endl;

	return newTree;
}

std::vector<lmu::CSGTree> lmu::CSGTreeCreator::crossover(const lmu::CSGTree& tree1, const lmu::CSGTree& tree2) const
{
	int numNodes1 = tree1.numNodes();
	int numNodes2 = tree2.numNodes();

	auto newTree1 = tree1;
	auto newTree2 = tree2;

	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	int nodeIdx1 = du(_rndEngine, parmu_t{ 0, numNodes1 - 1 });
	int nodeIdx2 = du(_rndEngine, parmu_t{ 0, numNodes2 - 1 });

	std::cout << "Crossover at " << nodeIdx1 << " and " << nodeIdx2 << std::endl;

	lmu::CSGTree* subtree1 = newTree1.node(nodeIdx1);
	lmu::CSGTree* subtree2 = newTree2.node(nodeIdx2);

	assert(subtree1 && subtree2);

	std::swap(*subtree1, *subtree2);

	std::cout << "    1: old tree depth: " << tree1.depth() << " new tree depth: " << newTree1.depth() << " Max: " << _maxTreeDepth << std::endl;
	std::cout << "    2: old tree depth: " << tree2.depth() << " new tree depth: " << newTree2.depth() << " Max: " << _maxTreeDepth << std::endl;

	return std::vector<lmu::CSGTree> 
	{
		newTree1.depth() <= _maxTreeDepth ? newTree1 : tree1,
		newTree2.depth() <= _maxTreeDepth ? newTree2 : tree2
	};
}

lmu::CSGTree lmu::CSGTreeCreator::create() const 
{
	//std::cout << "Create called " << _maxTreeDepth << std::endl;
	auto c = create(_maxTreeDepth);
	//	std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DONE" << c.depth() << std::endl;
	return c;
}

bool funcAlreadyUsed(const std::vector<int>& usedFuncIndices, int funcIdx)
{
	return std::find(usedFuncIndices.begin(), usedFuncIndices.end(), funcIdx) != usedFuncIndices.end();
}

bool funcConnectsToSiblingFuncs(const lmu::CSGTree& tree, std::shared_ptr<lmu::ImplicitFunction> func, const lmu::Graph& connectionGraph)
{
	//Connection graph empty? return true;
	if (connectionGraph.m_vertices.empty())
		return true;

	//If there is no child and no function in the tree, there does not have to be a connection.
	if (tree.childs.empty())
	{
		if (tree.functions.empty())
			return true;
		else
			return funcsConnect(tree.functions, func, connectionGraph);
	}

	for (const auto& child : tree.childs)
	{
		auto funcsInChild = child.functionsRecursively();

		//If func exists in sibling, func connects in any case 
		if (std::find(funcsInChild.begin(), funcsInChild.end(), func) != funcsInChild.end())
			return true;

		//If a func in sibling is connected to func, func index is also valid.
		if (funcsConnect(funcsInChild, func, connectionGraph))
			return true;
	}

	return false;
}

int lmu::CSGTreeCreator::getRndFuncIndex(const std::vector<int>& usedFuncIndices, const lmu::CSGTree& tree) const 
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	int funcIdx;
	bool again;
	do
	{
		again = false;
		funcIdx = du(_rndEngine, parmu_t{ 0, static_cast<int>(_functions.size() - 1) });
		
		if (funcAlreadyUsed(usedFuncIndices, funcIdx) /* || !funcConnectsToSiblingFuncs(tree, _functions[funcIdx], _connectionGraph)*/)
		{
			again = true;
			break;
		}
		
	} while (again);

	return funcIdx;
}

void lmu::CSGTreeCreator::create(lmu::CSGTree& tree, int maxDepth, int curDepth) const 
{
	static std::bernoulli_distribution db{};
	using parmb_t = decltype(db)::param_type;

	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	int op = du(_rndEngine, parmu_t{ 1, numOpTypes() - 1 }); //0 is OperationType::Unknown
	tree.operation = static_cast<OperationType>(op);

	int numChilds = 2; //tree.operation == OperationType::Complement ? 1 : 2;

	std::vector<int> usedFuncIndices;
	usedFuncIndices.reserve(numChilds - 1);

	for (int i = 0; i < numChilds ; ++i)
	{
		if (db(_rndEngine, parmb_t{ _subtreeProb }) && curDepth < maxDepth)
		{		
			lmu::CSGTree child;
			create(child, maxDepth, curDepth+1);
			tree.childs.push_back(child);			
		}
		else
		{
			//Get random function index. Avoid multiple appearances of a function in one operation.
			int funcIdx = getRndFuncIndex(usedFuncIndices, tree);
			
			tree.functions.push_back(_functions[funcIdx]);	

			usedFuncIndices.push_back(funcIdx);
		}
	}
}

lmu::CSGTree lmu::CSGTreeCreator::create(int maxDepth) const 
{	
	lmu::CSGTree tree;
	create(tree, maxDepth, 0);
	return tree;
}

std::string lmu::CSGTreeCreator::info() const
{
	std::stringstream ss; 
	ss << "CSGTree Creator (create new random prob: " << _createNewRandomProb << ", sub tree prob: " << _subtreeProb << ", max tree depth: " << _maxTreeDepth << ")";
	return ss.str();
}
