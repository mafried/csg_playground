#include "evolution.h"

#define _USE_MATH_DEFINES
#include <math.h>

lmu::CSGTreeRanker::CSGTreeRanker(double lambda, const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& functions) :
	_lambda(lambda),
	_functions(functions)
{
}

double lmu::CSGTreeRanker::rank(const lmu::CSGTree& tree) const
{	
	const double alpha = M_PI / 18.0;
	const double epsilon = 0.01;

	double geometryScore = tree.computeGeometryScore(epsilon, alpha, _functions);

	double score = geometryScore -_lambda * (tree.sizeWithFunctions());
	
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

lmu::CSGTreeCreator::CSGTreeCreator(const std::vector<std::shared_ptr<ImplicitFunction>>& functions, double createNewRandomProb, double subtreeProb, int maxTreeDepth):
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

int lmu::CSGTreeCreator::getRndFuncIndex(const std::vector<int>& usedFuncIndices) const
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	int funcIdx;
	bool again;
	do
	{
		again = false;
		funcIdx = du(_rndEngine, parmu_t{ 0, static_cast<int>(_functions.size() - 1) });
		for (int usedFuncIdx : usedFuncIndices)
		{
			if (funcIdx == usedFuncIdx)
			{
				again = true;
				break;
			}
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
			int funcIdx = getRndFuncIndex(usedFuncIndices);
			
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
