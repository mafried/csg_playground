#include "../include/csgnode_evo.h"
#include "../include/csgnode_helper.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <boost/dynamic_bitset.hpp>
#include <boost/graph/adjacency_list.hpp>


#include "../include/constants.h"


using namespace lmu;

lmu::CSGNodeRanker::CSGNodeRanker(double lambda, const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& functions, const lmu::Graph& connectionGraph) :
	_lambda(lambda),
	_functions(functions),
	_earlyOutTest(!connectionGraph.structure.m_vertices.empty()),
	_connectionGraph(connectionGraph)
{
}

double lmu::CSGNodeRanker::rank(const lmu::CSGNode& node) const
{
	const double alpha = M_PI / 18.0;
	const double epsilon = 0.01;
		
	double geometryScore = computeGeometryScore(node, epsilon, alpha, _functions);

	double score = geometryScore - _lambda * numNodes(node);
		
	//std::cout << "lambda: " << _lambda << std::endl;
	//std::cout << "geometry score: " << geometryScore << std::endl;
		
		
	return score;
}

std::string lmu::CSGNodeRanker::info() const
{
	std::stringstream ss;
	ss << "CSGNode Ranker (lambda: " << _lambda << ", early out test: " << _earlyOutTest << ")";
	return ss.str();
}

boost::dynamic_bitset<> getFunctionConnectionBitfield(const std::shared_ptr<lmu::ImplicitFunction>& func, const lmu::Graph& connectionGraph, const std::unordered_map<std::shared_ptr<lmu::ImplicitFunction>, int>& funcToIdx, int bitfieldSize)
{
	boost::dynamic_bitset<> bf(bitfieldSize);

	//Go through all adjacent nodes of the node holding func and mark them in the bitfield as connected.
	boost::graph_traits<GraphStructure>::adjacency_iterator  neighbour, neighbourEnd;
	for (boost::tie(neighbour, neighbourEnd) = boost::adjacent_vertices(connectionGraph.vertexLookup.at(func), connectionGraph.structure); neighbour != neighbourEnd; ++neighbour)	
		bf.set(funcToIdx.at(connectionGraph.structure[*neighbour]), true);	

	return bf;
}

bool treeIsInvalidRec(const lmu::CSGNode& node, boost::dynamic_bitset<>& bf, const Graph& connectionGraph, const std::unordered_map<std::shared_ptr<lmu::ImplicitFunction>, int>& funcToIdx)
{
	auto numAllowedChilds = node.numAllowedChilds();
	if (node.childs().size() < std::get<0>(numAllowedChilds) || node.childs().size() > std::get<1>(numAllowedChilds))
		return true;

	if (node.type() == CSGNodeType::Operation)
	{	
		boost::dynamic_bitset<> lastBF(bf.size());
		static boost::dynamic_bitset<> emptyBF(bf.size());

		bool firstRun = true; 

		for (const auto& child : node.childsCRef())
		{
			boost::dynamic_bitset<> childBF(bf.size());

			if (treeIsInvalidRec(child, childBF, connectionGraph, funcToIdx))
				return true;
						
			if ((childBF & lastBF) == emptyBF && !firstRun)
				return true; 

			firstRun = false;
			lastBF = childBF;		

			bf |= lastBF;
		}		
	}
	else
	{
		bf = getFunctionConnectionBitfield(node.function(), connectionGraph, funcToIdx, bf.size());
	}

	return false;
}

bool lmu::CSGNodeRanker::treeIsInvalid(const lmu::CSGNode& node) const
{
	//Check if all functions were used in the tree.
	std::unordered_set<std::shared_ptr<lmu::ImplicitFunction>> usedFuncs; 
	for (auto const& func : allGeometryNodePtrs(node))
		usedFuncs.insert(func->function());

	if (usedFuncs.size() != _functions.size())
		return true;

	//Check if subtrees have overlapping functions sets.
	boost::dynamic_bitset<> bf(_functions.size());
	std::unordered_map<std::shared_ptr<lmu::ImplicitFunction>, int> funcToIdx;
	for (int i = 0; i < _functions.size(); ++i)
		funcToIdx[_functions[i]] = i;

	return treeIsInvalidRec(node, bf, _connectionGraph, funcToIdx);
}

CSGNodeCreator::CSGNodeCreator(const std::vector<std::shared_ptr<ImplicitFunction>>& functions, double createNewRandomProb, double subtreeProb, int maxTreeDepth, const Graph& connectionGraph) :
	_functions(functions),
	_createNewRandomProb(createNewRandomProb),
	_subtreeProb(subtreeProb),
	_maxTreeDepth(maxTreeDepth),
	_connectionGraph(connectionGraph)
{
	_rndEngine.seed(_rndDevice());
}


// For a given probability, create a new creature 
// Otherwise, select a node and create a new sub-tree at the node
CSGNode CSGNodeCreator::mutate(const CSGNode& node) const
{
	static std::bernoulli_distribution d{};
	using parm_t = decltype(d)::param_type;

	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	static std::uniform_real_distribution<double> dur(-0.1, 0.1);
	using parmur_t = decltype(dur)::param_type;
	
	//Node mutation probability 
	//auto rate = node.attribute<double>("mutationRate");
	//if (d(_rndEngine, parm_t{ 1.0 - rate }))
	//	return node;

	//_createNewRandomProb (my_0) 
	if (d(_rndEngine, parm_t{ _createNewRandomProb }))
		return create(_maxTreeDepth);

	int nodeIdx = du(_rndEngine, parmu_t{ 0, numNodes(node) - 1 });

	std::cout << "Mutation at " << nodeIdx << std::endl;

	auto newNode = node;

	//auto delta = dur(_rndEngine);
	//newNode.setAttribute("mutationRate", lmu::clamp(rate + delta, 0.0, 1.0));

	CSGNode* subNode = nodePtrAt(newNode, nodeIdx);

	//int nodeBudget = numNodes(*subNode);

	int maxSubtreeDepth = _maxTreeDepth - depthAt(newNode, nodeIdx);

	*subNode = create(maxSubtreeDepth);

	//std::cout << "    old node depth: " << depth(node) << " new tree depth: " << depth(newNode) << " subtree depth: " << depth(subNode) << " max subtree depth: " << maxSubtreeDepth << std::endl;

	return newNode;
}

std::vector<lmu::CSGNode> lmu::CSGNodeCreator::crossover(const lmu::CSGNode& node1, const lmu::CSGNode& node2) const
{
	int numNodes1 = numNodes(node1);
	int numNodes2 = numNodes(node2);

	auto newNode1 = node1;
	auto newNode2 = node2;

	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	int nodeIdx1 = du(_rndEngine, parmu_t{ 0, numNodes1 - 1 });
	int nodeIdx2 = du(_rndEngine, parmu_t{ 0, numNodes2 - 1 });

	std::cout << "Crossover at " << nodeIdx1 << " and " << nodeIdx2 << std::endl;

	CSGNode* subNode1 = nodePtrAt(newNode1, nodeIdx1);
	CSGNode* subNode2 = nodePtrAt(newNode2, nodeIdx2);
	
	//CSGNode tmp = subNode1->clone();
	//*subNode1 = *subNode2;
	//*subNode2 = *tmp;
	std::swap(*subNode1, *subNode2);

	//std::cout << "    1: old tree depth: " << tree1.depth() << " new tree depth: " << newTree1.depth() << " Max: " << _maxTreeDepth << std::endl;
	//std::cout << "    2: old tree depth: " << tree2.depth() << " new tree depth: " << newTree2.depth() << " Max: " << _maxTreeDepth << std::endl;

	return std::vector<lmu::CSGNode>
	{
		//newTree1.depth() <= _maxTreeDepth ? newTree1 : tree1,
		//newTree2.depth() <= _maxTreeDepth ? newTree2 : tree2

		depth(newNode1) <= _maxTreeDepth ? newNode1 : node1,
		depth(newNode2) <= _maxTreeDepth ? newNode2 : node2
	};
}

lmu::CSGNode lmu::CSGNodeCreator::create() const
{	
	return create(_maxTreeDepth);
}

lmu::CSGNode lmu::CSGNodeCreator::create(int maxDepth) const
{
	static std::bernoulli_distribution db{};
	using parmb_t = decltype(db)::param_type;

	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	static std::uniform_real_distribution<double> dur(0, 1);
	using parmur_t = decltype(dur)::param_type;

	if (maxDepth == 0)
	{
		int funcIdx = du(_rndEngine, parmu_t{ 0, static_cast<int>(_functions.size() - 1) });

		auto node = geometry(_functions[funcIdx]);
	
		return node;
	}
	else
	{
		int op = du(_rndEngine, parmu_t{ 1, 3 }); //0 is OperationType::Unknown, 6 is OperationType::Invalid.

		auto node = createOperation(static_cast<CSGNodeOperationType>(op));
	
		create(node, maxDepth, 1);

		return node;
	}
}

void lmu::CSGNodeCreator::create(lmu::CSGNode& node, int maxDepth, int curDepth) const
{
	static std::bernoulli_distribution db{};
	using parmb_t = decltype(db)::param_type;

	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	static std::uniform_real_distribution<double> dur(0, 1);
	using parmur_t = decltype(dur)::param_type;

	if (curDepth >= maxDepth)
	{
		if (node.type() == CSGNodeType::Operation)
		{
			int funcIdx = du(_rndEngine, parmu_t{ 0, static_cast<int>(_functions.size() - 1) });
			node = geometry(_functions[funcIdx]);
		}

		return;
	}

	auto numAllowedChilds = node.numAllowedChilds();
	int numChilds = clamp(std::get<1>(numAllowedChilds), std::get<0>(numAllowedChilds), 2); //2 is the maximum number of childs allowed for create

	for (int i = 0; i < numChilds; ++i)
	{
		lmu::CSGNode child(nullptr);

		if (db(_rndEngine, parmb_t{ _subtreeProb }))
		{
			int op = du(_rndEngine, parmu_t{ 1, 3 }); //0 is OperationType::Unknown, 5 is OperationType::Complement, 6 is OperationType::Invalid.

			child = createOperation(static_cast<CSGNodeOperationType>(op));

			create(child, maxDepth, curDepth + 1);
		}
		else
		{
			//Get random function index.
			int funcIdx = du(_rndEngine, parmu_t{ 0, static_cast<int>(_functions.size() - 1) }); 																								 

			child = geometry(_functions[funcIdx]);
		}

		node.addChild(child);
	}
}

bool functionAlreadyUsed(const std::vector<int>& usedFuncIndices, int funcIdx)
{
	return std::find(usedFuncIndices.begin(), usedFuncIndices.end(), funcIdx) != usedFuncIndices.end();
}

int lmu::CSGNodeCreator::getRndFuncIndex(const std::vector<int>& usedFuncIndices) const
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	int funcIdx;
	bool again;
	do
	{
		again = false;
		funcIdx = du(_rndEngine, parmu_t{ 0, static_cast<int>(_functions.size() - 1) });

		if (functionAlreadyUsed(usedFuncIndices, funcIdx))
		{
			again = true;
			break;
		}

	} while (again);

	return funcIdx;
}

std::string lmu::CSGNodeCreator::info() const
{
	std::stringstream ss;
	ss << "CSGTree Creator (create new random prob: " << _createNewRandomProb << ", sub tree prob: " << _subtreeProb << ", max tree depth: " << _maxTreeDepth << ")";
	return ss.str();
}

double lmu::lambdaBasedOnPoints(const std::vector<lmu::ImplicitFunctionPtr>& shapes)
{
	int numPoints = 0;
	for (const auto& shape : shapes)
		numPoints += shape->points().rows();

	return std::log(numPoints);
}

long long binom(int n, int k)
{
	long long ans = 1;
	k = k>n - k ? n - k : k;
	int j = 1;
	for (; j <= k; j++, n--)
	{
		if (n%j == 0)
		{
			ans *= n / j;
		}
		else
			if (ans%j == 0)
			{
				ans = ans / j*n;
			}
			else
			{
				ans = (ans*n) / j;
			}
	}
	return ans;
}

lmu::CSGNode lmu::createCSGNodeWithGA(const std::vector<std::shared_ptr<ImplicitFunction>>& shapes, bool inParallel, const lmu::Graph& connectionGraph, const std::string& statsFile)
{
	if (shapes.size() == 1)
		return lmu::geometry(shapes[0]);

	lmu::CSGNodeGA ga;
	lmu::CSGNodeGA::Parameters p(150, 2, 0.7, 0.3, inParallel, 
		[](const auto& p)
	{
		/*double rate = 0.0;
		for (const auto& i : p)
		{
			rate += i.creature.attribute<double>("mutationRate");
		}
		std::cout << "MUTATION: " << (rate / (double)p.size()) << std::endl;
		*/
	});

	lmu::CSGNodeTournamentSelector s(2, true);

	//lmu::CSGNodeIterationStopCriterion isc(100); 
	lmu::CSGNodeNoFitnessIncreaseStopCriterion isc(500, 0.01, 500);

	int maxDepth =(int)(/*2.0**/ sqrt((double)(boost::num_edges(connectionGraph.structure) > 0 ? boost::num_edges(connectionGraph.structure) : binom(shapes.size(),2)) * M_PI));
	std::cout << "Num Shapes: " << shapes.size() << " MaxDepth: " << maxDepth << std::endl;

	lmu::CSGNodeCreator c(shapes, 0.5, 0.7, maxDepth, connectionGraph);

	double lambda = lambdaBasedOnPoints(shapes);
	std::cout << "lambda: " << lambda << std::endl;

	lmu::CSGNodeRanker r(lambda, shapes, connectionGraph);

	//auto task = ga.runAsync(p, s, c, r, isc);

	//int i;
	//std::cin >> i;

	//ga.stop();

	auto res = ga.run(p, s, c, r, isc);// task.get();

	res.statistics.save(statsFile, &res.population[0].creature);
	return res.population[0].creature;
}


// Mimic computeShapiroWithPartitions in dnf.cpp
// Apply a GA to each group of intersecting shapes
lmu::CSGNode 
lmu::computeGAWithPartitions
(const std::vector<Graph>& partitions,
 bool inParallel, const std::string& statsFile)
{
  lmu::CSGNode res = lmu::op<Union>();
  
  //for (const auto& pi: get<1>(partition)) {
  //  res.addChild(lmu::geometry(pi));
  //}

  for (const auto& p: partitions) 
  {
    std::vector<std::shared_ptr<ImplicitFunction>> shapes = lmu::getImplicitFunctions(p);
    
    // This is a first try:
    // Possibly we can change the parameters of the GA (smallest population), 
    // also we do not need union operators at all

    lmu::CSGNode ga = lmu::createCSGNodeWithGA(shapes, inParallel, p, statsFile);
    
	if (partitions.size() == 1)
		return ga;

	res.addChild(ga);
  }

  return res;
}


std::tuple<long long, double> computeNodesForClique(const Clique& clique, bool gAInParallel, std::vector<GeometryCliqueWithCSGNode>& res)
{
	TimeTicker ticker; 
	double score = 0.0;

	if (clique.functions.empty())
	{
		return std::make_tuple(0, 0.0);
	}
	else if (clique.functions.size() == 1)
	{
		res.push_back(std::make_tuple(clique, CSGNode(std::make_shared<CSGNodeGeometry>(clique.functions[0]))));		
	}
	else if (clique.functions.size() == 2)
	{
		lmu::CSGNodeRanker ranker(lambdaBasedOnPoints(clique.functions), clique.functions);
		
		std::vector<CSGNode> candidates;

		CSGNode un(std::make_shared<UnionOperation>("un"));
		un.addChild(CSGNode(std::make_shared<CSGNodeGeometry>(clique.functions[0])));
		un.addChild(CSGNode(std::make_shared<CSGNodeGeometry>(clique.functions[1])));
		candidates.push_back(un);

		CSGNode inter(std::make_shared<IntersectionOperation>("inter"));
		inter.addChild(CSGNode(std::make_shared<CSGNodeGeometry>(clique.functions[0])));
		inter.addChild(CSGNode(std::make_shared<CSGNodeGeometry>(clique.functions[1])));
		candidates.push_back(inter);

		CSGNode lr(std::make_shared<DifferenceOperation>("lr"));
		lr.addChild(CSGNode(std::make_shared<CSGNodeGeometry>(clique.functions[0])));
		lr.addChild(CSGNode(std::make_shared<CSGNodeGeometry>(clique.functions[1])));
		candidates.push_back(lr);

		CSGNode rl(std::make_shared<DifferenceOperation>("rl"));
		rl.addChild(CSGNode(std::make_shared<CSGNodeGeometry>(clique.functions[1])));
		rl.addChild(CSGNode(std::make_shared<CSGNodeGeometry>(clique.functions[0])));
		candidates.push_back(rl);

		double maxScore = -std::numeric_limits<double>::max();
		const CSGNode* bestCandidate = nullptr;
		for (const auto& candidate : candidates)
		{
			double curScore = ranker.rank(candidate);
			std::cout << candidate.name() << " for " << clique.functions[0]->name() << " " << clique.functions[1]->name() << " rank: " << curScore /*<< " tree rank: " << ranker2.rank( trees[i++]) */ << std::endl;

			if (maxScore < curScore)
			{
				maxScore = curScore;
				bestCandidate = &candidate;
			}
		}

		score = maxScore;
		res.push_back(std::make_tuple(clique, *bestCandidate));
	}
	else
	{
		std::stringstream ss; 
		ss << clique << ".dat";
		score = 666.0;

		res.push_back(std::make_tuple(clique, createCSGNodeWithGA(clique.functions, gAInParallel, Graph(), ss.str())));
	}

	return std::make_tuple(ticker.tick(), score);
}

ParallelismOptions lmu::operator|(ParallelismOptions lhs, ParallelismOptions rhs)
{
	return static_cast<ParallelismOptions>(static_cast<int>(lhs) | static_cast<int>(rhs));
}
ParallelismOptions lmu::operator&(ParallelismOptions lhs, ParallelismOptions rhs)
{
	return static_cast<ParallelismOptions>(static_cast<int>(lhs) & static_cast<int>(rhs));
}

std::vector<GeometryCliqueWithCSGNode> lmu::computeNodesForCliques(const std::vector<Clique>& geometryCliques, ParallelismOptions po)
{
	std::vector<GeometryCliqueWithCSGNode> res;

	bool cliquesParallel = (po & ParallelismOptions::PerCliqueParallelism) == ParallelismOptions::PerCliqueParallelism;
	bool gAParallel = (po & ParallelismOptions::GAParallelism) == ParallelismOptions::GAParallelism;

	std::ofstream f("clique_info.dat");
	
	if (cliquesParallel)
	{
#ifndef _OPENMP 
		throw std::runtime_error("Cliques should run in parallel but OpenMP is not available.");
#endif

#pragma omp parallel
		{
#pragma omp master  
			{
				f << "OpenMP is running with " << omp_get_num_threads() << " threads." << std::endl;
			}
#pragma omp for
			for (int i = 0; i < geometryCliques.size(); ++i)
			{
				f << "Clique " << (i + 1) << " of " << geometryCliques.size() << " is started" << geometryCliques[i] << std::endl;
							
				auto stats = computeNodesForClique(geometryCliques[i], gAParallel, res);
				f << "Timing: " << std::get<0>(stats) << " Score: " << std::get<1>(stats) << std::endl;

				f << geometryCliques[i] << " done." << std::endl;
			}
		}
	}
	else
	{
		int i = 1;
		for (const auto& clique : geometryCliques)
		{
			f << "Clique " << (i++) << " of " << geometryCliques.size() << " is started: " << clique << std::endl;

			auto stats = computeNodesForClique(clique, gAParallel, res);
			f << "Timing: " << std::get<0>(stats) << " Score: " << std::get<1>(stats) << std::endl;

			f << clique << " done." << std::endl;
		}
	}

	f.close();

	return res;
}

size_t getHash(const CSGNode* n1, const CSGNode* n2)
{
	std::size_t seed = 0;
	boost::hash_combine(seed, reinterpret_cast<std::uintptr_t>(n1));
	boost::hash_combine(seed, reinterpret_cast<std::uintptr_t>(n2));
	
	return seed;
}

void lmu::optimizeCSGNodeClique(CSGNodeClique& clique, float tolerance)
{
	for (auto& item : clique)
	{
		optimizeCSGNodeStructure(std::get<1>(item));// , tolerance);
	}
}

CSGNode lmu::mergeCSGNodeCliqueSimple(CSGNodeClique& clique)
{
	if (clique.empty())
		throw std::runtime_error("Cannot merge empty clique.");

	if (clique.size() == 1)
		return std::get<1>(clique.front());

	//Fill candidate list. 
	std::list<CSGNode*> candidateList;	
	for (auto& item : clique)	
		candidateList.push_back(&std::get<1>(item));	

	bool allowIntersections = false;

	while (candidateList.front() != candidateList.back())
	{
		auto n1 = candidateList.front(); 
		candidateList.pop_front();

		auto n2 = candidateList.front();
		candidateList.pop_front();

		std::cout << "Took two new nodes" << std::endl;

		auto firstN2 = n2;
		while (true)
		{
			std::cout << "Find css" << std::endl;

			auto css = findCommonSubgraphs(*n1, *n2);

			std::cout << "Found css" << std::endl;

			CSGNode* mergedNode = nullptr;
			for (const auto& cs : css)
			{
				std::cout << "Write nodes" << std::endl;

				writeNode(*n1, "n1.dot");
				writeNode(*n2, "n2.dot");

				std::cout << "Wrote nodes" << std::endl;

				std::cout << "Merge nodes" << std::endl;

				switch (mergeNodes(cs, allowIntersections))
				{
				case MergeResult::First:
					std::cout << "Merged with n1" << std::endl;
					mergedNode = n1;
					break;
				case MergeResult::Second:
					std::cout << "Merged with n2" << std::endl;
					mergedNode = n2;
					break;
				case MergeResult::None:
					std::cout << "Not merged" << std::endl;			
					break;
				}
				
				if (mergedNode)
					break;
			}

			std::cout << "Merged nodes" << std::endl;

			if (mergedNode)
			{
				std::cout << "Merge node available" << std::endl;

				candidateList.push_front(mergedNode);
				allowIntersections = false;
				break;
			}
			else
			{
				std::cout << "Merge node not available" << std::endl;

				candidateList.push_back(n2);
				auto n2 = candidateList.front();
				candidateList.pop_front();

				if (n2 == firstN2)
				{
					if (allowIntersections)
					{
						std::cout << "could not merge n1 with any other node - n1 is ignored now." << std::endl;
						break;
					}
					else
					{
						std::cout << "Try to merge now with intersections allowed." << std::endl;
						allowIntersections = true;
					}					
				}
			}
		}
	}

	std::cout << "Candidate list: " << candidateList.size() << std::endl;

	return *candidateList.front();
}

double computeGeometryScore(const CSGNode& node, double distAngleDeviationRatio, double maxDistance, const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& funcs)
{
	//std::cout << "Compute Geometry Score" << std::endl;

	double score = 0.0;
	for (const auto& func : funcs)
	{
		for (int i = 0; i < func->points().rows(); ++i)
		{
			auto row = func->points().row(i);

			Eigen::Vector3d p = row.head<3>();
			Eigen::Vector3d n = row.tail<3>();

			Eigen::Vector4d distAndGrad = node.signedDistanceAndGradient(p);

			double distance = lmu::clamp(distAndGrad[0] / maxDistance, 0.0, 1.0); //distance in [0,1]

			Eigen::Vector3d grad = distAndGrad.tail<3>();
			double gradientDotN = lmu::clamp(/*-*/grad.dot(n), -1.0, 1.0); //clamp is necessary, acos is only defined in [-1,1].			

			double theta = std::acos(gradientDotN) / M_PI; //theta in [0,1]

			//double scoreDelta = (std::exp(-(d*d)) + std::exp(-(theta*theta)));

			//if (scoreDelta < 0)
			//	std::cout << "Theta: " << theta << " minusGradientDotN: " << minusGradientDotN << std::endl;
			

			score += (1.0 - distAngleDeviationRatio) * distance + distAngleDeviationRatio * theta;
		}
	}

	//std::cout << "ScoreGeo: " << score << std::endl;

	return /*1.0 / score*/ score;
}


/*double lmu::CSGNodeRankerNew::rank(const CSGNode& node) const
{
	double geo = computeGeometryScore(node, _distAngleDeviationRatio, _maxDistance, _functions);

	double normalizedGeo = lmu::clamp(geo / _maxGeo, 0.0, 1.0);

	double normalizedSize = lmu::numNodes(node) / _maxSize <= 1.0 ? 0.0 : 1.0; //TODO

	return lmu::clamp(normalizedGeo - normalizedSize, 0.0, 1.0);
}

std::string lmu::CSGNodeRankerNew::info() const
{
	return std::string();
}*/
