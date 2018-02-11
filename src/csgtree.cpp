#include "..\include\csgtree.h"
#include "..\include\mesh.h"
#include "..\include\evolution.h"

#include "boost/graph/graphviz.hpp"
#include <boost/graph/kruskal_min_spanning_tree.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <array>
#include <cmath>
#include <algorithm>

#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/copyleft/cgal/CSGTree.h>

void createCSGTreeTemplateFromCliquesRec(lmu::CSGTree& tree, const std::vector<lmu::Clique>& cliques, size_t start, size_t size)
{	
	if (size > 1)
	{
		lmu::CSGTree leftTree;
		lmu::CSGTree rightTree;

		createCSGTreeTemplateFromCliquesRec(leftTree, cliques, start, size / 2);
		createCSGTreeTemplateFromCliquesRec(rightTree, cliques, start + size / 2, size / 2 + size % 2);

		tree.operation = lmu::OperationType::Unknown;//lmu::OperationType::Union; 
		tree.childs.push_back(leftTree);
		tree.childs.push_back(rightTree);
	}
	else if (size > 0)
	{	
		tree.operation = lmu::OperationType::Unknown;
		tree.functions = cliques[start].functions;
	}
}

int lmu::numOpTypes()
{
	return 5;
}

std::string lmu::opTypeToString(OperationType type)
{
	switch (type)
	{
	case OperationType::Intersection:
		return "Intersection";
	case OperationType::DifferenceLR:
		return "Difference LR";
	case OperationType::DifferenceRL:
		return "Difference RL";
	case OperationType::Union:
		return "Union";
	case OperationType::Unknown:
		return "Unknown";
	//case OperationType::Complement: 
	//	return "Complement";
	default:
		return "Undefined Type";
	}
}

lmu::CSGTree lmu::createCSGTreeTemplateFromCliques(std::vector<lmu::Clique>& cliques)
{
	//Fill implicit function => clique lookup, fill clique graph.
	CliqueGraph cliqueGraph;
	//std::unordered_map<const ImplicitFunction*, std::vector<size_t>> cliqueLookup;

	boost::property_map<CliqueGraph, CliqueVertex_t>::type cliqueMap = boost::get(CliqueVertex_t(), cliqueGraph);
	
	for (auto& clique : cliques)
	{
		auto v = boost::add_vertex(cliqueGraph);
		cliqueMap[v].clique = &clique;

		//for (const auto& f : clique.functions)
		//	cliqueLookup[f.get()].push_back(v);
	}
	
	//Add edges to clique graph.
	boost::graph_traits<Graph>::vertex_iterator cliqueIt1, cliqueIt2, cliqueItEnd;
	boost::property_map<CliqueGraph, boost::edge_weight_t>::type weightMap = boost::get(boost::edge_weight, cliqueGraph);	
	boost::property_map<CliqueGraph, CliqueEdge_t>::type cliqueEdgeMap = boost::get(CliqueEdge_t(), cliqueGraph);

	for (boost::tie(cliqueIt1, cliqueItEnd) = boost::vertices(cliqueGraph); cliqueIt1 != cliqueItEnd; ++cliqueIt1)
	{
		for (boost::tie(cliqueIt2, cliqueItEnd) = boost::vertices(cliqueGraph); cliqueIt2 != cliqueItEnd; ++cliqueIt2)
		{
			lmu::Clique* clique1 = cliqueMap[*cliqueIt1].clique;
			lmu::Clique* clique2 = cliqueMap[*cliqueIt2].clique;

			bool alreadyChecked = boost::edge(*cliqueIt1, *cliqueIt2, cliqueGraph).second;
			if (clique1 == clique2 || alreadyChecked)
				continue;

			std::unordered_set<std::shared_ptr<ImplicitFunction>> funcs1(clique1->functions.begin(), clique1->functions.end());
			std::unordered_set<std::shared_ptr<ImplicitFunction>> funcs2(clique2->functions.begin(), clique2->functions.end());

			CliqueEdge edge;

			for (auto func : clique1->functions)			
				if (funcs2.find(func) != funcs2.end())				
					edge.sharedIfs.insert(func);
				else
					edge.separatedIfs[clique2].insert(func);
			
			for (auto func : clique2->functions)
				if (funcs1.find(func) != funcs1.end())
					edge.sharedIfs.insert(func);
				else
					edge.separatedIfs[clique1].insert(func);

			//Do both cliques share a function? if so, non-shared must be subtracted from those that do.  
			if (!edge.sharedIfs.empty())
			{
				auto e = boost::add_edge(*cliqueIt1, *cliqueIt2, cliqueGraph);
				weightMap[e.first] = 1.0;
				cliqueEdgeMap[e.first] = edge;


				std::cout << "SHARED: " << (*edge.sharedIfs.begin())->name() << " SIZE: " << edge.sharedIfs.size()  << std::endl;
				std::cout << "SEPARATE" << std::endl;
				for (auto t : edge.separatedIfs)
				{
					std::cout << "  Clique ";
					for (auto c : t.first->functions)
						std::cout << c->name() << " ";
					std::cout << std::endl;

					for (auto sf : t.second)
						std::cout << "    Separated: " << sf->name();
					std::cout << std::endl;
				}
				//for (auto func : edge.sharedIfs)
				//{
				//	clique1->functionReplacements[func] = edge.separatedIfs[clique2];
				//	clique2->functionReplacements[func] = edge.separatedIfs[clique1];
				//}
			}
		}
	}
	
	std::ofstream f("tree_tmp.dot");
	boost::write_graphviz(f, cliqueGraph);
	f.close();

	//std::vector<CliqueEdgeDesc> minCliqueTreeEdges;
	//boost::kruskal_minimum_spanning_tree(cliqueGraph, std::back_inserter(minCliqueTreeEdges));
	
	auto resTree = createCSGTreeFromCliqueGraph(cliqueGraph /*, cliqueMap, minCliqueTreeEdges*/);

	resTree.fillUnknownOperations(cliques);

	resTree.resolveIFReplacements();

	return resTree; 
}

lmu::CSGTree lmu::createCSGTreeFromCliqueGraph(lmu::CliqueGraph& cliqueGraph)
{
	boost::graph_traits<Graph>::vertex_iterator cliqueIt, cliqueItEnd;
	boost::property_map<CliqueGraph, CliqueEdge_t>::type cliqueEdgeMap = boost::get(CliqueEdge_t(), cliqueGraph);
	boost::property_map<CliqueGraph, CliqueVertex_t>::type cliqueMap = boost::get(CliqueVertex_t(), cliqueGraph);

	lmu::CSGTree rootTree;
	rootTree.operation = lmu::OperationType::Union;
	lmu::CSGTree* curCSGTree = &rootTree;

	for (boost::tie(cliqueIt, cliqueItEnd) = boost::vertices(cliqueGraph); cliqueIt != cliqueItEnd; ++cliqueIt)
	{
		lmu::Clique* clique = cliqueMap[*cliqueIt].clique;

		lmu::CSGTree cliqueCSGTree;
		cliqueCSGTree.functions = clique->functions;
		cliqueCSGTree.operation = lmu::OperationType::Unknown;

		boost::graph_traits<CliqueGraph>::in_edge_iterator edgeIt, edgeItEnd;
		for (boost::tie(edgeIt, edgeItEnd) = boost::in_edges(*cliqueIt, cliqueGraph); edgeIt != edgeItEnd; ++edgeIt)
		{	
			CliqueEdge edge = cliqueEdgeMap[*edgeIt];
			cliqueCSGTree.iFReplacements.sharedIfs.insert(edge.sharedIfs.begin(), edge.sharedIfs.end());
			cliqueCSGTree.iFReplacements.separatedIfs.insert(edge.separatedIfs[clique].begin(), edge.separatedIfs[clique].end());
		}

		if (curCSGTree->childs.size() >= 1 && cliqueIt+1 != cliqueItEnd)
		{	
			lmu::CSGTree newTree;
			newTree.operation = lmu::OperationType::Union;

			curCSGTree->childs.push_back(newTree);
			curCSGTree = &curCSGTree->childs.back();
		}
		
		curCSGTree->childs.push_back(cliqueCSGTree);
	}

	return rootTree;
}

lmu::CSGTree lmu::createCSGTreeFromCliqueGraph(const lmu::CliqueGraph& cliqueGraph, const boost::property_map<lmu::CliqueGraph, lmu::CliqueVertex_t>::type& cliqueMap, 
	std::vector<lmu::CliqueEdgeDesc> minCliqueTreeEdges)
{
	boost::graph_traits<lmu::CliqueGraph>::in_edge_iterator ei, edge_end;

	lmu::CSGTree rootTree;
	rootTree.operation = lmu::OperationType::Union;

	//Add tree of first vertex.
	//auto curSourceVertex = boost::source(minCliqueTreeEdges[0], cliqueGraph);
	//lmu::CSGTree cliqueCSGTree;
	//cliqueCSGTree.functions = cliqueMap[curSourceVertex].clique->functions;
	//cliqueCSGTree.operation = lmu::OperationType::Unknown;
	//rootTree.childs.push_back(cliqueCSGTree);

	std::unordered_set<size_t> visitedVertices;

	lmu::CSGTree* curCSGTree = &rootTree;

	for (const auto& edge : minCliqueTreeEdges)
	{
		std::vector<size_t> vertices = { boost::source(edge, cliqueGraph), boost::target(edge, cliqueGraph) };

		for (auto vertex : vertices)
		{	
			std::cout << " V: " << vertex;

			if (visitedVertices.find(vertex) != visitedVertices.end())
				continue;

			if (boost::degree(vertex, cliqueGraph) > 2)
			{
				lmu::CSGTree childTree;
				childTree.operation = lmu::OperationType::Union;

				curCSGTree->childs.push_back(childTree);
				curCSGTree = &curCSGTree->childs.back();

				
				//curSourceVertex = boost::source(edge, cliqueGraph);
			}

			lmu::CSGTree cliqueCSGTree;
			cliqueCSGTree.functions = cliqueMap[vertex].clique->functions;
			cliqueCSGTree.operation = lmu::OperationType::Unknown;
			curCSGTree->childs.push_back(cliqueCSGTree);

			visitedVertices.insert(vertex);
		}
		
	}

	return rootTree;
}

double lambdaFromPoints(const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& shapes)
{
	int numPoints = 0;
	for (const auto& shape : shapes)
		numPoints += shape->points().rows();

	return std::log(numPoints);
}

lmu::CSGTree lmu::createCSGTreeWithGA(const std::vector<std::shared_ptr<ImplicitFunction>>& shapes, const lmu::Graph& connectionGraph)
{
	int i;
	std::cout << "Num shapes: " << shapes.size() << std::endl;
	std::cout << "Connection graph: " << !connectionGraph.m_vertices.empty() << std::endl;

	lmu::CSGTreeGA ga;
	lmu::CSGTreeGA::Parameters p(150, 2, 0.3, 0.3);

	lmu::CSGTreeTournamentSelector s(2, true);

	//lmu::CSGTreeIterationStopCriterion isc(10); 
	lmu::CSGTreeNoFitnessIncreaseStopCriterion isc(10, 0.01,100);


	lmu::CSGTreeCreator c(shapes, 0.5, 0.7, 7, connectionGraph);
	
	double lambda = lambdaFromPoints(shapes);
	std::cout << "lambda: " << lambda << std::endl;

	lmu::CSGTreeRanker r(lambda, shapes, connectionGraph);

	auto task = ga.runAsync(p, s, c, r, isc);
		
	std::cin >> i; 

	ga.stop();

	auto res = task.get();

	res.statistics.save("stats.dat"); 
	return res.population[0].creature;
}

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, lmu::CSGTree> TreeGraph;

void createGraphRec(const lmu::CSGTree& tree, TreeGraph& graph, size_t parentVertex)
{
	auto v = boost::add_vertex(graph);
	graph[v] = tree;
	if(parentVertex < std::numeric_limits<size_t>::max())
		boost::add_edge(parentVertex, v, graph);

	for (const auto& child : tree.childs)
	{
		createGraphRec(child, graph, v);
	}
}

template <class Name>
class VertexWriter {
public:
	VertexWriter(Name _name) : name(_name) {}
	template <class VertexOrEdge>
	void operator()(std::ostream& out, const VertexOrEdge& v) const
	{
		std::stringstream ss; 
		ss << "Operation: " << opTypeToString(name[v].operation) << std::endl;
		for (const auto& func : name[v].functions)
			ss << func->name() << " ";

		out << "[label=\"" << ss.str() << "\"]";
	}
private:
	Name name;
};

lmu::CSGTree::CSGTree(const std::vector<CSGTree>& c) : 
	childs(c)
{
}

lmu::CSGTree::CSGTree()
{
}

void lmu::CSGTree::write(const std::string & file)
{
	TreeGraph graph;
	createGraphRec(*this, graph, std::numeric_limits<size_t>::max());

	std::ofstream f(file);
	boost::write_graphviz(f, graph, VertexWriter<TreeGraph>(graph));
	f.close();
}

lmu::Mesh createIglCSGTreeRec(const lmu::CSGTree& tree)
{
	igl::MeshBooleanType iglOp;
	std::string opStr; 

	switch (tree.operation)
	{
	case lmu::OperationType::Union:
		iglOp = igl::MESH_BOOLEAN_TYPE_UNION;
		opStr = "union";
		break;

	case lmu::OperationType::Intersection:
		iglOp = igl::MESH_BOOLEAN_TYPE_INTERSECT;
		opStr = "intersect";
		break;
	case lmu::OperationType::DifferenceLR:
	case lmu::OperationType::DifferenceRL:
		iglOp = igl::MESH_BOOLEAN_TYPE_MINUS;
		opStr = "difference";
		break;
	default:
		throw std::runtime_error("Invalid CSG operation");
	}

	//iglOp = igl::MESH_BOOLEAN_TYPE_UNION;
	std::cout << "operation: " << opStr << std::endl;

	std::vector<lmu::Mesh> childs; 
	for (const auto& child : tree.childs)
	{
		std::cout << "go to child recursively." << std::endl;
		childs.push_back(createIglCSGTreeRec(child));
	}

	for (const auto& func : tree.functions)
	{
		std::cout << "function " << func->name() << std::endl;
		childs.push_back(func->mesh());
	}

	if (childs.size() != 2)
		throw std::runtime_error("Child size is not 2.");
	
	lmu::Mesh res;
	igl::copyleft::cgal::CSGTree::VectorJ vJ;

	if (tree.operation != lmu::OperationType::DifferenceRL)
	{		
		igl::copyleft::cgal::mesh_boolean(childs[0].vertices, childs[0].indices, childs[1].vertices, childs[1].indices, iglOp, res.vertices, res.indices, vJ);		
	}
	else
	{
		igl::copyleft::cgal::mesh_boolean(childs[1].vertices, childs[1].indices, childs[0].vertices, childs[0].indices, iglOp, res.vertices, res.indices, vJ);
	}
	
	return res;
}

lmu::Mesh lmu::CSGTree::createMesh() const
{
	return createIglCSGTreeRec(*this);
}

Eigen::Vector4d lmu::CSGTree::signedDistanceAndGradient(const Eigen::Vector3d & point) const
{
	Eigen::Vector4d res(0,0,0,0);

	std::vector<Eigen::Vector4d> sdsGrads;
	for (const auto& child : childs)
		sdsGrads.push_back(child.signedDistanceAndGradient(point));
	for (const auto& function : functions)
		sdsGrads.push_back(function->signedDistanceAndGradient(point));

	switch (operation)
	{
	case OperationType::Union:
		res[0] = std::numeric_limits<double>::max();
		for (const auto& sdGrad : sdsGrads)
		{
			//std::cout << " d union: " << sdGrad[0] << std::endl;

			res = sdGrad[0] < res[0] ? sdGrad : res;
		}
		break;
	case OperationType::Intersection:
		res[0] = -std::numeric_limits<double>::max();
		for (const auto& sdGrad : sdsGrads)
		{
			//std::cout << " d inter: " << sdGrad[0] << std::endl;

			res = sdGrad[0] > res[0] ? sdGrad : res;
		}
		break;	
	case OperationType::DifferenceLR:

		if (sdsGrads.size() == 2)
		{
			auto sdGrad1 = sdsGrads[0];
			auto sdGrad2 = (-1.0)*sdsGrads[1];

			if (sdGrad2[0] > sdGrad1[0])
				res = sdGrad2;
			else
				res = sdGrad1;

			//Negate gradient
			res[1] = (-1.0)*res[1];
			res[2] = (-1.0)*res[2];
			res[3] = (-1.0)*res[3];
		}
		else
		{
			std::cout << "Warning: Not exactly two operands for difference operation." << std::endl;
			res[0] = std::numeric_limits<double>::max();
		}

		break;

	case OperationType::DifferenceRL:

		if (sdsGrads.size() == 2)
		{
			auto sdGrad1 = sdsGrads[1];
			auto sdGrad2 = (-1.0)*sdsGrads[0];

			if (sdGrad2[0] > sdGrad1[0])
				res = sdGrad2;
			else
				res = sdGrad1;

			//Negate gradient
			res[1] = (-1.0)*res[1];
			res[2] = (-1.0)*res[2];
			res[3] = (-1.0)*res[3];
		}
		else
		{
			std::cout << "Warning: Not exactly two operands for difference operation." << std::endl;
			res[0] = std::numeric_limits<double>::max();
		}

		break;
	/*case OperationType::Complement:
		
		if (sdsGrads.size() == 1)
		{
			res = -sdsGrads[0];

			//TODO: CHECK IF THIS IS CORRECT
		}
		else
		{
			std::cout << "Warning: Not exactly one operand for complement operation." << std::endl;
			res[0] = std::numeric_limits<double>::max();
		}

		break;
		*/
	default:
		std::cout << "Warning: Undefined operation." << std::endl;
		res[0] = std::numeric_limits<double>::max();
		break;
	
	}

	return res;
}

void getIFsRec(const lmu::CSGTree& tree, std::vector<std::shared_ptr<lmu::ImplicitFunction>>& funcs)
{
	funcs.insert(funcs.end(), tree.functions.begin(), tree.functions.end());

	for (const auto& child : tree.childs)		
		getIFsRec(child, funcs);
}


double lmu::CSGTree::computeGeometryScore(double epsilon, double alpha, const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& funcs) const
{
	double score = 0.0;
	for (const auto& func : funcs)
	{
		for (int i = 0; i < func->points().rows(); ++i)
		{
			auto row = func->points().row(i);

			Eigen::Vector3d p = row.head<3>();
			Eigen::Vector3d n = row.tail<3>();
			
			Eigen::Vector4d distAndGrad = signedDistanceAndGradient(p);

			double d = distAndGrad[0] / epsilon;

			Eigen::Vector3d grad = distAndGrad.tail<3>();
			double minusGradientDotN = lmu::clamp(-grad.dot(n), -1.0, 1.0); //clamp is necessary, acos is only defined in [-1,1].
			double theta = std::acos(minusGradientDotN) / alpha;
			
			double scoreDelta = (std::exp(-(d*d)) + std::exp(-(theta*theta)));
			
			//if (scoreDelta < 0)
			//	std::cout << "Theta: " << theta << " minusGradientDotN: " << minusGradientDotN << std::endl;

			score += scoreDelta;
		}			
	}

	//std::cout << "ScoreGeo: " << score << std::endl;

	return /*1.0 / score*/ score;
}

void findAndSetBestOperation(lmu::CSGTree& tree)
{	
	for (auto& child : tree.childs)
		findAndSetBestOperation(child);
	
	auto functions = tree.functionsRecursively();

	lmu::CSGTreeRanker ranker(lambdaFromPoints(functions), functions);

	std::array<lmu::OperationType, 4> ops = 
	{ lmu::OperationType::Intersection , lmu::OperationType::Union, 
		lmu::OperationType::DifferenceLR,  lmu::OperationType::DifferenceRL };

	double maxScore = std::numeric_limits<double>::min();
	lmu::OperationType bestOp;
	for (auto const& op : ops)
	{
		tree.operation = op; 
		double score = ranker.rank(tree);

		if (score > maxScore)
		{
			maxScore = score;
			bestOp = op;
		}

		std::cout << "Score: " << score << " OP: " << opTypeToString(op) << std::endl;
	}	

	tree.operation = bestOp;	
}

void fillUnknownOperationsRec(lmu::CSGTree& tree, const std::vector<lmu::Clique>& cliques)
{
	static lmu::CSGTreeGA ga; 

	static lmu::CSGTreeGA::Parameters p(150,2, 0.3, 0.3);
	static lmu::CSGTreeTournamentSelector s(2);	
	
	//static lmu::CSGTreeIterationStopCriterion isc(100);
	static lmu::CSGTreeNoFitnessIncreaseStopCriterion isc(10, 0.01,100);

	if (tree.operation == lmu::OperationType::Unknown)
	{
		//assert(tree.childs.empty());
		//assert(tree.functions.size() > 1);
		
		if(tree.functions.size() == 2 || tree.childs.size() == 2)
			findAndSetBestOperation(tree);
		else
		{
			lmu::CSGTreeCreator c(tree.functions);

			double lambda = std::log(tree.numPoints());
			lmu::CSGTreeRanker r(lambda, tree.functions);

			//Run genetic algorithm.
			tree = ga.run(p, s, c, r, isc).population[0].creature;
		}
	}
	else
	{
		for (auto& child : tree.childs)
			fillUnknownOperationsRec(child, cliques);
	}
}

void lmu::CSGTree::fillUnknownOperations(const std::vector<lmu::Clique>& cliques)
{
	fillUnknownOperationsRec(*this, cliques);
}

void resolveIFReplacementsRec(lmu::CSGTree& tree, std::unordered_map<std::shared_ptr<lmu::ImplicitFunction>, lmu::CSGTree>& minusTreeLookup, const lmu::CliqueIFReplacements& iFReplacements)
{	
	std::cout << "resolveIFReplacementsRec" << std::endl;

	for (auto& child : tree.childs)
		resolveIFReplacementsRec(child, minusTreeLookup, iFReplacements);

	std::vector<std::shared_ptr<lmu::ImplicitFunction>> remainingFunctions;

	//Replacement only necessary if operation is union.
	if (tree.operation != lmu::OperationType::Union)
		return; 
	
	for (auto func : tree.functions)
	{
		if (iFReplacements.sharedIfs.find(func) == iFReplacements.sharedIfs.end())
		{
			remainingFunctions.push_back(func);
		}
		else
		{
			tree.childs.push_back(minusTreeLookup[func]);
		}
	}
	tree.functions = remainingFunctions;
}

void lmu::CSGTree::resolveIFReplacements()
{
	std::cout << "resolveIFReplacements" << std::endl;

	if (iFReplacements.sharedIfs.empty())
	{
		for (auto& child : childs)
		{
			child.resolveIFReplacements();
		}
		return;
	}

	std::unordered_map<std::shared_ptr<ImplicitFunction>, CSGTree> minusTreeLookup;

	//Fill tree lookup.
	for (auto func : functions)
	{
		if (iFReplacements.sharedIfs.find(func) != iFReplacements.sharedIfs.end())
		{
			CSGTree minusTree;
			minusTree.operation = OperationType::DifferenceLR;
			CSGTree* curTree = &minusTree; 

			//Create tree for shared function			
			int i = 0; 
			for (auto sepIF : iFReplacements.separatedIfs)
			{
			
				if (i == iFReplacements.separatedIfs.size() - 1)
				{
					curTree->functions = { func, sepIF };
				}
				else
				{
					curTree->functions = { sepIF };

					CSGTree childTree;
					childTree.operation = OperationType::DifferenceLR;
					curTree->childs.push_back(childTree);
					curTree = &curTree->childs.back();
				}
				i++;
			}
			
			minusTreeLookup[func] = minusTree;

			std::cout << "Func: " << func->name() << std::endl;
		}
	}

	resolveIFReplacementsRec(*this, minusTreeLookup, iFReplacements);
}

int lmu::CSGTree::depth(int curDepth) const
{
	int maxDepth = curDepth;
	
	for (const auto& child : childs)
	{
		int childDepth = child.depth(curDepth + 1); 
		maxDepth = std::max(maxDepth, childDepth);
	}
	
	return maxDepth;
}

int lmu::CSGTree::numNodes() const
{
	int num = 1; 
	for (const auto& child : childs)
	{
		num += child.numNodes();
	}

	return num;
}

int lmu::CSGTree::numPoints() const
{
	int n = 0; 
	for (const auto& f : functions)
		n += f->points().rows();
	
	for (const auto& c : childs)
		n += c.numPoints();

	return n;
}

lmu::CSGTree* nodeRec(lmu::CSGTree& tree, int idx, int& curIdx)
{
	if (idx == curIdx)
		return &tree; 
	
	curIdx++;

	for (auto& child : tree.childs)
	{
		auto foundTree = nodeRec(child, idx, curIdx);
		if (foundTree)
			return foundTree;
	}

	return nullptr;
}

lmu::CSGTree* lmu::CSGTree::node(int idx)
{
	int curIdx = 0; 
	return nodeRec(*this, idx, curIdx);
}

int nodeDepthRec(const lmu::CSGTree& tree, int idx, int& curIdx, int depth)
{
	if (idx == curIdx)
		return depth;

	curIdx++;

	for (auto& child : tree.childs)
	{
		auto foundDepth = nodeDepthRec( child, idx, curIdx, depth + 1);
		if (foundDepth != -1)
			return foundDepth;
	}

	return -1;
}

int lmu::CSGTree::nodeDepth(int idx) const
{
	int curIdx = 0;
	return nodeDepthRec(*this, idx, curIdx,0);
}

int lmu::CSGTree::sizeWithFunctions() const
{
	int size = 1 + functions.size(); 

	for (auto& child : childs)	
		size += child.sizeWithFunctions();
	
	return size;
}

std::vector<std::shared_ptr<lmu::ImplicitFunction>> lmu::CSGTree::functionsRecursively() const
{
	std::vector<std::shared_ptr<lmu::ImplicitFunction>> res;
	for (const auto& child : childs)
	{
		auto childFuncs = child.functionsRecursively();
		res.insert(res.end(), childFuncs.begin(), childFuncs.end()); 
	}
	
	res.insert(res.end(), functions.begin(), functions.end());

	return res;
}


