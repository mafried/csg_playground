#include "..\include\csgnode.h"

#include <limits>
#include <fstream>
#include <random>

#include "boost/graph/graphviz.hpp"

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>

#include <igl/copyleft/marching_cubes.h>

using namespace lmu;

CSGNodePtr UnionOperation::clone() const
{
	return std::make_shared<UnionOperation>(*this);
}
Eigen::Vector4d UnionOperation::signedDistanceAndGradient(const Eigen::Vector3d& p) const
{
	Eigen::Vector4d res(0, 0, 0, 0);

	res[0] = std::numeric_limits<double>::max();
	for (const auto& child : _childs)
	{
		auto childRes = child.signedDistanceAndGradient(p);
		res = childRes[0] < res[0] ? childRes : res;
	}

	return res;
}
CSGNodeOperationType UnionOperation::operationType() const
{
	return CSGNodeOperationType::Union;
}
std::tuple<int, int> UnionOperation::numAllowedChilds() const
{
	return std::make_tuple(1, std::numeric_limits<int>::max());
}
Mesh lmu::UnionOperation::mesh() const
{
	/*if (_childs.size() == 0)
		return Mesh(); 
	if (_childs.size() == 1)
		return _childs[0].mesh();

	Mesh res, left, right;
	igl::copyleft::cgal::CSGTree::VectorJ vJ;

	left = _childs[0].mesh();

	for (int i = 1; i < _childs.size();++i)
	{
		right = _childs[i].mesh();

		igl::copyleft::cgal::mesh_boolean(left.vertices, left.indices, right.vertices, right.indices, igl::MESH_BOOLEAN_TYPE_UNION, res.vertices, res.indices, vJ);
		
		left = res;
	}

	return res;*/

	return Mesh();
}

CSGNodePtr IntersectionOperation::clone() const
{
	return std::make_shared<IntersectionOperation>(*this);
}
Eigen::Vector4d IntersectionOperation::signedDistanceAndGradient(const Eigen::Vector3d & p) const
{
	Eigen::Vector4d res(0, 0, 0, 0);

	res[0] = -std::numeric_limits<double>::max();
	for (const auto& child : _childs)
	{
		auto childRes = child.signedDistanceAndGradient(p);
		res = childRes[0] > res[0] ? childRes : res;
	}

	return res;
}
CSGNodeOperationType IntersectionOperation::operationType() const
{
	return CSGNodeOperationType::Intersection;
}
std::tuple<int, int> IntersectionOperation::numAllowedChilds() const
{
	return std::make_tuple(1, std::numeric_limits<int>::max());
}
Mesh lmu::IntersectionOperation::mesh() const
{
	/*if (_childs.size() == 0)
		return Mesh();
	if (_childs.size() == 1)
		return _childs[0].mesh();

	Mesh res, left, right;
	igl::copyleft::cgal::CSGTree::VectorJ vJ;

	left = _childs[0].mesh();

	for (int i = 1; i < _childs.size(); ++i)
	{
		right = _childs[i].mesh();

		igl::copyleft::cgal::mesh_boolean(left.vertices, left.indices, right.vertices, right.indices, igl::MESH_BOOLEAN_TYPE_INTERSECT, res.vertices, res.indices, vJ);

		left = res;
	}

	return res;*/

	return Mesh();
}

CSGNodePtr DifferenceOperation::clone() const
{
	return std::make_shared<DifferenceOperation>(*this);
}
Eigen::Vector4d DifferenceOperation::signedDistanceAndGradient(const Eigen::Vector3d& p) const
{
	auto left = _childs[0].signedDistanceAndGradient(p);
	auto right = _childs[1].signedDistanceAndGradient(p);

	Eigen::Vector3d grad;	
	double value;

	if (left.x() > -right.x())
	{
		value = left.x();
		grad = Eigen::Vector3d(left.y(), left.z(), left.w());
	}
	else
	{
		value = -right.x();
		grad = Eigen::Vector3d(-right.y(), -right.z(), -right.w());
	}
	
	return Eigen::Vector4d(value, grad.x(), grad.y(), grad.z());
}
CSGNodeOperationType DifferenceOperation::operationType() const
{
	return CSGNodeOperationType::Difference;
}
std::tuple<int, int> DifferenceOperation::numAllowedChilds() const
{
	return std::make_tuple(2, 2);
}
Mesh lmu::DifferenceOperation::mesh() const
{
	/*if (_childs.size() != 2)
		return Mesh();
	
	Mesh res, left, right;
	igl::copyleft::cgal::CSGTree::VectorJ vJ;

	left = _childs[0].mesh();
	right = _childs[1].mesh();

	igl::copyleft::cgal::mesh_boolean(left.vertices, left.indices, right.vertices, right.indices, igl::MESH_BOOLEAN_TYPE_MINUS, res.vertices, res.indices, vJ);
	
	return res;*/

	return Mesh();
}

CSGNodePtr ComplementOperation::clone() const
{
	return std::make_shared<ComplementOperation>(*this);
}
Eigen::Vector4d ComplementOperation::signedDistanceAndGradient(const Eigen::Vector3d& p) const
{	
	return _childs[0].signedDistanceAndGradient(p) * -1.0;
}
CSGNodeOperationType ComplementOperation::operationType() const
{
	return CSGNodeOperationType::Complement;
}
std::tuple<int, int> ComplementOperation::numAllowedChilds() const
{
	return std::make_tuple(1, 1);
}
Mesh lmu::ComplementOperation::mesh() const
{
	return Mesh();
}

/*
CSGNodePtr DifferenceRLOperation::clone() const
{
	return std::make_shared<DifferenceRLOperation>(*this);
}
Eigen::Vector4d DifferenceRLOperation::signedDistanceAndGradient(const Eigen::Vector3d & p) const
{
	Eigen::Vector4d res(0, 0, 0, 0);

	auto sdGrad1 = _childs[1].signedDistanceAndGradient(p);
	auto sdGrad2 = (-1.0)*_childs[0].signedDistanceAndGradient(p);

	if (sdGrad2[0] > sdGrad1[0])
		res = sdGrad2;
	else
		res = sdGrad1;

	//Negate gradient
	res[1] = (-1.0)*res[1];
	res[2] = (-1.0)*res[2];
	res[3] = (-1.0)*res[3];

	return res;
}
CSGNodeOperationType DifferenceRLOperation::operationType() const
{
	return CSGNodeOperationType::DifferenceRL;
}
std::tuple<int, int> DifferenceRLOperation::numAllowedChilds() const
{
	return std::make_tuple(2,2);
}
Mesh lmu::DifferenceRLOperation::mesh() const
{
	if (_childs.size() != 2)
		return Mesh();

	Mesh res, left, right;
	igl::copyleft::cgal::CSGTree::VectorJ vJ;

	left = _childs[1].mesh();
	right = _childs[0].mesh();

	igl::copyleft::cgal::mesh_boolean(left.vertices, left.indices, right.vertices, right.indices, igl::MESH_BOOLEAN_TYPE_MINUS, res.vertices, res.indices, vJ);

	return res;
}*/

std::string lmu::operationTypeToString(CSGNodeOperationType type)
{
	switch (type)
	{
	case CSGNodeOperationType::Intersection:
		return "Intersection";
	case CSGNodeOperationType::Difference:
		return "Difference";
	case CSGNodeOperationType::Union:
		return "Union";
	case CSGNodeOperationType::Unknown:
		return "Unknown";
	case CSGNodeOperationType::Complement:
		return "Complement";
	case CSGNodeOperationType::Invalid:
		return "Invalid";
	default:
		return "Undefined Type";
	}
}

std::string lmu::nodeTypeToString(CSGNodeType type)
{
	switch (type)
	{
	case CSGNodeType::Operation:
		return "Operation";
	case CSGNodeType::Geometry:
		return "Geometry";
	default:
		return "Undefined Type";
	}
}

CSGNodeType lmu::stringToNodeType(const std::string & type)
{
	if (type == "Operation")
		return CSGNodeType::Operation;
	else if (type == "Geometry")
		return CSGNodeType::Geometry;
	else
		throw std::runtime_error("Undefined Type.");
}

CSGNodeOperationType lmu::stringToOperationType(const std::string & type)
{
	if (type == "Intersection")
		return CSGNodeOperationType::Intersection;
	else if (type == "Difference")
		return CSGNodeOperationType::Difference;
	else if (type == "Union")
		return CSGNodeOperationType::Union;
	else if (type == "Unknown")
		return CSGNodeOperationType::Unknown;
	else if (type == "Complement")
		return CSGNodeOperationType::Complement;
	else if (type == "Invalid")
		return CSGNodeOperationType::Invalid;
	else
		throw std::runtime_error("Undefined Type.");
}

CSGNode lmu::createOperation(CSGNodeOperationType type, const std::string & name, const std::vector<CSGNode>& childs)
{
	switch (type)
	{
	case CSGNodeOperationType::Union:
		return CSGNode(std::make_shared<UnionOperation>(name, childs));
	case CSGNodeOperationType::Intersection:
		return CSGNode(std::make_shared<IntersectionOperation>(name, childs));
	case CSGNodeOperationType::Difference:
		return CSGNode(std::make_shared<DifferenceOperation>(name, childs));
	case CSGNodeOperationType::Complement:
		return CSGNode(std::make_shared<ComplementOperation>(name, childs));
	default:
		throw std::runtime_error("Operation type is not supported");
	}
}

int lmu::depth(const CSGNode& node, int curDepth)
{
	int maxDepth = curDepth;

	for (const auto& child : node.childs())
	{
		int childDepth = depth(child, curDepth + 1);
		maxDepth = std::max(maxDepth, childDepth);
	}

	return maxDepth;
}

void allGeometryNodePtrsRec(const CSGNode& node, std::vector<CSGNodePtr>& res)
{
	if (node.type() == CSGNodeType::Geometry)
		res.push_back(node.nodePtr());

	for (const auto& child : node.childsCRef())
		allGeometryNodePtrsRec(child, res);
}

std::vector<CSGNodePtr> lmu::allGeometryNodePtrs(const CSGNode& node)
{
	std::vector<CSGNodePtr> res;
	allGeometryNodePtrsRec(node, res);

	return res;
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d> computeDimensions(const CSGNode& node);

/*double lmu::computeGeometryScore(const CSGNode& node, double epsilon, double alpha, const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& funcs) 
{
	auto dims = computeDimensions(node);
	double maxSize = (std::get<1>(dims) - std::get<0>(dims)).maxCoeff();
	double score = 0.0;
	int numPoints = 0;

	for (const auto& func : funcs)
	{
		for (int i = 0; i < func->points().rows(); ++i)
		{
			auto row = func->points().row(i);

			Eigen::Vector3d p = row.head<3>();
			Eigen::Vector3d n = row.tail<3>();

			n.normalize();

			Eigen::Vector4d distAndGrad = node.signedDistanceAndGradient(p);

			double d = distAndGrad[0];// / epsilon;

			Eigen::Vector3d grad = distAndGrad.tail<3>();
			double minusGradientDotN = lmu::clamp(grad.dot(n), -1.0, 1.0); //clamp is necessary, acos is only defined in [-1,1].
			double theta = std::acos(minusGradientDotN) / M_PI;// / alpha;
		
			double scoreDelta = lmu::clamp(abs(d) / maxSize, 0.0, 1.0) + theta;

			score += scoreDelta;			
		}

		numPoints += func->points().rows();
	}

	//std::cout << "ScoreGeo: " << score << std::endl;

	return 1.0 / (score / 2.0 / (double)numPoints);
}*/

double lmu::computeGeometryScore(const CSGNode& node, double epsilon, double alpha, const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& funcs)
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

			double d = distAndGrad[0] / epsilon;

			Eigen::Vector3d grad = distAndGrad.tail<3>();
			double minusGradientDotN = lmu::clamp(/*-*/grad.dot(n), -1.0, 1.0); //clamp is necessary, acos is only defined in [-1,1].
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

int lmu::numNodes(const CSGNode & node)
{
	int num = 1;
	for (const auto& child : node.childsCRef())
	{
		num += numNodes(child);
	}

	return num;
}

int lmu::numPoints(const CSGNode& node)
{
	int n = 0;
	
	for (const auto& c : node.childsCRef())
	{
		if(c.function())
			n += c.function()->points().rows();
		else 
			n += numPoints(c);
	}

	return n;
}

CSGNode* nodeRec(CSGNode& node, int idx, int& curIdx)
{
	if (idx == curIdx)
		return &node;

	curIdx++;

	for (auto& child : node.childsRef())
	{
		auto foundTree = nodeRec(child, idx, curIdx);
		if (foundTree)
			return foundTree;
	}

	return nullptr;
}

CSGNode* lmu::nodePtrAt(CSGNode& node, int idx)
{
	int curIdx = 0;
	return nodeRec(node, idx, curIdx);
}

int nodeDepthRec(const CSGNode& node, int idx, int& curIdx, int depth)
{
	if (idx == curIdx)
		return depth;

	curIdx++;

	for (const auto& child : node.childsCRef())
	{
		auto foundDepth = nodeDepthRec(child, idx, curIdx, depth + 1);
		if (foundDepth != -1)
			return foundDepth;
	}

	return -1;
}

int lmu::depthAt(const CSGNode& node, int idx)
{
	int curIdx = 0;
	return nodeDepthRec(node, idx, curIdx, 0);
}

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, CSGNodePtr> TreeGraph;

void createGraphRec(const CSGNode& node, TreeGraph& graph, size_t parentVertex)
{
	auto v = boost::add_vertex(graph);
	graph[v] = node.nodePtr();
	if (parentVertex < std::numeric_limits<size_t>::max())
		boost::add_edge(parentVertex, v, graph);

	for (auto& child : node.childs())
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

		CSGNodeType type = name[v]->type(); 
		if (type == CSGNodeType::Geometry)
		{
			ss << name[v]->name();

		}
		else if (type == CSGNodeType::Operation)
		{
			ss << operationTypeToString(name[v]->operationType()) << std::endl;
		}

		out << "[label=\"" << ss.str() << "\"]";
	}
private:
	Name name;
};

void lmu::writeNode(const CSGNode& node, const std::string & file)
{
	TreeGraph graph;
	createGraphRec(node, graph, std::numeric_limits<size_t>::max());

	std::ofstream f(file);
	boost::write_graphviz(f, graph, VertexWriter<TreeGraph>(graph));
	f.close();
}

void serializeNodeRec(CSGNode& node, SerializedCSGNode& res)
{
	if (node.childsCRef().size() == 2)
	{		
		res.push_back(NodePart(NodePartType::LeftBracket, nullptr));
		serializeNodeRec(node.childsRef()[0], res);
		res.push_back(NodePart(NodePartType::RightBracket, nullptr));

		res.push_back(NodePart(NodePartType::Node, &node));

		res.push_back(NodePart(NodePartType::LeftBracket, nullptr));
		serializeNodeRec(node.childsRef()[1], res);
		res.push_back(NodePart(NodePartType::RightBracket, nullptr));

	}
	else if (node.childsCRef().size() == 0)
	{
		res.push_back(NodePart(NodePartType::Node, &node));
	}
}

SerializedCSGNode lmu::serializeNode(CSGNode& node)
{
	SerializedCSGNode res;
	serializeNodeRec(node, res);
	return res;
}

CSGNode* getRoot(const SerializedCSGNode& n, int start, int end)
{
	//Note: We assume that n is representing a correct serialization of a tree.

	//end index is exclusive
	int size = end - start;

	if (size == 1)
		return n[start].node;

	int counter = 0;
	for (int i = start; i < end; ++i)
	{
		NodePart np = n[i];

		if (np.type == NodePartType::LeftBracket)
		{
			counter++;
		}
		else if (np.type == NodePartType::RightBracket)
		{
			counter--;
		}

		if (counter == 0)
			return n[i + 1].node;
	}

	return nullptr; 
}

//Code from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_substring#C++_2
/*LargestCommonSubgraph lmu::findLargestCommonSubgraph(const SerializedCSGNode& n1, const SerializedCSGNode& n2)
{
	int startN1 = 0;
	int startN2 = 0;
	int max = 0;
	for (int i = 0; i < n1.size(); i++)
	{
		for (int j = 0; j < n2.size(); j++)
		{
			int x = 0;
			while (n1[i + x] == n2[j + x])
			{
				x++;
				if (((i + x) >= n1.size()) || ((j + x) >= n2.size())) break;
			}
			if (x > max)
			{
				max = x;
				startN1 = i;
				startN2 = j;
			}
		}
	}

	//return S1.substring(Start, (Start + Max));

	return LargestCommonSubgraph(
		getRoot(n1, startN1, startN1 + max),
		getRoot(n2, startN2, startN2 + max), max);
}*/

using SubgraphMap = std::unordered_map<std::string, std::vector<CSGNode*>>;

void getSubgraphsRec(CSGNode& node, SubgraphMap& res, const std::vector<std::string>& blackList)
{
	std::stringstream ss;
	ss << serializeNode(node);
	
	std::string graph = ss.str(); 
	
	if (std::find(blackList.begin(), blackList.end(), graph) == blackList.end())
	{
		res[graph].push_back(&node);
	}
	else
	{
		std::cout << "Blacklisted: " << graph << std::endl;
	}

	for (auto& child : node.childsRef())
		getSubgraphsRec(child, res, blackList);
}

void printSubgraphMap(const SubgraphMap& map)
{
	std::cout << "SubgraphMap:" << std::endl;
	for (const auto& item : map)
	{
		std::cout << item.first << ": " << item.second.size() << std::endl;
	}
}

CommonSubgraph lmu::findLargestCommonSubgraph(CSGNode& n1, CSGNode& n2, const std::vector<std::string>& blackList)
{
	SubgraphMap n1Subgraphs, n2Subgraphs;
	getSubgraphsRec(n1, n1Subgraphs, blackList);
	getSubgraphsRec(n2, n2Subgraphs, blackList);

	//printSubgraphMap(n1Subgraphs);
	//printSubgraphMap(n2Subgraphs);
	
	CommonSubgraph lgs(&n1, &n2, {}, {}, 0);

	for (auto serN1 : n1Subgraphs)
	{
		auto it = n2Subgraphs.find(serN1.first);
		if (it != n2Subgraphs.end())
		{			
			int sgSize = numNodes(*it->second[0]);
			if (lgs.size < sgSize)
			{	
				lgs = CommonSubgraph(&n1, &n2, serN1.second, it->second, sgSize);
			}			
		}
	}

	return lgs;
}

std::vector<CommonSubgraph> lmu::findCommonSubgraphs(CSGNode& n1, CSGNode& n2)
{
	std::vector<CommonSubgraph> res;

	SubgraphMap n1Subgraphs, n2Subgraphs;
	getSubgraphsRec(n1, n1Subgraphs, {});
	getSubgraphsRec(n2, n2Subgraphs, {});

	for (auto serN1 : n1Subgraphs)
	{
		auto it = n2Subgraphs.find(serN1.first);
		if (it != n2Subgraphs.end())
		{
			int sgSize = numNodes(*it->second[0]);			
			auto lgs = CommonSubgraph(&n1, &n2, serN1.second, it->second, sgSize);
			res.push_back(lgs);
		}
	}

	std::sort(res.begin(), res.end(), [](const CommonSubgraph& a, const CommonSubgraph& b)
	{
		return a.size > b.size;
	});

	return res;
}

bool isValidMergeNode(const CSGNode& node, const CSGNode& searchNode, bool allowIntersections)
{
	if (&node == &searchNode)
		return true;
	 
	if (node.type() == CSGNodeType::Operation)
	{
		if (node.operationType() == CSGNodeOperationType::Difference)
		{
			return isValidMergeNode(node.childsCRef()[0], searchNode, allowIntersections);
		}
		else if (node.operationType() == CSGNodeOperationType::Union)
		{
			for (const auto& child : node.childsCRef())
			{
				if (isValidMergeNode(child, searchNode, allowIntersections))
					return true;
			}
		}
		else if (node.operationType() == CSGNodeOperationType::Intersection)
		{
			return allowIntersections;
		}
	}
		
	return false;
}

CSGNode* getValidMergeNode(const CSGNode& root, const std::vector<CSGNode*>& candidateNodes, bool allowIntersections)
{
	for (const auto& candidateNode : candidateNodes)
	{
		if (isValidMergeNode(root, *candidateNode, allowIntersections))
			return candidateNode;
	}

	return nullptr;
}

void mergeNode(CSGNode* dest, const CSGNode& source)
{
	*dest = source;
}

MergeResult lmu::mergeNodes(const CommonSubgraph& lcs, bool allowIntersections)
{
	if (lcs.isEmptyOrInvalid())
		return MergeResult::None;

	CSGNode* validMergeNodeInN1 = getValidMergeNode(*lcs.n1Root, lcs.n1Appearances, allowIntersections);
	CSGNode* validMergeNodeInN2 = getValidMergeNode(*lcs.n2Root, lcs.n2Appearances, allowIntersections);

	if (validMergeNodeInN1 && validMergeNodeInN2)
	{
		if (numNodes(*lcs.n1Root) >= numNodes(*lcs.n2Root))
		{
			mergeNode(validMergeNodeInN1, *lcs.n2Root);
			return MergeResult::First;
		}
		else
		{
			mergeNode(validMergeNodeInN2, *lcs.n1Root);
			return MergeResult::Second;
		}
	}
	else if (validMergeNodeInN1)
	{
		mergeNode(validMergeNodeInN1, *lcs.n2Root);
		return MergeResult::First;
	}
	else if (validMergeNodeInN2)
	{
		mergeNode(validMergeNodeInN2, *lcs.n1Root);
		return MergeResult::Second;
	}
	else
	{
		return MergeResult::None;
	}
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d> computeDimensions(const CSGNode& node)
{	
	auto geos = allGeometryNodePtrs(node);

	double minS = std::numeric_limits<double>::max();
	double maxS = -std::numeric_limits<double>::max();

	Eigen::Vector3d min(minS, minS, minS);
	Eigen::Vector3d max(maxS, maxS, maxS);
	
	for (const auto& geo : geos)
	{
		Eigen::Vector3d minCandidate = geo->function()->meshCRef().vertices.colwise().minCoeff();
		Eigen::Vector3d maxCandidate = geo->function()->meshCRef().vertices.colwise().maxCoeff();

		min(0) = min(0) < minCandidate(0) ? min(0) : minCandidate(0);
		min(1) = min(1) < minCandidate(1) ? min(1) : minCandidate(1);
		min(2) = min(2) < minCandidate(2) ? min(2) : minCandidate(2);

		max(0) = max(0) > maxCandidate(0) ? max(0) : maxCandidate(0);
		max(1) = max(1) > maxCandidate(1) ? max(1) : maxCandidate(1);
		max(2) = max(2) > maxCandidate(2) ? max(2) : maxCandidate(2);
	}

	return std::make_tuple(min, max);
}

Mesh lmu::computeMesh(const CSGNode& node, const Eigen::Vector3i& numSamples, const Eigen::Vector3d& minDim, const Eigen::Vector3d& maxDim)
{
	Eigen::Vector3d min, max;

	if (minDim == Eigen::Vector3d(0.0,0.0,0.0) && maxDim == Eigen::Vector3d(0.0, 0.0, 0.0))
	{
		auto dims = computeDimensions(node);
		min = std::get<0>(dims);
		max = std::get<1>(dims);

		std::cout << "Min: " << min << " Max: " << max << std::endl;
	}
	else
	{
		min = minDim;
		max = maxDim;
	}

	//Add a bit dimensions to avoid cuts (TODO: do it right with modulo stepSize).
	min -= (max - min) * 0.05;
	max += (max - min) * 0.05;

	Eigen::Vector3d stepSize((max(0) - min(0)) / numSamples(0), (max(1) - min(1)) / numSamples(1), (max(2) - min(2)) / numSamples(2));

	int num = numSamples(0)*numSamples(1)*numSamples(2);
	Eigen::MatrixXd samplingPoints(num, 3);
	Eigen::VectorXd samplingValues(num);
		
	for (int x = 0; x < numSamples(0); ++x)
	{
		for (int y= 0; y < numSamples(1); ++y)
		{
			for (int z = 0; z < numSamples(2); ++z)
			{
				int idx = numSamples(0) * numSamples(1) * z + numSamples(0) * y + x;

				Eigen::Vector3d samplingPoint((double)x * stepSize(0) + min(0), (double)y * stepSize(1) + min(1), (double)z * stepSize(2) + min(2));

				samplingPoints.row(idx) = samplingPoint;
			
				samplingValues(idx) = node.signedDistanceAndGradient(samplingPoint)(0);
			
				//if(samplingValues(idx)  < 0)
				//	std::cout << samplingValues(idx) << std::endl;
			}
		}
	}

	Mesh mesh;

	igl::copyleft::marching_cubes(samplingValues, samplingPoints, numSamples(0), numSamples(1), numSamples(2), mesh.vertices, mesh.indices);

	return mesh;
}

bool optimizeCSGNodeStructureRec(CSGNode& node, const std::shared_ptr<ImplicitFunction>& nullFunc)
{
	bool optimizedSomething = false;

	if (node.type() == CSGNodeType::Operation)
	{
		auto& childs = node.childsRef();

		switch (node.operationType())
		{
		case CSGNodeOperationType::Intersection:
						
			if (childs[0].type() == CSGNodeType::Geometry && childs[1].type() == CSGNodeType::Geometry)
			{
				if (childs[0].function() == childs[1].function())
				{
					node = childs[0];
					optimizedSomething = true;
					std::cout << "Optimize Intersection 0" << std::endl;
					break;
				}
			}
			
			if (childs[0].type() == CSGNodeType::Geometry && childs[0].function() == nullFunc)
			{
				node = CSGNode(std::make_shared<CSGNodeGeometry>(nullFunc));
				optimizedSomething = true;
				std::cout << "Optimize Intersection 1" << std::endl;
				break;
			}
			
			if (childs[1].type() == CSGNodeType::Geometry && childs[1].function() == nullFunc)
			{
				node = CSGNode(std::make_shared<CSGNodeGeometry>(nullFunc));
				optimizedSomething = true;
				std::cout << "Optimize Intersection 2" << std::endl;
				break;
			}
			
			break;

		case CSGNodeOperationType::Union:

			if (childs[0].type() == CSGNodeType::Geometry && childs[1].type() == CSGNodeType::Geometry)
			{
				if (childs[0].function() == childs[1].function())
				{
					node = childs[0];
					optimizedSomething = true;
					std::cout << "Optimize Union 0" << std::endl;
					break;
				}
			}
			
			if (childs[0].type() == CSGNodeType::Geometry && childs[0].function() == nullFunc)
			{
				node = childs[1];
				optimizedSomething = true;
				std::cout << "Optimize Union 1" << std::endl;
				break;
			}
			
			if (childs[1].type() == CSGNodeType::Geometry && childs[1].function() == nullFunc)
			{
				node = childs[0];
				optimizedSomething = true;
				std::cout << "Optimize Union 2" << std::endl;
				break;
			}
			
			break;

		case CSGNodeOperationType::Difference:

			if (childs[0].type() == CSGNodeType::Geometry && childs[1].type() == CSGNodeType::Geometry)
			{
				if (childs[0].function() == childs[1].function())
				{
					node = CSGNode(std::make_shared<CSGNodeGeometry>(nullFunc));
					optimizedSomething = true;
					std::cout << "Optimize Difference 0" << std::endl;
					break;
				}
			}
			
			if (childs[0].type() == CSGNodeType::Geometry && childs[0].function() == nullFunc)
			{
				std::vector<CSGNode> childs = { childs[1]};
				node = CSGNode(std::make_shared<ComplementOperation>("Complement", childs));
				optimizedSomething = true;
				std::cout << "Optimize Difference 0" << std::endl;
				break;
			}

			if (childs[1].type() == CSGNodeType::Geometry && childs[1].function() == nullFunc)
			{
				node = childs[0];
				optimizedSomething = true;
				std::cout << "Optimize Difference 1" << std::endl;
				break;
			}

			break;
		}
	}

	if (!optimizedSomething)
	{
		for (auto & child : node.childsRef())
		{
			optimizedSomething |= optimizeCSGNodeStructureRec(child, nullFunc);
		}
	}
	
	return optimizedSomething;
}

void lmu::optimizeCSGNodeStructure(CSGNode& node)
{
	auto nullFunc = std::make_shared<IFNull>("Null");

	while (optimizeCSGNodeStructureRec(node, nullFunc))
	{
		std::cout << "Optimized structure" << std::endl;
	}
}

void lmu::optimizeCSGNode(CSGNode& node, double tolerance)
{
	std::vector<std::shared_ptr<lmu::ImplicitFunction>> funcs;
	auto funcNodes = allGeometryNodePtrs(node);
	for (const auto& funcNode : funcNodes)
		funcs.push_back(funcNode->function());

	double score = computeGeometryScore(node, 1.0, 1.0, funcs);

	double closestScoreDelta = std::numeric_limits<double>::max();
	CSGNodePtr closestScoreFuncNode = nullptr;
	for (const auto& funcNode : funcNodes)
	{
		double funcNodeScore = computeGeometryScore(CSGNode(funcNode), 1.0, 1.0, funcs);
		if (std::abs(score - funcNodeScore) < closestScoreDelta)
		{
			closestScoreDelta = std::abs(score - funcNodeScore);
			closestScoreFuncNode = funcNode;
		}
	}
	
	std::cout << "Try to optimize node. Delta:  " << closestScoreDelta << std::endl;
	if (closestScoreFuncNode && closestScoreDelta <= tolerance)
	{
		std::cout << "optimized node. Delta: " << closestScoreDelta << std::endl;
				
		std::cout << "  from " << serializeNode(node) << std::endl;
		std::cout << "  to   " << serializeNode(CSGNode(closestScoreFuncNode)) << std::endl;

		node = CSGNode(closestScoreFuncNode);		
	}
	else
	{
		for (auto& child : node.childsRef())
		{
			optimizeCSGNode(child, tolerance);
		}
	}
}

Eigen::MatrixXd lmu::computePointCloud(const CSGNode & node, const Eigen::Vector3i & numSamples, double maxDistance, double errorSigma, const Eigen::Vector3d & minDim, const Eigen::Vector3d & maxDim)
{

	Eigen::Vector3d min, max;

	if (minDim == Eigen::Vector3d(0.0, 0.0, 0.0) && maxDim == Eigen::Vector3d(0.0, 0.0, 0.0))
	{
		auto dims = computeDimensions(node);
		min = std::get<0>(dims);
		max = std::get<1>(dims);
	}
	else
	{
		min = minDim;
		max = maxDim;
	}

	//Add a bit dimensions to avoid cuts (TODO: do it right with modulo stepSize).
	min -= (max - min) * 0.05;
	max += (max - min) * 0.05;

	Eigen::Vector3d stepSize((max(0) - min(0)) / numSamples(0), (max(1) - min(1)) / numSamples(1), (max(2) - min(2)) / numSamples(2));

	std::vector<Eigen::Matrix<double,1,6>> samplingPoints;
	
	std::random_device rd{};
	std::mt19937 gen{ rd() };

	std::normal_distribution<> dx{ 0.0 , errorSigma };
	std::normal_distribution<> dy{ 0.0 , errorSigma };
	std::normal_distribution<> dz{ 0.0 , errorSigma };

	for (int x = 0; x < numSamples(0); ++x)
	{
		for (int y = 0; y < numSamples(1); ++y)
		{
			for (int z = 0; z < numSamples(2); ++z)
			{	
				Eigen::Vector3d samplingPoint((double)x * stepSize(0) + min(0), (double)y * stepSize(1) + min(1), (double)z * stepSize(2) + min(2));

				auto samplingValue = node.signedDistanceAndGradient(samplingPoint);
				
				if (abs(samplingValue(0)) < maxDistance)
				{
					Eigen::Matrix<double, 1, 6> sp; 
					sp.row(0) << samplingPoint(0) + dx(gen), samplingPoint(1) + dy(gen), samplingPoint(2) + dz(gen), samplingValue(1), samplingValue(2), samplingValue(3);

					samplingPoints.push_back(sp);
				}
			}
		}
	}

	Eigen::MatrixXd res(samplingPoints.size(), 6);
	for (int i = 0; i < samplingPoints.size(); ++i)
		res.row(i) = samplingPoints[i].row(0);
	
	return res;
}

Eigen::MatrixXd lmu::computePointCloud(const CSGNode & node, int numSamples, double maxDistance, double errorSigma, const Eigen::Vector3d & minDim, const Eigen::Vector3d & maxDim)
{
	Eigen::Vector3d min, max;

	if (minDim == Eigen::Vector3d(0.0, 0.0, 0.0) && maxDim == Eigen::Vector3d(0.0, 0.0, 0.0))
	{
		auto dims = computeDimensions(node);
		min = std::get<0>(dims);
		max = std::get<1>(dims);
	}
	else
	{
		min = minDim;
		max = maxDim;
	}

	Eigen::MatrixXd res(numSamples, 6);

	int i = 0; 

	std::random_device rd{};
	std::mt19937 gen{ rd() };

	std::normal_distribution<> dx{ 0.0 , errorSigma };
	std::normal_distribution<> dy{ 0.0 , errorSigma };
	std::normal_distribution<> dz{ 0.0 , errorSigma };

	std::uniform_real_distribution<> px{ min.x(), max.x() };
	std::uniform_real_distribution<> py{ min.y(), max.y() };
	std::uniform_real_distribution<> pz{ min.z(), max.z() };

	while (i < numSamples)
	{
		Eigen::Vector3d samplingPoint(px(gen), py(gen), pz(gen));
		auto samplingValue = node.signedDistanceAndGradient(samplingPoint);

		if (abs(samplingValue(0)) < maxDistance)
		{
			Eigen::Matrix<double, 1, 6> sp;
			sp.row(0) << samplingPoint(0) + dx(gen), samplingPoint(1) + dy(gen), samplingPoint(2) + dz(gen), samplingValue(1), samplingValue(2), samplingValue(3);
			res.row(i) = sp;
			i++;
		}
	}

	return res;
}

Eigen::VectorXd lmu::computeDistanceError(const Eigen::MatrixXd& samplePoints, const CSGNode& referenceNode, const CSGNode& node, bool normalize)
{
	Eigen::VectorXd res(samplePoints.rows());

	double maxError = 0; 

	for (int i = 0; i < samplePoints.rows(); ++i)
	{
		double error = abs(referenceNode.signedDistanceAndGradient(samplePoints.row(i))(0) - node.signedDistanceAndGradient(samplePoints.row(i))(0));

		maxError = maxError < error ? error : maxError;

		res.row(i) << error;
	}

	if (normalize && maxError > 0.0)
	{
		for (int i = 0; i < samplePoints.rows(); ++i)
		{
			res.row(i) << (res.row(i) / maxError);
		}
	}

	return res;
}

CSGNode nodeFromJson(const Json& json)
{
	auto name = json.at("name").get<std::string>();
	auto type = stringToNodeType(json.at("type").get<std::string>());

	CSGNode node(nullptr);

	if (type == CSGNodeType::Operation)
	{
		auto opType = stringToOperationType(json.at("operationType").get<std::string>());

		node = createOperation(opType, name);

		auto childs = json.at("childs");
		for (Json::iterator it = childs.begin(); it != childs.end(); ++it) 
		{
			node.addChild(nodeFromJson(*it));
		}
	}
	else if (type == CSGNodeType::Geometry)
	{
		auto geoType = json.at("geometryType").get<std::string>();

		Eigen::Vector3d pos(
			json.at("pos").at(0).get<double>(),			
			json.at("pos").at(1).get<double>(), 
			json.at("pos").at(2).get<double>());

		Eigen::Vector3d rot(
			json.at("rot").at(0).get<double>(),
			json.at("rot").at(1).get<double>(),
			json.at("rot").at(2).get<double>());

		Eigen::Affine3d rx = Eigen::Affine3d(Eigen::AngleAxisd(rot.x(), Eigen::Vector3d(1, 0, 0)));
		Eigen::Affine3d ry = Eigen::Affine3d(Eigen::AngleAxisd(rot.y(), Eigen::Vector3d(0, 1, 0)));
		Eigen::Affine3d rz = Eigen::Affine3d(Eigen::AngleAxisd(rot.z(), Eigen::Vector3d(0, 0, 1)));		
		
		auto transform = (Eigen::Affine3d)(Eigen::Translation3d(pos.x(), pos.y(), pos.z())*(rz * ry * rx));

		if (geoType == "Sphere")
		{	
			node = CSGNode(std::make_shared<CSGNodeGeometry>(std::make_shared<IFSphere>(transform, json.at("radius").get<double>(), name)));			
		} 
		else if (geoType == "Box")
		{
			Eigen::Vector3d size(
				json.at("size").at(0).get<double>(),
				json.at("size").at(1).get<double>(),
				json.at("size").at(2).get<double>());

			node = CSGNode(std::make_shared<CSGNodeGeometry>(std::make_shared<IFBox>(transform, size, json.at("subdivisions").get<int>(), name)));
		}
		else if (geoType == "Cylinder")
		{
			node = CSGNode(std::make_shared<CSGNodeGeometry>(std::make_shared<IFCylinder>(transform, json.at("radius").get<double>(), json.at("height").get<double>(), name)));
		}
		else
		{
			throw std::runtime_error("Geometry type unknown.");
		}
	}
	else
	{
		throw std::runtime_error("Operation type unknown.");
	}

	return node;
}

CSGNode lmu::fromJson(const std::string & file)
{
	std::ifstream i(file);


	Json j ;


	i >> j;

	//std::cout << j.dump();

	return nodeFromJson(j);
}

std::ostream& lmu::operator<<(std::ostream& os, const SerializedCSGNode& v)
{
	for (const auto& np : v)
		os << np;

	return os;
}

bool lmu::operator==(const NodePart& lhs, const NodePart& rhs)
{
	if (lhs.type == NodePartType::Node && rhs.type == NodePartType::Node)
	{
		if (lhs.node->type() != rhs.node->type())
			return false;

		if (lhs.node->type() == CSGNodeType::Operation)
			return lhs.node->operationType() == rhs.node->operationType();
		else if (lhs.node->type() == CSGNodeType::Geometry)
			return lhs.node->function() == rhs.node->function();

		return false;
	}

	return lhs.type == rhs.type;
}

bool lmu::operator!=(const NodePart& lhs, const NodePart& rhs)
{
	return !(lhs == rhs);
}

std::ostream& lmu::operator<<(std::ostream& os, const NodePart& np)
{
	switch (np.type)
	{
	case NodePartType::LeftBracket:
		os << "(";
		break;
	case NodePartType::RightBracket:
		os << ")";
		break;
	case NodePartType::Node:
		switch (np.node->type())
		{
		case CSGNodeType::Operation:
			os << operationTypeToString(np.node->operationType());
			break;
		case CSGNodeType::Geometry:
			os << np.node->function()->name();
			break;
		}
		break;
	}
	return os;
}



