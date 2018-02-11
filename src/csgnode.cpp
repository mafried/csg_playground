#include "..\include\csgnode.h"

#include <limits>
#include <fstream>

#include "boost/graph/graphviz.hpp"

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>

#include <igl/copyleft/cgal/mesh_boolean.h>
#include <igl/copyleft/cgal/CSGTree.h>

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
	if (_childs.size() == 0)
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

	return res;	
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
	if (_childs.size() == 0)
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

	return res;
}

CSGNodePtr DifferenceOperation::clone() const
{
	return std::make_shared<DifferenceOperation>(*this);
}
/*
/*
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
*/
Eigen::Vector4d DifferenceOperation::signedDistanceAndGradient(const Eigen::Vector3d& p) const
{
	Eigen::Vector4d res(0, 0, 0, 0);

	auto sdGrad1 = _childs[0].signedDistanceAndGradient(p);
	auto sdGrad2 = (-1.0)*_childs[1].signedDistanceAndGradient(p);

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
	if (_childs.size() != 2)
		return Mesh();
	
	Mesh res, left, right;
	igl::copyleft::cgal::CSGTree::VectorJ vJ;

	left = _childs[0].mesh();
	right = _childs[1].mesh();

	igl::copyleft::cgal::mesh_boolean(left.vertices, left.indices, right.vertices, right.indices, igl::MESH_BOOLEAN_TYPE_MINUS, res.vertices, res.indices, vJ);
	
	return res;
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

void lmu::writeNode(const CSGNode & node, const std::string & file)
{
	TreeGraph graph;
	createGraphRec(node, graph, std::numeric_limits<size_t>::max());

	std::ofstream f(file);
	boost::write_graphviz(f, graph, VertexWriter<TreeGraph>(graph));
	f.close();
}

