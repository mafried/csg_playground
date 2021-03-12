#include "..\include\csgnode.h"
#include "..\include\csgnode_helper.h"
#include "..\include\dnf.h"
#include "..\include\curvature.h"

#include <limits>
#include <fstream>
#include <random>
#include <iostream>

#include <vector>
#include <memory>
#include <array>

#include "boost/graph/graphviz.hpp"
#include <boost/functional/hash.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/math/special_functions/erf.hpp>


#include <igl/copyleft/marching_cubes.h>


#include "../include/constants.h"


using namespace lmu;

CSGNode const CSGNode::invalidNode = CSGNode(nullptr);

CSGNodePtr UnionOperation::clone() const
{
	return std::make_shared<UnionOperation>(*this);
}
Eigen::Vector4d UnionOperation::signedDistanceAndGradient(const Eigen::Vector3d& p, double h) const
{
	Eigen::Vector4d res(0, 0, 0, 0);

	res[0] = std::numeric_limits<double>::max();
	for (const auto& child : _childs)
	{
		auto childRes = child.signedDistanceAndGradient(p,h);
		res = childRes[0] < res[0] ? childRes : res;
	}

	return res;
}
double UnionOperation::signedDistance(const Eigen::Vector3d& p) const
{
	double res = 0.0;

	res = std::numeric_limits<double>::max();
	for (const auto& child : _childs)
	{
		auto childRes = child.signedDistance(p);
		res = childRes < res ? childRes : res;
	}

	return res;
}
CSGNodeOperationType UnionOperation::operationType() const
{
	return CSGNodeOperationType::Union;
}
std::tuple<int, int> UnionOperation::numAllowedChilds() const
{
	return std::make_tuple(2, std::numeric_limits<int>::max());
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
Eigen::Vector4d IntersectionOperation::signedDistanceAndGradient(const Eigen::Vector3d & p, double h) const
{
	Eigen::Vector4d res(0, 0, 0, 0);

	res[0] = -std::numeric_limits<double>::max();
	for (const auto& child : _childs)
	{
		auto childRes = child.signedDistanceAndGradient(p, h);
		res = childRes[0] > res[0] ? childRes : res;
	}

	return res;
}
double IntersectionOperation::signedDistance(const Eigen::Vector3d & p) const
{
	double res = 0.0;

	res = -std::numeric_limits<double>::max();
	for (const auto& child : _childs)
	{
		auto childRes = child.signedDistance(p);
		res = childRes > res ? childRes : res;
	}

	return res;
}
CSGNodeOperationType IntersectionOperation::operationType() const
{
	return CSGNodeOperationType::Intersection;
}
std::tuple<int, int> IntersectionOperation::numAllowedChilds() const
{
	return std::make_tuple(2, std::numeric_limits<int>::max());
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
Eigen::Vector4d DifferenceOperation::signedDistanceAndGradient(const Eigen::Vector3d& p, double h) const
{
	auto left = _childs[0].signedDistanceAndGradient(p, h);
	auto right = _childs[1].signedDistanceAndGradient(p, h);

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
double DifferenceOperation::signedDistance(const Eigen::Vector3d& p) const
{
	auto left = _childs[0].signedDistance(p);
	auto right = _childs[1].signedDistance(p);

	Eigen::Vector3d grad;
	double value;

	if (left > -right)
	{
		value = left;
		
	}
	else
	{
		value = -right;		
	}

	return value;
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
	return Mesh();
}

CSGNodePtr ComplementOperation::clone() const
{
	return std::make_shared<ComplementOperation>(*this);
}
Eigen::Vector4d ComplementOperation::signedDistanceAndGradient(const Eigen::Vector3d& p, double h) const
{	
	return _childs[0].signedDistanceAndGradient(p,h) * -1.0;
}
double ComplementOperation::signedDistance(const Eigen::Vector3d& p) const
{
	return _childs[0].signedDistance(p) * -1.0;
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

CSGNodePtr IdentityOperation::clone() const
{
	return std::make_shared<IdentityOperation>(*this);
}
Eigen::Vector4d IdentityOperation::signedDistanceAndGradient(const Eigen::Vector3d& p, double h) const
{
	return _childs[0].signedDistanceAndGradient(p, h);
}
double IdentityOperation::signedDistance(const Eigen::Vector3d& p) const
{
	return _childs[0].signedDistance(p);
}
CSGNodeOperationType IdentityOperation::operationType() const
{
	return CSGNodeOperationType::Identity;
}
std::tuple<int, int> IdentityOperation::numAllowedChilds() const
{
	return std::make_tuple(1, 1);
}
Mesh lmu::IdentityOperation::mesh() const
{
	return Mesh();
}

CSGNodePtr NoOperation::clone() const
{
	return std::make_shared<NoOperation>(*this);
}
Eigen::Vector4d NoOperation::signedDistanceAndGradient(const Eigen::Vector3d& p, double h) const
{
	return Eigen::Vector4d(std::numeric_limits<double>::max(), 0.0, 0.0, 0.0);
}
double NoOperation::signedDistance(const Eigen::Vector3d& p) const
{
	return std::numeric_limits<double>::max();
}
CSGNodeOperationType NoOperation::operationType() const
{
	return CSGNodeOperationType::Noop;
}
std::tuple<int, int> NoOperation::numAllowedChilds() const
{
	return std::make_tuple(0, 0);
}
Mesh lmu::NoOperation::mesh() const
{
	return Mesh();
}

BoundedPrimitiveOperation::BoundedPrimitiveOperation(const std::string& name, const std::vector<CSGNode>& childs) :
CSGNodeOperation(name, childs),
_diffs(generate_diffs(childs))
{
	writeNode(_diffs, "test_diff.gv");
}
CSGNodePtr BoundedPrimitiveOperation::clone() const
{
	return std::make_shared<BoundedPrimitiveOperation>(*this);
}
Eigen::Vector4d BoundedPrimitiveOperation::signedDistanceAndGradient(const Eigen::Vector3d& p, double h) const
{
	return _diffs.signedDistanceAndGradient(p, h);
}
double BoundedPrimitiveOperation::signedDistance(const Eigen::Vector3d& p) const
{
	return _diffs.signedDistance(p);
}
CSGNodeOperationType BoundedPrimitiveOperation::operationType() const
{
	return CSGNodeOperationType::BoundedPrimitive;
}
std::tuple<int, int> BoundedPrimitiveOperation::numAllowedChilds() const
{
	return std::make_tuple(2, std::numeric_limits<int>::max());
}
Mesh lmu::BoundedPrimitiveOperation::mesh() const
{
	return Mesh();
}
CSGNode lmu::BoundedPrimitiveOperation::generate_diffs(const std::vector<CSGNode>& childs)
{
	if (childs.size() == 1)
		return childs[0];

	auto left_childs = childs;
	left_childs.erase(left_childs.begin());
	return /*opDiff*/ opDiff({ childs[0], opUnion(left_childs) });
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
	case CSGNodeOperationType::Identity:
		return "Identity";
	case CSGNodeOperationType::BoundedPrimitive:
		return "BoundedPrimitive";
	case CSGNodeOperationType::Invalid:
		return "Invalid";
	case CSGNodeOperationType::Noop:
		return "Noop";
	default:
		return "Undefined Type";
	}
}

CSGNodeOperationType lmu::operationTypeFromString(std::string type)
{
	auto parsedType = CSGNodeOperationType::Unknown;

	std::transform(type.begin(), type.end(), type.begin(), ::tolower);
	
	if (type == "union")
		parsedType = CSGNodeOperationType::Union;
	else if (type == "difference" || type == "subtract")
		parsedType = CSGNodeOperationType::Difference;
	else if (type == "intersection" || type == "intersect")
		parsedType = CSGNodeOperationType::Intersection;
	else if (type == "unknown")
		parsedType = CSGNodeOperationType::Unknown;
	else if (type == "complement" || type == "negate")
		parsedType = CSGNodeOperationType::Complement;
	else if (type == "identity")
		parsedType = CSGNodeOperationType::Identity;
	else if (type == "bounded")
		parsedType = CSGNodeOperationType::BoundedPrimitive;
	else if (type == "invalid")
		parsedType = CSGNodeOperationType::Invalid;
	else if (type == "noop")
		parsedType = CSGNodeOperationType::Noop;



	return parsedType;
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
	case CSGNodeOperationType::Complement:
		return CSGNode(std::make_shared<ComplementOperation>(name, childs));
	case CSGNodeOperationType::Identity:
		return CSGNode(std::make_shared<IdentityOperation>(name, childs));
	case CSGNodeOperationType::BoundedPrimitive:
		return CSGNode(std::make_shared<BoundedPrimitiveOperation>(name, childs));
	case CSGNodeOperationType::Noop:
		return CSGNode(std::make_shared<NoOperation>(name));

	default:
		throw std::runtime_error("Operation type is not supported: " + operationTypeToString(type) + ".");
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

std::vector<ImplicitFunctionPtr> lmu::allDistinctFunctions(const CSGNode& node)
{
	std::set<ImplicitFunctionPtr> set;
	std::vector<ImplicitFunctionPtr> vec;

	visit(node, [&set, &vec](const CSGNode& n)
	{
		if (n.type() == CSGNodeType::Geometry)
		{
			if (set.find(n.function()) == set.end())
			{
				set.insert(n.function());
				vec.push_back(n.function());
			}
		}
	});

	return vec;
}

CSGNode lmu::filter_name_duplicates(const CSGNode& node)
{
	CSGNode res = node;

	std::unordered_map<std::string, ImplicitFunctionPtr> func_map; 
	for (const auto& f : allDistinctFunctions(node))
		func_map[f->name()] = f;

	visit(res, [&func_map](CSGNode& n) 
	{
		if (n.type() == CSGNodeType::Geometry)
		{
			n = geometry(func_map[n.function()->name()]);
		}
	});

	return res;
}

void lmu::visit(const CSGNode& node, const std::function<void(const CSGNode&node)>& f)
{
	f(node);
	for (const auto& child : node.childsCRef())	
		visit(child, f);	
}

void lmu::visit(CSGNode& node, const std::function<void(CSGNode&node)>& f)
{
	f(node);
	for (auto& child : node.childsRef())
		visit(child, f);
}

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

double lmu::computeGeometryScore(const CSGNode& node, double epsilon, double alpha, double h, const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& funcs)
{	
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;
	static std::random_device rd;  
	static std::mt19937 gen(rd());

	int num = 0; 

	double score = 0.0;

	for (const auto& func : funcs)
	{
		int numSamples = func->points().rows();//(int)((double)func->points().rows() * 0.01) + 1;

		double perFuncScore = 0.0;

		for (int i = 0; i < numSamples; ++i)
		{
			int idx = i;//du(rd, parmu_t{ 0, (int) func->points().rows() - 1 });

			num++;

			//const double* data = func->pointsCRef().data() + i * 6;
			//Eigen::Vector3d p(data[0], data[1], data[2]);
			//Eigen::Vector3d n(data[3], data[4], data[5]);

			auto row = func->pointsCRef().row(idx);
			Eigen::Vector3d p = row.head<3>();
			Eigen::Vector3d n = row.tail<3>();

			Eigen::Vector4d distAndGrad = node.signedDistanceAndGradient(p,h);

			double d = distAndGrad[0] / epsilon;
			
			Eigen::Vector3d grad = distAndGrad.tail<3>();
			grad.normalize();			
			if (std::isnan(grad.norm()))
			{	
				continue;
			}

			double gradientDotN = lmu::clamp(grad.dot(n), -1.0, 1.0); //clamp is necessary, acos is only defined in [-1,1].
						
			double theta = std::acos(gradientDotN) / alpha;

			double scoreDelta = (std::exp(-(d*d)) + std::exp(-(theta*theta)));

			perFuncScore += (scoreDelta * func->pointWeights()[i] * (func->points().rows() / numSamples));
		}

		score += perFuncScore;
	}

	return score;
}

double lmu::computeRawDistanceScore(const CSGNode & node, const Eigen::MatrixXd & points)
{
	double d = d;

	for (int i = 0; i < points.rows(); ++i)
	{
		auto row = points.row(i);

		Eigen::Vector3d p = row.head<3>();
		Eigen::Vector3d n = row.tail<3>();

		d += std::abs(node.signedDistance(p));
	}

	return d;
}

int lmu::numNodes(const CSGNode& node, bool ignore_leaves)
{
	int num = ignore_leaves && node.childsCRef().size() == 0 ? 0 : 1;
	for (const auto& child : node.childsCRef())
	{
		num += numNodes(child, ignore_leaves);
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

//Note that this function assumes that a convex hull mesh exists for each function.
std::tuple<Eigen::Vector3d, Eigen::Vector3d> 
lmu::computeDimensions(const std::vector<std::shared_ptr<ImplicitFunction>>& shapes)
{
  double minS = std::numeric_limits<double>::max();
  double maxS = -std::numeric_limits<double>::max();

  Eigen::Vector3d min(minS, minS, minS);
  Eigen::Vector3d max(maxS, maxS, maxS);
	
  for (const auto& shape : shapes) {    
    Eigen::Vector3d minCandidate = shape->meshCRef().vertices.colwise().minCoeff();
    Eigen::Vector3d maxCandidate = shape->meshCRef().vertices.colwise().maxCoeff();
    
    min(0) = min(0) < minCandidate(0) ? min(0) : minCandidate(0);
    min(1) = min(1) < minCandidate(1) ? min(1) : minCandidate(1);
    min(2) = min(2) < minCandidate(2) ? min(2) : minCandidate(2);
    
    max(0) = max(0) > maxCandidate(0) ? max(0) : maxCandidate(0);
    max(1) = max(1) > maxCandidate(1) ? max(1) : maxCandidate(1);
    max(2) = max(2) > maxCandidate(2) ? max(2) : maxCandidate(2);
  }
  
  return std::make_tuple(min, max);
}

//Note that this function assumes that a convex hull mesh exists for each function.
std::tuple<Eigen::Vector3d, Eigen::Vector3d> lmu::computeDimensions(const CSGNode& node)
{	
	auto geos = allGeometryNodePtrs(node);
	
	double minS = std::numeric_limits<double>::max();
	double maxS = -std::numeric_limits<double>::max();

	Eigen::Vector3d min(minS, minS, minS);
	Eigen::Vector3d max(maxS, maxS, maxS);
	
	for (const auto& geo : geos)
	{
		//Transform vertices.
		//auto vertices = geo->function()->meshCRef().vertices;
		//for (int i = 0; i < vertices.rows(); i++) {
		//	Eigen::Vector3d v = vertices.row(i);
		//	v = geo->function()->transform() * v;
		//	vertices.row(i) = v;
		//}

		auto mesh = geo->function()->createMesh();
		
		if (mesh.vertices.size() == 0)
			continue;

		Eigen::Vector3d minCandidate = mesh.vertices.colwise().minCoeff();
		Eigen::Vector3d maxCandidate = mesh.vertices.colwise().maxCoeff();

		
		min(0) = min(0) < minCandidate(0) ? min(0) : minCandidate(0);
		min(1) = min(1) < minCandidate(1) ? min(1) : minCandidate(1);
		min(2) = min(2) < minCandidate(2) ? min(2) : minCandidate(2);

		max(0) = max(0) > maxCandidate(0) ? max(0) : maxCandidate(0);
		max(1) = max(1) > maxCandidate(1) ? max(1) : maxCandidate(1);
		max(2) = max(2) > maxCandidate(2) ? max(2) : maxCandidate(2);
	}

	return std::make_tuple(min, max);
}

CSGNode* lmu::findSmallestSubgraphWithImplicitFunctions(CSGNode& node, const std::vector<ImplicitFunctionPtr>& funcs)
{
	auto nfs = lmu::allDistinctFunctions(node);
	std::unordered_set<ImplicitFunctionPtr> nodeFuncs(nfs.begin(), nfs.end());
	
	for (const auto& func : funcs)
	{
		if (nodeFuncs.count(func) == 0)
			return nullptr;
	}	
	CSGNode* foundNode = &node;
	
	for (auto& child : node.childsRef())
	{
		auto childNode = findSmallestSubgraphWithImplicitFunctions(child, funcs);
		if (childNode)
		{	
			foundNode = childNode;			
		}
	}

	return foundNode;
}

Mesh lmu::computeMesh(const CSGNode& node, const Eigen::Vector3i& numSamples, const Eigen::Vector3d& minDim, const Eigen::Vector3d& maxDim)
{
	Eigen::Vector3d min, max;

	if (minDim == Eigen::Vector3d(0.0,0.0,0.0) && maxDim == Eigen::Vector3d(0.0, 0.0, 0.0))
	{
		std::cout << "Compute Dimensions" << std::endl;

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

bool containsNullFunc(const CSGNode& node, const ImplicitFunctionPtr& nullFunc) 
{
	if (node.type() == CSGNodeType::Geometry && node.function() == nullFunc)
	{
		return true;
	}

	bool nullFound = false;

	for (const auto& child : node.childsCRef())
	{
		nullFound = nullFound || containsNullFunc(child, nullFunc);
	}

	return nullFound;
}

void removeAllFunctionDuplicates(std::vector<CSGNode>& childs)
{
	std::sort(childs.begin(), childs.end(), [](const CSGNode& n0, const CSGNode& n1) { return n0.function() > n1.function(); });
	childs.erase(std::unique(childs.begin(), childs.end(), [](const CSGNode& n0, const CSGNode& n1) { return n0.function() == n1.function() && n0.function() != nullptr; }), childs.end());
}

bool optimizeCSGNodeStructureRec(CSGNode& node, const std::shared_ptr<ImplicitFunction>& nullFunc)
{	
	if (node.type() != CSGNodeType::Operation)
		return false; 
	
	auto& childs = node.childsRef();
	size_t oldChildSize = childs.size();
	bool insertedNullFunc = false;

	switch (node.operationType())
	{

	case CSGNodeOperationType::Intersection:
				
		removeAllFunctionDuplicates(childs);
				
		if (childs.size() == 1)
		{
			node = childs.front();
			insertedNullFunc = containsNullFunc(node, nullFunc);
		}
		//if one operand of the intersection is the nullFunc, intersection's result is the nullFunc.
		else if (childs.size() == 0 || std::find_if(childs.begin(), childs.end(), [nullFunc](const CSGNode& n) { return n.function() == nullFunc; }) != childs.end())
		{
			node = geometry(nullFunc);
			insertedNullFunc = true;
		}
								
		break;

	case CSGNodeOperationType::Union:
		
		removeAllFunctionDuplicates(childs);
				
		//Remove all nullfuncs since they do not have any effect in union.
		childs.erase(std::remove_if(childs.begin(), childs.end(), [nullFunc](const CSGNode& n) {return n.function() == nullFunc; }), childs.end());
				
		if (childs.size() == 0)
		{
			node = geometry(nullFunc);
			insertedNullFunc = true;
		}
		else if (childs.size() == 1)
		{
			node = childs.front();
			insertedNullFunc = containsNullFunc(node, nullFunc);
		}
		
		break;

	case CSGNodeOperationType::Difference:
				
		if (childs.size() == 0)
		{	
			node = geometry(nullFunc);
			insertedNullFunc = true;
		}
		else if (childs.size() == 1)
		{
			node = childs.front();
			insertedNullFunc = containsNullFunc(node, nullFunc);
		}
		else if (childs.size() == 2)
		{	
			if (childs[0].type() == CSGNodeType::Geometry && childs[0].function() == childs[1].function())
			{
				node = geometry(nullFunc);
				insertedNullFunc = true;
			}
			else if (childs[0].function() == nullFunc)
			{	
				node = opComp({ childs[1] });					
				insertedNullFunc = containsNullFunc(node, nullFunc);

			}
			else if (childs[1].function() == nullFunc)
			{
				node = childs[0];	
				insertedNullFunc = containsNullFunc(node, nullFunc);
			}
		}
		else
		{
			std::cout << "Difference with more than two operands detected." << std::endl;
		}

		break;

	case CSGNodeOperationType::Complement:
				
		if (childs.size() == 1 && childs[0].function() == nullFunc)
		{
			node = geometry(nullFunc);
			insertedNullFunc = true;
		}
		else if (childs.size() != 1) {
			std::cout << "Complement operation with more than 1 child\n";
		}
		
		break;
	}
	
	if (!insertedNullFunc)
	{
		for (auto& child : node.childsRef())
		{
			insertedNullFunc |= optimizeCSGNodeStructureRec(child, nullFunc);
		}
	}

	return insertedNullFunc;
}

int lmu::optimizeCSGNodeStructure(CSGNode& node)
{
	auto nullFunc = std::make_shared<IFNull>("Null");
	
	//writeNode(node, "debug_bef.dot");
	//std::cout << "Optimize Node" << std::endl;

	int i = 0;
	int limit = 10;
	while (optimizeCSGNodeStructureRec(node, nullFunc))
	{	
		if (node.function() == nullFunc)
		{
			std::cout << "added noop." << std::endl;
			node = opNo();
			break;
		}
		
		i++;

		if (i > limit)
			std::cout << "WARNING: over limit " << i << std::endl;
	}	

	
	//if (containsNullFunc(node, nullFunc))
	//{
	//	std::cout << "WARNING: tree still contains a null function" << std::endl;
	//	writeNode(node, "debug.dot");
	//	std::terminate();
	//}

	return i;
}

void lmu::optimizeCSGNode(CSGNode& node, double tolerance)
{
	std::vector<std::shared_ptr<lmu::ImplicitFunction>> funcs;
	auto funcNodes = allGeometryNodePtrs(node);
	for (const auto& funcNode : funcNodes)
		funcs.push_back(funcNode->function());

	double score = computeGeometryScore(node, 1.0, 1.0, 0.001, funcs);

	double closestScoreDelta = std::numeric_limits<double>::max();
	CSGNodePtr closestScoreFuncNode = nullptr;
	for (const auto& funcNode : funcNodes)
	{
		double funcNodeScore = computeGeometryScore(CSGNode(funcNode), 1.0, 1.0, 0.001, funcs);
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
		CSGNode closest(closestScoreFuncNode);
		//std::cout << "  to   " << serializeNode(CSGNode(closestScoreFuncNode)) << std::endl;
		std::cout << "  to   " << serializeNode(closest) << std::endl;


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

void lmu::convertToTreeWithMaxNChilds(CSGNode& node, int n)
{
	auto& childs = node.childsRef();
	n = clamp(n, std::get<0>(node.numAllowedChilds()), std::get<1>(node.numAllowedChilds()));

	if (childs.size() > n)
	{
		CSGNode newChild = createOperation(node.operationType());

		for (int i = n - 1; i < childs.size(); ++i)
		{
			newChild.addChild(childs[i]);
		}

		childs.erase(childs.begin() + n - 1, childs.end());
		node.addChild(newChild);
	}
	
	for (auto& child : childs)
	{
		convertToTreeWithMaxNChilds(child, n);
	}	
}

lmu::PointCloud lmu::computePointCloud(const CSGNode& node, const CSGNodeSamplingParams& params)
{

	Eigen::Vector3d min, max;

	if (params.minDim == Eigen::Vector3d(0.0, 0.0, 0.0) && params.maxDim == Eigen::Vector3d(0.0, 0.0, 0.0))
	{
		auto dims = computeDimensions(node);
		min = std::get<0>(dims);
		max = std::get<1>(dims);
	}
	else
	{
		min = params.minDim;
		max = params.maxDim;
	}

	//Add a bit dimensions to avoid cuts (TODO: do it right with modulo stepSize).
	double extend = params.samplingStepSize * 2.0;
	min -= Eigen::Vector3d(extend, extend, extend);
	max += Eigen::Vector3d(extend, extend, extend);

	//Eigen::Vector3d stepSize((max(0) - min(0)) / numSamples(0), (max(1) - min(1)) / numSamples(1), (max(2) - min(2)) / numSamples(2));

	Eigen::Vector3i numSamples((max(0) - min(0)) / params.samplingStepSize, (max(1) - min(1)) / params.samplingStepSize, (max(2) - min(2)) / params.samplingStepSize);

	std::cout << "Number of samples: " << numSamples(0) 
		  << " " << numSamples(1) 
		  << " " << numSamples(2) << std::endl;

	std::vector<Eigen::Matrix<double,1,6>> validSamplingPoints;
	
	for (int x = 0; x < numSamples(0); ++x)
	{
		for (int y = 0; y < numSamples(1); ++y)
		{
			for (int z = 0; z < numSamples(2); ++z)
			{	
				Eigen::Vector3d samplingPoint((double)x * params.samplingStepSize + min(0), (double)y * params.samplingStepSize + min(1), (double)z * params.samplingStepSize + min(2));

				auto samplingValue = node.signedDistanceAndGradient(samplingPoint);
				
				if (abs(samplingValue(0)) < params.maxDistance)
				{
					Eigen::Matrix<double, 1, 6> sp; 
					sp.row(0) << samplingPoint(0), samplingPoint(1), samplingPoint(2), samplingValue(1), samplingValue(2), samplingValue(3);

					validSamplingPoints.push_back(sp);
				}
			}
		}
	}

	PointCloud res(validSamplingPoints.size(), 6);
	for (int i = 0; i < validSamplingPoints.size(); ++i)
		res.row(i) = validSamplingPoints[i].row(0);

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
			os << (np.node->function() ? np.node->function()->name() : "EMPTY FUNC");
			break;
		}
		break;
	}
	return os;
}

size_t lmu::CSGNodeOperation::hash(size_t seed) const
{
	boost::hash_combine(seed, operationType());
	for (const auto& child : _childs)
		seed = child.hash(seed);

	return seed;
}

size_t lmu::CSGNodeGeometry::hash(size_t seed) const
{
	boost::hash_combine(seed, reinterpret_cast<std::uintptr_t>(_function.get()));
	return seed;
}

lmu::CSGNodeSamplingParams::CSGNodeSamplingParams() :
	samplingStepSize(0.0),
	maxDistance(0.0),
	maxAngleDistance(0.0),
	errorSigma(0.0),
	minDim(Eigen::Vector3d(0,0,0)),
	maxDim(Eigen::Vector3d(0, 0, 0))
{
}

lmu::CSGNodeSamplingParams::CSGNodeSamplingParams(double maxDistance, double maxAngleDistance, double errorSigma, double samplingStepSize, const Eigen::Vector3d & min, const Eigen::Vector3d & max) :
	samplingStepSize(samplingStepSize == 0.0 ? maxDistance * 2.0 : samplingStepSize),
	maxDistance(maxDistance),
	maxAngleDistance(maxAngleDistance),
	errorSigma(errorSigma),
	minDim(min),
	maxDim(max)
{
}

void lmu::reducePointsBasedOnVariance(const std::vector<std::shared_ptr<ImplicitFunction>>& functions, const lmu::Graph& graph, double h)
{
	double maxVariance = 0.0;
	double minVariance = std::numeric_limits<double>::max();
	for (auto& f : functions)
	{
		std::vector<Eigen::Matrix<double, 1, 6>> selectedPoints;
		std::vector<double> selectedPointWeights;

		//Check orientation
		int numSameSide = 0;
		for (int i = 0; i < f->pointsCRef().rows(); ++i)
		{
			Eigen::Vector3d p = f->pointsCRef().row(i).leftCols(3);
			Eigen::Vector3d n = f->pointsCRef().row(i).rightCols(3);
			Eigen::Vector3d g = f->signedDistanceAndGradient(p, h).bottomRows(3);
			numSameSide += g.dot(n) > 0.0;
		}
		bool outside = numSameSide >= f->pointsCRef().rows() / 2;

		for (auto& f2 : functions)
		{
			if (f == f2 || !areConnected(graph, f, f2))
				continue;

			int idx = 0;
			double curLargestVariance = 0.0;

			auto g1 = geometry(f);
			auto g2 = geometry(f2);
			std::array<CSGNode,2> operations = {opUnion({g1, g2}), opInter({g1,g2})};

			int minDistOpBestPoint = -1;

			std::array<size_t, 2> bestOperationCounter = {0,0};

			for (int i = 0; i < f->pointsCRef().rows(); ++i)
			{
				Eigen::Vector3d p = f->pointsCRef().row(i).leftCols(3);
				Eigen::Vector3d n = f->pointsCRef().row(i).rightCols(3);
				Eigen::Vector3d g = f->signedDistanceAndGradient(p, h).bottomRows(3);

				std::array<double, operations.size()> distances;
				double distSum = 0.0;
				double minDist = std::numeric_limits<double>::max();
				int minDistOp = 0;
				for (int j = 0; j < operations.size(); ++j)
				{
					distances[j] = operations[j].signedDistance(p);
					distSum += distances[j];

					if (std::abs(distances[j]) < minDist)
					{
						minDist = std::abs(distances[j]);
						minDistOp = j;
					}
				}
				double distMean = distSum / operations.size();

				bestOperationCounter[minDistOp]++;


				/*double distVar = 0.0;
				for (int j = 0; j < operations.size(); ++j)
				{
					distVar += ((distances[j] - distMean) * (distances[j] - distMean));
				}*/
				double distVar = std::abs(distances[0] - distances[1]);


				if (distVar > curLargestVariance)
				{
					idx = i;
					curLargestVariance = distVar;
					minDistOpBestPoint = minDistOp;

					//std::cout << "Var: " << distVar << std::endl;
				}
			}
					
			auto bestOpIdx = std::max_element(bestOperationCounter.begin(), bestOperationCounter.end()) - bestOperationCounter.begin();
			
			std::cout << f->name() << " " << f2->name() << ": " << curLargestVariance << " MinDistIdx: " << minDistOpBestPoint << " " << bestOperationCounter[0] << "|" 
				<< bestOperationCounter[1] << "|" << bestOperationCounter[2] << std::endl;

			if(bestOpIdx != minDistOpBestPoint)
				std::cout << "###################################"  << std::endl;

			auto point = f->points().row(idx);
			Eigen::Vector3d p = point.leftCols(3);
			Eigen::Vector3d n = point.rightCols(3);
			Eigen::Vector3d g = f->signedDistanceAndGradient(p, h).bottomRows(3);

			Eigen::Matrix<double, 1, 6> newPoint;
			newPoint << p.transpose(), (outside ? g : -g).transpose();

			selectedPoints.push_back(newPoint);
			selectedPointWeights.push_back(curLargestVariance);

			if (maxVariance < curLargestVariance)
				maxVariance = curLargestVariance;

			if (minVariance > curLargestVariance)
				minVariance = curLargestVariance;
		}

		//f->setScoreWeight(f->points().rows() / selectedPoints.size());

		PointCloud pc;
		pc.resize(selectedPoints.size(), 6);
		for (int i = 0; i < pc.rows(); ++i)
		{
pc.row(i) = selectedPoints[i];
		}
		f->points() = pc;
		f->pointWeights() = selectedPointWeights;

		//std::cout << f->name() << ": " << (outside ? "Outside" : "Inside") << " " << (g.dot(n) > 0.0 ? "Outside" : "Inside") << std::endl;
	}

	std::cout << "Max Variance: " << maxVariance << std::endl;
	std::cout << "Min Variance: " << minVariance << std::endl;

	for (auto& f : functions)
	{
		for (int i = 0; i < f->pointWeights().size(); ++i)
		{
			f->pointWeights()[i] = maxVariance / f->pointWeights()[i];
			std::cout << f->name() << "-var: " << f->pointWeights()[i] << std::endl;
		}
	}
}


inline double median(std::vector<double> v)
{
	std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
	return v[v.size() / 2];
}

std::tuple<double, double> scaled3MADAndMedian(const lmu::ImplicitFunctionPtr& f)
{
	std::vector<double> values(f->pointsCRef().rows());

	for (int j = 0; j < f->pointsCRef().rows(); ++j)
	{
		Eigen::Vector3d p = f->pointsCRef().row(j).leftCols(3);		
		values[j] = std::abs(f->signedDistance(p));
	}

	double med = median(values);
	std::transform(values.begin(), values.end(), values.begin(), [med](double v) -> double { return std::abs(v - med); });

	const double c = -1.0 / (std::sqrt(2.0)*boost::math::erfc_inv(3.0 / 2.0));

	return std::make_tuple(c * median(values) * 3.0, med);
}

void lmu::filterPoints(const std::vector<std::shared_ptr<ImplicitFunction>>& functions, const lmu::Graph& graph, double h, bool useSelection)
{
	auto outlierTestValues = computeOutlierTestValues(functions, h);

	//Check and set orientation.
	for (auto& f : functions)
	{						
		int numSameSide = 0;
		for (int i = 0; i < f->pointsCRef().rows(); ++i)
		{
			Eigen::Vector3d p = f->pointsCRef().row(i).leftCols(3);
			Eigen::Vector3d n = f->pointsCRef().row(i).rightCols(3);
			Eigen::Vector3d g = f->signedDistanceAndGradient(p, h).bottomRows(3);
			numSameSide += g.dot(n) >= 0.0;
		}
		bool outside = numSameSide >= f->pointsCRef().rows() / 2;

		f->setNormalsPointOutside(outside);
	}
	
	//Project points.
	for (auto& f : functions)
	{
		auto points = f->points();
		auto distOutlierTest = scaled3MADAndMedian(f);

		for (int i = 0; i < points.rows(); ++i)
		{
			Eigen::Vector3d p = points.row(i).leftCols(3);
			Eigen::Vector3d n = points.row(i).rightCols(3);
			Eigen::Vector4d dg = f->signedDistanceAndGradient(p, h);
			Eigen::Vector3d g = dg.bottomRows(3).normalized();
			double d = dg(0);

			if (std::abs(std::abs(d) - std::get<1>(distOutlierTest)) < std::get<0>(distOutlierTest))
			{
				p = p - (g*d);
				g = f->signedDistanceAndGradient(p, h).bottomRows(3).normalized();
			}
			
			//points.row(i) << p.transpose(), g.transpose();

			Eigen::Vector3d mg(-g);			
			points.row(i) << p.transpose(), (f->normalsPointOutside() ? g : mg).transpose();
		}

		f->points() = points;
		std::cout << "projected. " << std::endl;
	}

	for (auto& f : functions)
	{
		std::vector<Eigen::Matrix<double, 1, 6>> selectedPoints;
		std::vector<double> selectedPointWeights;
		std::vector<Eigen::Matrix<double, 1, 6>> points;
		
		for (int i = 0; i < f->pointsCRef().rows(); ++i)
		{
			Eigen::Vector3d p = f->pointsCRef().row(i).leftCols(3);
			Eigen::Vector3d g = f->pointsCRef().row(i).rightCols(3); 
			double d = f->signedDistance(p);

			if (std::abs(d) > 0.000001)
			{
				continue;
			}

			bool co = false;
			for (int j = 0; j < functions.size(); ++j)
			{
				if (functions[j] == f)
					continue;

				if (std::abs(d) >= std::abs(functions[j]->signedDistance(p)))
				{
					//std::cout << "HERE-------------------" << std::endl;
					co = true;					
					break;
				}
			}
			if (co) continue;
								
			bool nc = false;
			for (int i = 0; i < functions.size(); ++i)
			{
				if (functions[i] == f)
					continue;

				if (!lmu::areConnected(graph, functions[i], f))
					continue;
							
				if (std::abs(functions[i]->signedDistance(p)) < 0.1)
				{
					bool f1Outside = f->normalsPointOutside();
					bool f2Outside = functions[i]->normalsPointOutside();

					Eigen::Vector4d g1 = f->signedDistanceAndGradient(p, h).bottomRows(3);
					Eigen::Vector4d g2 = functions[i]->signedDistanceAndGradient(p, h).bottomRows(3);

					//std::cout << "CLOSE" << std::endl;

					if (f1Outside != f2Outside )//&& g1.dot(g2) < 0.0)
					{
						//std::cout << "----------------------------------------NOW " << g1.dot(g2) << std::endl;
						nc = true;
						break;
					}
				}
			}
			if (nc) continue;

			/*auto outlierTestValue = outlierTestValues[f];
			lmu::Curvature c = lmu::curvature(p.transpose(), geometry(f), 0.1);
			double deviationFromFlatness = std::sqrt(c.k1 * c.k1 + c.k2 * c.k2);
			double median = std::get<1>(outlierTestValue);
			double maxDelta = std::get<0>(outlierTestValue);
			if (std::abs(deviationFromFlatness - median) > maxDelta)
			{
				//std::cout << p << std::endl;
				continue;
			}*/

			Eigen::Matrix<double, 1, 6> pg;
			pg << p.transpose(), g.transpose();
			points.push_back(pg);
			//pointDistances.push_back((f->normalsPointOutside() ? 1.0 : -1.0)*d);
		}

		if (!useSelection)
		{
			PointCloud pc1(points.empty() ? 1 : points.size(), 6);
			for (int i = 0; i < points.size(); ++i)
			{
				pc1.row(i) = points[i];
			}
			f->points() = pc1;

			continue;
		}

		//Selection
		std::vector<Eigen::Matrix<double, 1, 6>> perFuncPoints;		
		for (int i = 0; i < functions.size(); ++i)
		{
			if (!lmu::areConnected(graph, functions[i], f) || points.empty())
				continue;
				
			size_t idxMaxDistance;
			size_t idxMinDistance; 
			double minDistance = std::numeric_limits<double>::max(); 
			double maxDistance = 0.0;
			Eigen::Vector3d pmax, pmin;


			std::vector<std::tuple<double, size_t>> cds; 
			for (int j = 0; j < points.size(); ++j)
			{
				Eigen::Vector3d p = points[j].leftCols(3).transpose();

				double d = (functions[i]->pos() - p).squaredNorm();
								
				cds.push_back(std::make_tuple(d, j));

				if (minDistance > d)
				{
					minDistance = d; 
					idxMinDistance = j; 
					pmin = p;
				}

				if (maxDistance < d)
				{
					maxDistance = d;
					idxMaxDistance = j;
					pmax = p;
				}
			}

			std::sort(cds.begin(), cds.end(), [](const auto& p1, const auto& p2) {return std::get<0>(p1) > std::get<0>(p2);});
			
			idxMaxDistance = std::get<1>(cds.front());
			idxMinDistance = std::get<1>(cds.back());

			size_t idxMedDistance = std::get<1>(cds.at(cds.size() / 2));

			//perFuncDistances.push_back(pointDistances[idxMaxDistance]);
			//perFuncDistances.push_back(pointDistances[idxMinDistance]);
			//perFuncDistances.push_back(pointDistances[idxMedDistance]);
			
			perFuncPoints.push_back(points[idxMaxDistance]);
			perFuncPoints.push_back(points[idxMinDistance]);		
			perFuncPoints.push_back(points[idxMedDistance]);

			//std::cout << "(" << points[idxMaxDistance].x() << "," << points[idxMaxDistance].y() << "," << points[idxMaxDistance].z() << ")" << " D max: " << f->signedDistance(points[idxMaxDistance].leftCols(3).transpose()) << " " << pointDistances[idxMaxDistance] << std::endl;
			//std::cout << "(" << points[idxMinDistance].x() << "," << points[idxMinDistance].y() << "," << points[idxMinDistance].z() << ")" << " D min: " << f->signedDistance(points[idxMinDistance].leftCols(3).transpose()) << " " << pointDistances[idxMinDistance] << std::endl;
			//std::cout << "(" << points[idxMedDistance].x() << "," << points[idxMedDistance].y() << "," << points[idxMedDistance].z() << ")" << " D med: " << f->signedDistance(points[idxMedDistance].leftCols(3).transpose()) << " " << pointDistances[idxMedDistance] << std::endl;

		}
	
		std::cout << f->name() << ": " << perFuncPoints.size() << std::endl;

		/*if (f->name() == "cylinder_0") //this point is too close to cylinder_1 => add check
		{
			
			auto p = perFuncPoints[1];

			std::cout << "=========================" << p << std::endl;
			
			perFuncPoints.clear();
			perFuncPoints.push_back(p);
		}
		else
		{
			perFuncPoints.clear();
		}*/

		PointCloud pc;
		pc.resize(perFuncPoints.size(), 6);
		for (int i = 0; i < perFuncPoints.size(); ++i)
		{
			pc.row(i) << perFuncPoints[i];

			//lmu::Curvature c = lmu::curvature(perFuncPoints[i].leftCols(3), geo, h * 10.0);
			//double deviationFromFlatness = std::sqrt(c.k1 * c.k1 + c.k2 * c.k2);
			//std::cout << f->name() << " dev: " << deviationFromFlatness << std::endl;

		}

		f->points() = pc;
		//f->pointWeights() = perFuncDistances;
	}

	std::cout << "DONE" << std::endl;
}


