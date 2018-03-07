#ifndef CSGTREE_H
#define CSGTREE_H

#include <vector>
#include <memory>

#include "helper.h"

#include <Eigen/Core>
#include "congraph.h"

#include "test.h"
#include "mesh.h"

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>

namespace lmu
{
	struct CliqueEdge
	{
		std::unordered_set<std::shared_ptr<ImplicitFunction>> sharedIfs; 
		std::unordered_map<lmu::Clique*,std::unordered_set<std::shared_ptr<ImplicitFunction>>> separatedIfs;
	};

	struct CliqueIFReplacements
	{
		CliqueIFReplacements(std::unordered_set<std::shared_ptr<ImplicitFunction>> sharedIfs, std::unordered_set<std::shared_ptr<ImplicitFunction>> separatedIfs):
			sharedIfs(sharedIfs),
			separatedIfs(separatedIfs)
		{
		}

		CliqueIFReplacements()
		{
		}

		std::unordered_set<std::shared_ptr<ImplicitFunction>> sharedIfs;
		std::unordered_set<std::shared_ptr<ImplicitFunction>> separatedIfs;

	};

	struct CliqueEdge_t
	{
		typedef boost::edge_property_tag kind;
	};

	struct CliqueVertex
	{
		lmu::Clique* clique;
	};
	
	struct CliqueVertex_t
	{
		typedef boost::vertex_property_tag kind;
	};

	typedef boost::graph_traits<Graph>::edge_descriptor CliqueEdgeDesc;
	typedef boost::graph_traits<Graph>::vertex_descriptor CliqueVertexDesc;
	typedef boost::property<boost::edge_weight_t, int, boost::property<CliqueEdge_t, CliqueEdge>> CliqueEdgeProperty;
	typedef boost::property<CliqueVertex_t, CliqueVertex> CliqueVertexProperty;

	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, CliqueVertexProperty, CliqueEdgeProperty> CliqueGraph;
	
	enum class OperationType
	{
		Unknown = 0,
		Intersection, 
		Union,
		DifferenceLR,
		DifferenceRL,
		Complement
	};

	int numOpTypes();

	std::string opTypeToString(OperationType type);

	struct ImplicitFunction;

	struct CSGTree
	{
		CSGTree(const std::vector<CSGTree>& c);
		CSGTree();

		OperationType operation; 		
		std::vector<std::shared_ptr<ImplicitFunction>> functions;
		std::vector<lmu::CSGTree> childs; 

		CliqueIFReplacements iFReplacements;

		void write(const std::string& file);

		lmu::Mesh createMesh() const;

		Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& point) const;

		double computeGeometryScore(double epsilon, double alpha, const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& funcs) const;

		void fillUnknownOperations(const std::vector<lmu::Clique>& cliques);

		void resolveIFReplacements();

		int depth(int curDepth = 0) const;

		int numNodes() const;

		int numPoints() const;

		CSGTree* node(int idx);
		int nodeDepth(int idx) const;

		int sizeWithFunctions() const;

		std::vector<std::shared_ptr<ImplicitFunction>> functionsRecursively() const;

		std::string info() const
		{
			return std::string();
		}
	};  

	std::string info(const lmu::CSGTree& tree);

	lmu::CSGTree createCSGTreeTemplateFromCliques(std::vector<lmu::Clique>& cliques);
	lmu::CSGTree createCSGTreeWithGA(const std::vector<std::shared_ptr<ImplicitFunction>>& shapes, const lmu::Graph& connectionGraph = lmu::Graph());
	
	lmu::CSGTree createCSGTreeFromCliqueGraph(lmu::CliqueGraph& cliqueGraph);

	lmu::CSGTree createCSGTreeFromCliqueGraph(const lmu::CliqueGraph& cliqueGraph, 
		const boost::property_map<lmu::CliqueGraph, lmu::CliqueVertex_t>::type& cliqueMap, std::vector<lmu::CliqueEdgeDesc> minCliqueTreeEdges);
}

#endif