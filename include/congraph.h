#ifndef CONGRAPH_H
#define CONGRAPH_H

#include <vector>
#include <Eigen/Core>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/filtered_graph.hpp>
#include "boost/graph/copy.hpp"

namespace boost
{
	struct edge_component_t
	{
		enum
		{
			num = 555
		};
		typedef edge_property_tag kind;
	};

	struct edge_pruned_t
	{
		enum
		{
			num = 666
		};
		typedef edge_property_tag kind;
	};
}

namespace lmu
{	
	struct ImplicitFunction;

	struct Clique
	{
		Clique()
		{
		}

		Clique(const std::vector<std::shared_ptr<ImplicitFunction>>& funcs) : 
			functions(funcs)
		{
		}

		std::vector<std::shared_ptr<ImplicitFunction>> functions;
	};

	std::ostream& operator<<(std::ostream& os, const Clique& c);


	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, 
		std::shared_ptr<lmu::ImplicitFunction>, 
		boost::property<boost::edge_component_t, std::size_t, 
		boost::property<boost::edge_pruned_t, bool>>> GraphStructure;

	using EdgeDescriptor = boost::graph_traits<GraphStructure>::edge_descriptor;
	using VertexDescriptor = boost::graph_traits<GraphStructure>::vertex_descriptor;

	struct Graph 
	{
		GraphStructure structure;
		std::unordered_map<std::shared_ptr<lmu::ImplicitFunction>, VertexDescriptor> vertexLookup;

		Graph& operator= (const Graph& other)
		{
			if (this != &other)
			{
				vertexLookup = other.vertexLookup; 
				boost::copy_graph(other.structure, structure);
			}
			return *this;
		}

		Graph()
		{
		}

		Graph(const Graph& other) : 
			vertexLookup(other.vertexLookup)
		{	
				boost::copy_graph(other.structure, structure);
		}
	};
			
	VertexDescriptor addVertex(lmu::Graph& g, const std::shared_ptr<lmu::ImplicitFunction>& f);
	EdgeDescriptor addEdge(lmu::Graph& g, const VertexDescriptor& v1, const VertexDescriptor& v2);

	bool areConnected(const lmu::Graph& g, const std::shared_ptr<lmu::ImplicitFunction>& f1, const std::shared_ptr<lmu::ImplicitFunction>& f2);

	//WARNING: NOT CORRECT ANYMORE!
	bool wasPruned(const lmu::Graph& g, const std::shared_ptr<lmu::ImplicitFunction>& f);

	lmu::Graph createConnectionGraph(const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& impFuncs);

	lmu::Graph createConnectionGraph(const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& impFuncs, const Eigen::Vector3d& min, const Eigen::Vector3d& max, double minCellSize);
	
	lmu::Graph createRandomConnectionGraph(int numVertices, double edgePropability);

	void writeConnectionGraph(const std::string& file, const lmu::Graph& graph);

	std::vector<lmu::Clique> getCliques(const lmu::Graph& graph);	

	std::vector<std::shared_ptr<lmu::ImplicitFunction>> getImplicitFunctions(const lmu::Graph& graph);

	std::vector<lmu::Graph> getConnectedComponents(const Graph &g);

	std::vector<lmu::Graph> getBridgeSeparatedConnectedComponents(const Graph& g);

	std::vector<lmu::Graph> getArticulationPointSeparatedConnectedComponents(const Graph& g);

	lmu::Graph pruneGraph(const lmu::Graph& g);

	using NeighborMap = std::unordered_map<VertexDescriptor, std::unordered_set<VertexDescriptor>>;
	using PruneList = std::unordered_map<VertexDescriptor, std::vector<VertexDescriptor>>;

	lmu::NeighborMap createNeighborMap(const lmu::Graph& g);
	lmu::PruneList createPruneList(const lmu::Graph& g, const lmu::NeighborMap& neighborMap);
	lmu::Graph pruneGraph(const lmu::Graph& g, const lmu::PruneList& pruneList);
	lmu::Graph recreatePrunedGraph(const lmu::Graph& originalGraph, const lmu::Graph& prunedGraph, const lmu::PruneList& pruneList);
	
	size_t numVertices(const lmu::Graph& g);

	size_t numEdges(const lmu::Graph& g);

	lmu::Graph getGraphWithPrunedVertices(const lmu::Graph& g, const std::shared_ptr<lmu::ImplicitFunction>& f);
	
	using EdgeFilterPredicate = std::function<bool(const EdgeDescriptor& edge)>;
	using VertexFilterPredicate = std::function<bool(const VertexDescriptor& vertex)>;

	
	lmu::Graph filterGraph(const lmu::Graph& g, const EdgeFilterPredicate& predicate);
	lmu::Graph filterGraph(const lmu::Graph& g, const VertexFilterPredicate& vp, const EdgeFilterPredicate& ep);

	void recreateVertexLookup(Graph& graph);
}

#endif