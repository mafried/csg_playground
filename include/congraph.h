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

	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, std::shared_ptr<lmu::ImplicitFunction>> GraphStructure;

	struct Graph 
	{
		GraphStructure structure;
		std::unordered_map<std::shared_ptr<lmu::ImplicitFunction>, GraphStructure::vertex_descriptor> vertexLookup;

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
			

	bool areConnected(const lmu::Graph& g, const std::shared_ptr<lmu::ImplicitFunction>& f1, const std::shared_ptr<lmu::ImplicitFunction>& f2);

	lmu::Graph createConnectionGraph(const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& impFuncs);

	lmu::Graph createConnectionGraph(const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& impFuncs, const Eigen::Vector3d& min, const Eigen::Vector3d& max, double minCellSize);
	
	lmu::Graph createRandomConnectionGraph(int numVertices, double edgePropability);

	void writeConnectionGraph(const std::string& file, lmu::Graph& graph);

	std::vector<lmu::Clique> getCliques(const lmu::Graph& graph);	

	std::vector<std::shared_ptr<lmu::ImplicitFunction>> getImplicitFunctions(const lmu::Graph& graph);

	std::vector<lmu::Graph> getConnectedComponents(Graph const&g);
	
	void recreateVertexLookup(Graph& graph);

}

#endif