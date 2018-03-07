#ifndef CONGRAPH_H
#define CONGRAPH_H

#include <vector>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>

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

	struct  Graph : public boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, std::shared_ptr<lmu::ImplicitFunction>>
	{
		std::unordered_map<std::shared_ptr<lmu::ImplicitFunction>, vertex_descriptor> vertexLookup;
	};
			
	lmu::Graph createConnectionGraph(const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& impFuncs);

	lmu::Graph createRandomConnectionGraph(int numVertices, double edgePropability);

	void writeConnectionGraph(const std::string& file, lmu::Graph& graph);

	std::vector<lmu::Clique> getCliques(const lmu::Graph& graph);	
}

#endif