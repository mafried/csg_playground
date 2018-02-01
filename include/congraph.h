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

	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, std::shared_ptr<lmu::ImplicitFunction>> Graph;
			
	lmu::Graph createConnectionGraph(const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& impFuncs);

	lmu::Graph createRandomConnectionGraph(int numVertices, double edgePropability);

	void writeConnectionGraph(const std::string& file, lmu::Graph& graph);

	std::vector<lmu::Clique> getCliques(const lmu::Graph& graph);	
}

#endif