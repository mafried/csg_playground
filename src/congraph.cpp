#include <random>

#include "..\include\congraph.h"
#include "..\include\mesh.h"
#include "..\include\collision.h"

#include "boost/graph/graphviz.hpp"
#include "boost/graph/bron_kerbosch_all_cliques.hpp"
#include <boost/graph/connected_components.hpp>

std::ostream& lmu::operator<<(std::ostream& os, const lmu::Clique& c)
{
	os << "Clique#";
	for (const auto& f : c.functions)
		os << f->name() << " ";
	os << "#";
	return os;
}

lmu::Graph lmu::createConnectionGraph(const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& impFuncs)
{
	Graph graph;

	for (const auto& impFunc : impFuncs)
	{
		auto v = boost::add_vertex(graph);
		graph[v] = impFunc;
		graph.vertexLookup[impFunc] = v;
	}
	
	boost::graph_traits<Graph>::vertex_iterator vi1, vi1_end;

	int i = 0; 
	for (boost::tie(vi1, vi1_end) = boost::vertices(graph); vi1 != vi1_end; ++vi1)
	{
		const auto& v1 = graph[*vi1];
		
		boost::graph_traits<Graph>::vertex_iterator vi2, vi2_end;

		int j = 0; 
		for (boost::tie(vi2, vi2_end) = boost::vertices(graph); vi2 != vi2_end; ++vi2)
		{
			if (i == j)
				break;

			const auto& v2 = graph[*vi2];

			//Add an edge if both primitives collide.
			if (v1 != v2 && lmu::collides(*v1, *v2))			
				boost::add_edge(*vi1, *vi2, graph);		

			j++;
		}

		i++;
	}

	return graph;
}

lmu::Graph lmu::createRandomConnectionGraph(double edgePropability, int numConnectedComponents, const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& impFuncs)
{
	int curNum = 0;
	Graph graph;
	while (curNum != numConnectedComponents)
	{
		graph = createRandomConnectionGraph(edgePropability, impFuncs);
		std::vector<int> component(boost::num_vertices(graph));
		curNum = boost::connected_components(graph, &component[0]);
	}
	return graph;
}

lmu::Graph lmu::createRandomConnectionGraph(double edgePropability, const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& impFuncs)
{
	Graph graph;

	for (const auto& fun : impFuncs)
	{
		auto v = boost::add_vertex(graph);		

		graph[v] = fun;//std::make_shared<IFNull>("Null_" + std::to_string(i));
		graph.vertexLookup[graph[v]] = v;
	}

	boost::graph_traits<Graph>::vertex_iterator vi1, vi1_end;

	int i = 0;
	for (boost::tie(vi1, vi1_end) = boost::vertices(graph); vi1 != vi1_end; ++vi1)
	{
		const auto& v1 = graph[*vi1];

		boost::graph_traits<Graph>::vertex_iterator vi2, vi2_end;

		int j = 0;
		for (boost::tie(vi2, vi2_end) = boost::vertices(graph); vi2 != vi2_end; ++vi2)
		{
			if (i == j)
				break;

			const auto& v2 = graph[*vi2];

			std::random_device rd;
			std::mt19937 mt(rd());
			std::uniform_real_distribution<double> dist(0.0, 1.0);

			//Add an edge if both primitives collide.
			if (v1 != v2 && dist(mt) <= edgePropability)
				boost::add_edge(*vi1, *vi2, graph);

			j++;
		}

		i++;
	}

	return graph;
}

template <class Name>
class VertexWriter {
public:
	VertexWriter(Name _name) : name(_name) {}
	template <class VertexOrEdge>
	void operator()(std::ostream& out, const VertexOrEdge& v) const {
		out << "[label=\"" << name[v]->name() << "\"]";
	}
private:
	Name name;
};

void lmu::writeConnectionGraph(const std::string& file, lmu::Graph & graph)
{	

	std::ofstream f(file);
	boost::write_graphviz(f, graph, VertexWriter<Graph>(graph));
	f.close();
}

struct CliqueCollector
{
	CliqueCollector(std::vector<lmu::Clique>& cliques) :
		cliques(cliques)
	{

	}

	template <typename Clique, typename Graph>
	void clique(const Clique& c, const Graph& g)
	{
		lmu::Clique clique;
		// Iterate over the clique and print each vertex within it.
		typename Clique::const_iterator i, end = c.end();
		for (i = c.begin(); i != end; ++i)
		{
			clique.functions.push_back(g[*i]);
		}
		cliques.push_back(clique);
		std::cout << "CLIQUE! " << cliques.size() << std::endl;
	}

	std::vector<lmu::Clique>& cliques;
};

std::vector<lmu::Clique> lmu::getCliques(const lmu::Graph & graph)
{
	std::vector<lmu::Clique> cliques;
	CliqueCollector cc(cliques);

	// Use the Bron-Kerbosch algorithm to find all cliques.
	boost::bron_kerbosch_all_cliques(graph, cc);

	return cliques;
}
