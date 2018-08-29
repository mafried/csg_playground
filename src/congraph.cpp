#include <random>

#include "..\include\congraph.h"
#include "..\include\mesh.h"
#include "..\include\collision.h"

#include "boost/graph/graphviz.hpp"
#include "boost/graph/bron_kerbosch_all_cliques.hpp"
#include <boost/dynamic_bitset.hpp>

std::ostream& lmu::operator<<(std::ostream& os, const lmu::Clique& c)
{
	os << "Clique#";
	for (const auto& f : c.functions)
		os << f->name() << " ";
	os << "#";
	return os;
}

bool lmu::areConnected(const lmu::Graph & g, const std::shared_ptr<lmu::ImplicitFunction>& f1, const std::shared_ptr<lmu::ImplicitFunction>& f2)
{
	return boost::edge(g.vertexLookup.at(f1), g.vertexLookup.at(f2), g).second;
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

void createConnectionGraphRec(const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& funcs, const Eigen::Vector3d & min, const Eigen::Vector3d & max, double minCellSize, std::vector<boost::dynamic_bitset<>>& overlaps)
{
	//std::cout << "part: " << std::endl << min << std::endl << max << std::endl;

	if ((max - min).norm() < minCellSize)
		return;

	Eigen::Vector3d s = (max - min);
	Eigen::Vector3d p = min + 0.5 * s;
	
	boost::dynamic_bitset<> isIn(funcs.size());
	
	for (int i = 0; i < funcs.size(); ++i)	
		isIn[i] = funcs[i]->signedDistance(p) < 0.0;	
	
	for (int i = 0; i < funcs.size(); ++i)
		overlaps[i] = isIn[i] ? overlaps[i] | isIn : overlaps[i]; // overlaps[i] | isIn;
		
	createConnectionGraphRec(funcs, min, min + 0.5 * s, minCellSize, overlaps);
	createConnectionGraphRec(funcs, min + Eigen::Vector3d(s.x() * 0.5, 0.0, 0.0),			min + Eigen::Vector3d(s.x() * 0.5, 0.0, 0.0) + 0.5 * s, minCellSize, overlaps);
	createConnectionGraphRec(funcs, min + Eigen::Vector3d(0.0, s.y() * 0.5, 0.0),			min + Eigen::Vector3d(0.0, s.y() * 0.5, 0.0) + 0.5 * s, minCellSize, overlaps);
	createConnectionGraphRec(funcs, min + Eigen::Vector3d(s.x() * 0.5, s.y() * 0.5, 0.0),	min + Eigen::Vector3d(s.x() * 0.5, s.y() * 0.5, 0.0) + 0.5 * s, minCellSize, overlaps);

	createConnectionGraphRec(funcs, min + Eigen::Vector3d(0.0, 0.0, s.z() * 0.5),			min + Eigen::Vector3d(0.0, 0.0, s.z() * 0.5) + 0.5 * s, minCellSize, overlaps);
	createConnectionGraphRec(funcs, min + Eigen::Vector3d(s.x() * 0.5, 0.0, s.z() * 0.5),	min + Eigen::Vector3d(s.x() * 0.5, 0.0, s.z() * 0.5) + 0.5 * s, minCellSize, overlaps);
	createConnectionGraphRec(funcs, min + Eigen::Vector3d(0.0, s.y() * 0.5, s.z() * 0.5),	min + Eigen::Vector3d(0.0, s.y() * 0.5, s.z() * 0.5) + 0.5 * s, minCellSize, overlaps);
	createConnectionGraphRec(funcs, min + Eigen::Vector3d(s.x() * 0.5, s.y() * 0.5, s.z() * 0.5), min + Eigen::Vector3d(s.x() * 0.5, s.y() * 0.5, s.z() * 0.5) + 0.5 * s, minCellSize, overlaps);
		
	//All bits are set.
	if (isIn.count() == funcs.size())
		return; 
}

lmu::Graph lmu::createConnectionGraph(const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& impFuncs, const Eigen::Vector3d & min, const Eigen::Vector3d & max, double minCellSize)
{
	lmu::Graph graph;
	std::vector<boost::dynamic_bitset<>> overlaps(impFuncs.size());

	int i = 0;
	for (const auto& impFunc : impFuncs)
	{
		auto v = boost::add_vertex(graph);
		graph[v] = impFunc;
		graph.vertexLookup[impFunc] = v;

		overlaps[i++] = boost::dynamic_bitset<>(impFuncs.size(), false);
	}

	createConnectionGraphRec(impFuncs, min, max, minCellSize, overlaps);

	boost::graph_traits<Graph>::vertex_iterator vi1, vi1_end;

	i = 0;
	for (boost::tie(vi1, vi1_end) = boost::vertices(graph); vi1 != vi1_end; ++vi1)
	{
		const auto& v1 = graph[*vi1];

		boost::graph_traits<Graph>::vertex_iterator vi2, vi2_end;

		int j = 0;
		for (boost::tie(vi2, vi2_end) = boost::vertices(graph); vi2 != vi2_end; ++vi2)
		{
			std::cout << overlaps[i][j] << " ";

			if (i == j)
				break;
			
			const auto& v2 = graph[*vi2];

			//Add an edge if both primitives collide.
			if (v1 != v2 && overlaps[i][j])
				boost::add_edge(*vi1, *vi2, graph);
			j++;
		}

		std::cout << std::endl;

		i++;
	}
	
	return graph;
}

lmu::Graph lmu::createRandomConnectionGraph(int numVertices, double edgePropability)
{
	Graph graph;

	for (int i = 0; i < numVertices; ++i)
	{
		auto v = boost::add_vertex(graph);
		graph[v] = std::make_shared<IFNull>("Null_" + std::to_string(i));
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
