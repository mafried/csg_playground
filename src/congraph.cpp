#include <random>

#include "..\include\congraph.h"
#include "..\include\mesh.h"
#include "..\include\collision.h"

#include "boost/graph/graphviz.hpp"
#include "boost/graph/bron_kerbosch_all_cliques.hpp"
#include "boost/graph/copy.hpp"

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
	return boost::edge(g.vertexLookup.at(f1), g.vertexLookup.at(f2), g.structure).second;
}

lmu::Graph lmu::createConnectionGraph(const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& impFuncs)
{
	Graph graph;

	for (const auto& impFunc : impFuncs)
	{
		auto v = boost::add_vertex(graph.structure);
		graph.structure[v] = impFunc;
		graph.vertexLookup[impFunc] = v;
	}
	
	boost::graph_traits<GraphStructure>::vertex_iterator vi1, vi1_end;

	int i = 0; 
	for (boost::tie(vi1, vi1_end) = boost::vertices(graph.structure); vi1 != vi1_end; ++vi1)
	{
		const auto& v1 = graph.structure[*vi1];
		
		boost::graph_traits<GraphStructure>::vertex_iterator vi2, vi2_end;

		int j = 0; 
		for (boost::tie(vi2, vi2_end) = boost::vertices(graph.structure); vi2 != vi2_end; ++vi2)
		{
			if (i == j)
				break;

			const auto& v2 = graph.structure[*vi2];

			//Add an edge if both primitives collide.
			if (v1 != v2 && lmu::collides(*v1, *v2))			
				boost::add_edge(*vi1, *vi2, graph.structure);

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
		auto v = boost::add_vertex(graph.structure);
		graph.structure[v] = impFunc;
		graph.vertexLookup[impFunc] = v;

		overlaps[i++] = boost::dynamic_bitset<>(impFuncs.size(), false);
	}

	createConnectionGraphRec(impFuncs, min, max, minCellSize, overlaps);

	boost::graph_traits<GraphStructure>::vertex_iterator vi1, vi1_end;

	i = 0;
	for (boost::tie(vi1, vi1_end) = boost::vertices(graph.structure); vi1 != vi1_end; ++vi1)
	{
		const auto& v1 = graph.structure[*vi1];

		boost::graph_traits<GraphStructure>::vertex_iterator vi2, vi2_end;

		int j = 0;
		for (boost::tie(vi2, vi2_end) = boost::vertices(graph.structure); vi2 != vi2_end; ++vi2)
		{
			std::cout << overlaps[i][j] << " ";

			if (i == j)
				break;
			
			const auto& v2 = graph.structure[*vi2];

			//Add an edge if both primitives collide.
			if (v1 != v2 && overlaps[i][j])
				boost::add_edge(*vi1, *vi2, graph.structure);
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
		auto v = boost::add_vertex(graph.structure);
		graph.structure[v] = std::make_shared<IFNull>("Null_" + std::to_string(i));
		graph.vertexLookup[graph.structure[v]] = v;
	}

	boost::graph_traits<GraphStructure>::vertex_iterator vi1, vi1_end;

	int i = 0;
	for (boost::tie(vi1, vi1_end) = boost::vertices(graph.structure); vi1 != vi1_end; ++vi1)
	{
		const auto& v1 = graph.structure[*vi1];

		boost::graph_traits<GraphStructure>::vertex_iterator vi2, vi2_end;

		int j = 0;
		for (boost::tie(vi2, vi2_end) = boost::vertices(graph.structure); vi2 != vi2_end; ++vi2)
		{
			if (i == j)
				break;

			const auto& v2 = graph.structure[*vi2];

			std::random_device rd;
			std::mt19937 mt(rd());
			std::uniform_real_distribution<double> dist(0.0, 1.0);

			//Add an edge if both primitives collide.
			if (v1 != v2 && dist(mt) <= edgePropability)
				boost::add_edge(*vi1, *vi2, graph.structure);

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
	boost::write_graphviz(f, graph.structure, VertexWriter<GraphStructure>(graph.structure));
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
	boost::bron_kerbosch_all_cliques(graph.structure, cc);

	return cliques;
}

std::vector<std::shared_ptr<lmu::ImplicitFunction>> lmu::getImplicitFunctions(const lmu::Graph & graph)
{
	std::vector<std::shared_ptr<lmu::ImplicitFunction>> res;
	res.reserve(graph.vertexLookup.size());

	for (const auto& pair : graph.vertexLookup)
	{
		res.push_back(pair.first);
	}

	return res;
}

void lmu::recreateVertexLookup(lmu::Graph& graph)
{
	boost::graph_traits<lmu::GraphStructure>::vertex_iterator vi1, vi1_end;

	int i = 0;
	for (boost::tie(vi1, vi1_end) = boost::vertices(graph.structure); vi1 != vi1_end; ++vi1)
	{
		const auto& v1 = graph.structure[*vi1];

		graph.vertexLookup[v1] = *vi1;
	}
}

//From https://stackoverflow.com/questions/26763193/return-a-list-of-connected-component-subgraphs-in-boost-graph
std::vector<lmu::Graph> lmu::getConnectedComponents(lmu::Graph const & g)
{
	using cid = lmu::GraphStructure::vertices_size_type;
	std::map<lmu::GraphStructure::vertex_descriptor, cid> mapping;
	//std::map<Graph::vertex_descriptor, default_color_type> colors;

	cid num = boost::connected_components(
		g.structure,
		boost::make_assoc_property_map(mapping)//,
		//color_map(make_assoc_property_map(colors))
	);

	std::vector<lmu::GraphStructure> componentGraphs(num);

	std::map<lmu::GraphStructure::vertex_descriptor, int> vim;
	for (auto const& vd : boost::make_iterator_range(vertices(g.structure)))
		vim.emplace(vd, vim.size());

	for (cid i = 0; i < num; i++)
	{
		typedef boost::filtered_graph<
			GraphStructure,
			std::function<bool(GraphStructure::edge_descriptor)>,
			std::function<bool(GraphStructure::vertex_descriptor)>
		> FilteredView;

		boost::copy_graph(FilteredView(g.structure,
			[&](GraphStructure::edge_descriptor e) {
			return mapping[source(e, g.structure)] == i
				|| mapping[target(e, g.structure)] == i;
		},
			[&](GraphStructure::vertex_descriptor v) {
			return mapping[v] == i;
		}
			),
			componentGraphs[i],
			vertex_index_map(boost::make_assoc_property_map(vim)));
	}

	std::vector<lmu::Graph> res;
	res.reserve(componentGraphs.size());

	for (const auto& componentGraph : componentGraphs)
	{
		lmu::Graph g;
		g.structure = componentGraph; 
		recreateVertexLookup(g);
		res.push_back(g);
	}

	return res;
	
}