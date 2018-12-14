#include <random>

#include "..\include\congraph.h"
#include "..\include\mesh.h"
#include "..\include\collision.h"

#include "boost/graph/graphviz.hpp"
#include "boost/graph/bron_kerbosch_all_cliques.hpp"
#include "boost/graph/copy.hpp"
#include <boost/graph/biconnected_components.hpp>
#include <boost/graph/adjacency_list.hpp>


#include <boost/dynamic_bitset.hpp>

std::ostream& lmu::operator<<(std::ostream& os, const lmu::Clique& c)
{
	os << "Clique#";
	for (const auto& f : c.functions)
		os << f->name() << " ";
	os << "#";
	return os;
}

lmu::VertexDescriptor lmu::addVertex(lmu::Graph& g, const std::shared_ptr<lmu::ImplicitFunction>& f)
{
	auto v = boost::add_vertex(g.structure);
	g.structure[v] = f;
	g.vertexLookup[f] = v;

	return v;
}

lmu::EdgeDescriptor lmu::addEdge(lmu::Graph &g, const VertexDescriptor & v1, const VertexDescriptor & v2)
{
	return std::get<0>(boost::add_edge(v1, v2, g.structure));
}

bool lmu::areConnected(const lmu::Graph& g, const std::shared_ptr<lmu::ImplicitFunction>& f1, const std::shared_ptr<lmu::ImplicitFunction>& f2)
{
	return boost::edge(g.vertexLookup.at(f1), g.vertexLookup.at(f2), g.structure).second;
}

std::vector<std::shared_ptr<lmu::ImplicitFunction>> lmu::getConnectedImplicitFunctions(const lmu::Graph& g, const std::shared_ptr<lmu::ImplicitFunction>& f)
{
	auto v = g.vertexLookup.at(f);

	std::vector<std::shared_ptr<lmu::ImplicitFunction>> neighbors;
	boost::graph_traits<GraphStructure>::adjacency_iterator neighbour, neighbour_end;

	//Get neighbor set.
	for (boost::tie(neighbour, neighbour_end) = boost::adjacent_vertices(v, g.structure); neighbour != neighbour_end; ++neighbour)
		neighbors.push_back(g.structure[*neighbour]);

	return neighbors;
}

bool lmu::wasPruned(const lmu::Graph& g, const std::shared_ptr<lmu::ImplicitFunction>& f)
{
	if (numEdges(g) == 0)
		return false;

	auto v = g.vertexLookup.at(f);

	typename boost::graph_traits < lmu::GraphStructure >::out_edge_iterator ei, ei_end;
	
	//Get pruned property map 
	boost::property_map <GraphStructure, boost::edge_pruned_t >::const_type
		pruned = boost::get(boost::edge_pruned_t(), g.structure);
	
	//A pruned vertex is a vertex with all edges being pruned.
	for (boost::tie(ei, ei_end) = boost::out_edges(v, g.structure); ei != ei_end; ++ei)
	{		
		if (!pruned[*ei])
			return false;
	}

	return true;
}

lmu::Graph lmu::createConnectionGraph(const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& impFuncs)
{
	Graph graph;

	for (const auto& impFunc : impFuncs)
	{
		addVertex(graph, impFunc);
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
				addEdge(graph, *vi1, *vi2);

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
		isIn[i] = funcs[i]->signedDistance(p) < 0.0;//s.x(); // < 0
	
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
		addVertex(graph, impFunc);
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
			//std::cout << overlaps[i][j] << " ";

			if (i == j)
				break;
			
			const auto& v2 = graph.structure[*vi2];

			//Add an edge if both primitives collide.
			if (v1 != v2 && overlaps[i][j])
				addEdge(graph, *vi1, *vi2);
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
		addVertex(graph, std::make_shared<IFNull>("Null_" + std::to_string(i)));
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
				addEdge(graph, *vi1, *vi2);

			j++;
		}

		i++;
	}

	return graph;
}

template <class Name>
class VertexWriter 
{
public:
	VertexWriter(Name _name) : name(_name) {}
	template <class VertexOrEdge>
	void operator()(std::ostream& out, const VertexOrEdge& v) const 
	{
		auto wasPruned = lmu::wasPruned(name, name.structure[v]);
		
		out << "[label=\"" << name.structure[v]->name() << (wasPruned ? "_PRUNED" : "") << "\"]";
	}
private:
	Name name;
};

void lmu::writeConnectionGraph(const std::string& file, const lmu::Graph & graph)
{	

	std::ofstream f(file);
	boost::write_graphviz(f, graph.structure, VertexWriter<lmu::Graph>(graph));
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
			std::function<bool(EdgeDescriptor)>,
			std::function<bool(VertexDescriptor)>
		> FilteredView;

		boost::copy_graph(FilteredView(g.structure,
			[&](EdgeDescriptor e) {
			return mapping[source(e, g.structure)] == i
				|| mapping[target(e, g.structure)] == i;
		},
			[&](VertexDescriptor v) {
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

/*
- Pruning has no impact on found bridges (pruned edges are always bridges)

1. Pruning 
2. Prime implicant partitioning 
   (simple: PIs with no pruned connection => simply remove it and later union, complex: PIs => remove together with all connected pruned and later union 
3. Per partition: bridge partitioning

*/

std::vector<lmu::Graph> lmu::getBridgeSeparatedConnectedComponents(const Graph& g)
{	
	/*
		Find bridges.
		From https://stackoverflow.com/questions/27105367/finding-bridges-in-a-graph-c-boost:
		1. Calculate the biconnected components
		2. Create a edge counter for each component. Iteratate over all edge and increase the coresponding edge counter of the respective component
		3. Iterate again over all edges and check if the edge counter of the corresponding component is 1, if so this edge is a bridge
	*/

	lmu::Graph res = g;

	//Get component property map 
	boost::property_map <GraphStructure, boost::edge_component_t >::type
		component = boost::get(boost::edge_component_t(), res.structure);

	//Get pruned property map
	//boost::property_map <GraphStructure, boost::edge_pruned_t >::type
	//	pruned = boost::get(boost::edge_pruned_t(), res.structure);


	//1
	auto numComponents = biconnected_components(res.structure, component);

	//2
	std::vector<size_t> edgeCounter (numComponents, 0);
	boost::graph_traits <GraphStructure>::edge_iterator ei, ei_end;
	for (boost::tie(ei, ei_end) = boost::edges(res.structure); ei != ei_end; ++ei)
			edgeCounter[component[*ei]]++;

	//3 + remove all bridges
	res = filterGraph(res, [&edgeCounter, &component](const EdgeDescriptor& edge)
	{
		return edgeCounter[component[edge]] != 1; // || pruned[edge];
	});
	
	//
	return getConnectedComponents(res);
}

std::vector<lmu::Graph> lmu::getArticulationPointSeparatedConnectedComponents(const Graph& g)
{
	auto gCopy = g;
	std::vector<lmu::Graph> res;

	//Get component property map 
	boost::property_map <GraphStructure, boost::edge_component_t >::type
		component = boost::get(boost::edge_component_t(), gCopy.structure);

	auto numComponents = biconnected_components(gCopy.structure, component);

	std::cout << "Num Components: " << numComponents << std::endl;

	std::vector<VertexDescriptor> artPoints;
	boost::articulation_points(gCopy.structure, std::back_inserter(artPoints));

	std::unordered_map<VertexDescriptor, std::unordered_set<size_t>> componentIds;
	std::unordered_map<VertexDescriptor, std::vector<size_t>> sortedComponentIds;

	//Iterate over all neighboring edges of all vertices to get components that a certain vertex is part of.
	boost::graph_traits<GraphStructure>::vertex_iterator vi, vi_end;
	for (boost::tie(vi, vi_end) = boost::vertices(gCopy.structure); vi != vi_end; ++vi)
	{
		typename boost::graph_traits <lmu::GraphStructure>::out_edge_iterator ei, ei_end;
		for (boost::tie(ei, ei_end) = boost::out_edges(*vi, gCopy.structure); ei != ei_end; ++ei)
		{
			componentIds[*vi].insert(component[*ei]);
		}

		sortedComponentIds[*vi].insert(sortedComponentIds[*vi].end(), componentIds[*vi].begin(), componentIds[*vi].end());
		std::sort(sortedComponentIds[*vi].begin(), sortedComponentIds[*vi].end());
	}


	
	for (auto ap : artPoints)
	{
		std::cout << "ART POINT " << gCopy.structure[ap]->name() << std::endl;
		for (size_t c : componentIds[ap])
			std::cout << c << std::endl;

		auto pGraph = lmu::filterGraph(gCopy,
			[&gCopy, &ap, &artPoints, &sortedComponentIds](const VertexDescriptor& v)
		{ 
			if (v == ap)
				return true;

			//Vertex must not be an articulation point itself.
			bool isArticulationPoint = std::find(artPoints.begin(), artPoints.end(), v) != artPoints.end();
			if (isArticulationPoint)
				return false; 

			//Vertex must belong to one of the ap's components
			std::vector<size_t> intersection;
			std::set_intersection(sortedComponentIds[v].begin(), sortedComponentIds[v].end(), sortedComponentIds[ap].begin(), sortedComponentIds[ap].end(), std::back_inserter(intersection));
			bool shareComponents = !intersection.empty();

			/*std::cout << "   SHARED COMPONENT " << gCopy.structure[v]->name() << ": " << shareComponents << std::endl;
			for (size_t c : sortedComponentIds[v])
				std::cout << "   "<< c << std::endl;
			std::cout << "    ---" << std::endl;
			for (size_t c : sortedComponentIds[ap])
				std::cout << "   " << c << std::endl;
			*/

			return shareComponents;
		},
			[&ap, &component, &componentIds](const EdgeDescriptor& e)
		{ 
			return componentIds[ap].find(component[e]) != componentIds[ap].end(); 
		});
		
		res.push_back(pGraph);
	}

	return res;
}

bool shouldBePruned(const lmu::Graph& g, const lmu::EdgeDescriptor& ed)
{
	// Get the two vertices that are joined by this edge...
	auto u = boost::source(ed, g.structure);
	auto v = boost::target(ed, g.structure);

	return boost::in_degree(u, g.structure) == 1 || boost::in_degree(v, g.structure) == 1;
}

lmu::Graph lmu::pruneGraph(const Graph& g)
{
	//lmu::Graph res = g;

	//Get pruned property map 
	//boost::property_map <GraphStructure, boost::edge_pruned_t>::type
	//	pruned = boost::get(boost::edge_pruned_t(), res.structure);

	//boost::graph_traits <GraphStructure>::edge_iterator ei, ei_end;
	//for (boost::tie(ei, ei_end) = boost::edges(res.structure); ei != ei_end; ++ei)
	//{
		//pruned[*ei] = shouldBePruned(res, *ei);
	//}

	auto g1 = lmu::filterGraph(g, [](const VertexDescriptor& v) {return true; }, [&g](const EdgeDescriptor& e) { return !shouldBePruned(g, e); });

	auto g2 = lmu::filterGraph(g1, [&g1](const VertexDescriptor& v) {return boost::in_degree(v, g1.structure) > 0;  }, [](const EdgeDescriptor& e) { return true; });
	
	return g2;
}

bool shouldBePruned(lmu::VertexDescriptor v, const lmu::Graph& g, const std::unordered_map<lmu::VertexDescriptor, std::unordered_set<lmu::VertexDescriptor>>& neighborMap)
{
	boost::graph_traits<lmu::GraphStructure>::adjacency_iterator neighbour, neighbour_end;
	const auto& neighbors = neighborMap.at(v);

	for (boost::tie(neighbour, neighbour_end) = boost::adjacent_vertices(v, g.structure); neighbour != neighbour_end; ++neighbour)
	{
		const auto& neighborNeighbors = neighborMap.at(*neighbour);

		bool fullyContained = true;
		for (const auto& n : neighbors)
		{
			if (neighborNeighbors.count(n) == 0)
			{
				fullyContained = false;
				break;
			}
		}

		if (fullyContained)
			return true;
	}

	return false;
}

lmu::NeighborMap lmu::createNeighborMap(const lmu::Graph& g)
{
	boost::graph_traits<GraphStructure>::vertex_iterator vi, vi_end;

	lmu::NeighborMap neighborMap;
	for (boost::tie(vi, vi_end) = boost::vertices(g.structure); vi != vi_end; ++vi)
	{
		std::unordered_set<VertexDescriptor> neighbors;
		boost::graph_traits<GraphStructure>::adjacency_iterator neighbour, neighbour_end;

		//Get neighbor set.
		for (boost::tie(neighbour, neighbour_end) = boost::adjacent_vertices(*vi, g.structure); neighbour != neighbour_end; ++neighbour)
			neighbors.insert(*neighbour);

		neighborMap[*vi] = neighbors;
	}

	return neighborMap;
}

lmu::PruneList lmu::createPruneList(const lmu::Graph& g, const lmu::NeighborMap& neighborMap)
{
	boost::graph_traits<GraphStructure>::vertex_iterator vi, vi_end;
	PruneList pruneList;
	for (boost::tie(vi, vi_end) = boost::vertices(g.structure); vi != vi_end; ++vi)
	{
		boost::graph_traits<GraphStructure>::adjacency_iterator neighbour, neighbour_end;

		for (boost::tie(neighbour, neighbour_end) = boost::adjacent_vertices(*vi, g.structure); neighbour != neighbour_end; ++neighbour)
		{
			const auto& neighborNeighbors = neighborMap.at(*neighbour);

			//are all neighbors of the neighbor also neighbor of *vi?
			bool fullyContained = true;
			for (const auto& nn : neighborNeighbors)
			{
				if (neighborMap.at(*vi).count(nn) == 0 && nn != *vi)
				{
					fullyContained = false;
					break;
				}
			}

			if (fullyContained)
			{
				pruneList[*vi].push_back(*neighbour);
			}
		}
	}

	return pruneList;
}

bool inPruneList(const lmu::VertexDescriptor& v, const lmu::PruneList& pruneList)
{
	for (const auto& list : pruneList)
	{
		if (std::find(list.second.begin(), list.second.end(), v) != list.second.end())
		{
			return true;
		}
	}
	
	return false;
}

lmu::Graph lmu::pruneGraph(const lmu::Graph& g, const lmu::PruneList & pruneList)
{
	return lmu::filterGraph(g, [&pruneList](const VertexDescriptor& v) {return !inPruneList(v, pruneList); }, [](const EdgeDescriptor& e) { return true; });
}

lmu::Graph lmu::recreatePrunedGraph(const lmu::Graph& originalGraph, const lmu::Graph& prunedGraph, const lmu::PruneList& pruneList)
{	
	
	if (numVertices(prunedGraph) > 1)
	{
		std::cerr << "Too many vertices. " << std::endl;
		return lmu::Graph();
	}

	boost::graph_traits<GraphStructure>::vertex_iterator vi, vi_end;
	for (boost::tie(vi, vi_end) = boost::vertices(prunedGraph.structure); vi != vi_end; ++vi)
	{
		auto func = prunedGraph.structure[*vi];
		auto v = originalGraph.vertexLookup.at(func);

		std::cout << " Partition " << func->name() << std::endl;
		auto it = pruneList.find(v);
		if (it == pruneList.end())
		{
			std::cout << "Has no pruned nodes." << std::endl;
			break;
		}
		auto prunedVertices = it->second;

		for (const auto& pv : prunedVertices)
			std::cout << "  " << originalGraph.structure[pv]->name() << std::endl;

		return lmu::filterGraph(originalGraph,
			[&prunedVertices, v](const VertexDescriptor& vp)
		{			
			return std::find(prunedVertices.begin(), prunedVertices.end(), vp) != prunedVertices.end() || vp == v;
		}, 
			[&prunedVertices, &originalGraph](const EdgeDescriptor& e)
		{ 
			return true;
				//std::find(prunedVertices.begin(), prunedVertices.end(), boost::source(e, originalGraph.structure)) != prunedVertices.end() &&
				//std::find(prunedVertices.begin(), prunedVertices.end(), boost::target(e, originalGraph.structure)) != prunedVertices.end();
		}
		);
	}

	return prunedGraph;
}

size_t lmu::numVertices(const lmu::Graph& g)
{
	return boost::num_vertices(g.structure);
}

size_t lmu::numEdges(const lmu::Graph& g)
{
	return boost::num_edges(g.structure);
}

lmu::Graph lmu::getGraphWithPrunedVertices(const lmu::Graph& g, const std::shared_ptr<lmu::ImplicitFunction>& f)
{	
	auto v = g.vertexLookup.at(f);

	typename boost::graph_traits <lmu::GraphStructure>::out_edge_iterator ei, ei_end;
		
	//Iterate through all neighbor vertices, check if they were pruned and add them to new graph.
	lmu::Graph res;
	auto fVertex = addVertex(res, f);
	for (boost::tie(ei, ei_end) = boost::out_edges(v, g.structure); ei != ei_end; ++ei)
	{
		if (shouldBePruned(g, *ei))
		{
			auto neighborVertex = boost::target(*ei, g.structure);
			auto neighborFunc = g.structure[neighborVertex];

			auto neighborFuncVertex = addVertex(res, neighborFunc);
			addEdge(res, fVertex, neighborFuncVertex);
		}
	}

	return res;
}

namespace boost
{
	struct EdgeFilterPredicateObj
	{
		EdgeFilterPredicateObj(const lmu::EdgeFilterPredicate& p) : p(p)
		{
		}

		bool operator()(const lmu::EdgeDescriptor& e)
		{
			return !p(e);
		}

		const lmu::EdgeFilterPredicate& p;
	};

	
}

lmu::Graph lmu::filterGraph(const lmu::Graph& g, const EdgeFilterPredicate& predicate)
{
	auto res = g;

	boost::remove_edge_if(boost::EdgeFilterPredicateObj(predicate), res.structure);

	recreateVertexLookup(res);

	return res;
}

lmu::Graph lmu::filterGraph(const lmu::Graph& g, const VertexFilterPredicate& vp, const EdgeFilterPredicate& ep)
{	
	typedef boost::filtered_graph<
		GraphStructure,
		EdgeFilterPredicate,
		VertexFilterPredicate
	> FilteredView;
	
	lmu::Graph res; 

	boost::copy_graph(FilteredView(g.structure, ep, vp), res.structure);

	recreateVertexLookup(res);

	return res;
}
