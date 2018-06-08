#define __N
#ifdef __N

#define BOOST_PARAMETER_MAX_ARITY 12

#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>

#include <Eigen/Core>
#include <iostream>
#include <tuple>
#include <chrono>

#include "mesh.h"
#include "ransac.h"
#include "pointcloud.h"
#include "congraph.h"
#include "tests.h"


void update(igl::opengl::glfw::Viewer& viewer)
{
}

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int mods)
{
	switch (key)
	{
	default:
		return false;
	case '-':
		viewer.core.camera_dnear -= 0.1;
		return true;
	case '+':
		viewer.core.camera_dnear += 0.1;
		return true;
	}
	update(viewer);
	return true;
}

int main(int argc, char *argv[])
{
	using namespace Eigen;
	using namespace std;


	igl::opengl::glfw::Viewer viewer;
	viewer.mouse_mode = igl::opengl::glfw::Viewer::MouseMode::Rotation;

	// Initialize
	update(viewer);

	int a = 0; 

	auto points = lmu::readPointCloudWithColors("2_pred.obj");


	//std::cout << "TASRWER";
	//std::cout << points;

	viewer.data().set_points(points.leftCols(3), points.rightCols(3));
	
	//viewer.core. = true;
	viewer.core.background_color = Eigen::Vector4f(1,1,1,1);

	viewer.data().point_size = 10.0;
	viewer.callback_key_down = &key_down;
	viewer.core.camera_dnear = 0.1;
	viewer.core.lighting_factor = 0;
	
	viewer.launch();
}

#else 

// (C) Copyright Andrew Sutton 2007
//
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0 (See accompanying file
// LICENSE_1_0.txt or http://www.boost.org/LICENSE_1_0.txt)

//[code_bron_kerbosch_print_cliques
#include <iostream>

#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/bron_kerbosch_all_cliques.hpp>

using namespace std;
using namespace boost;

// The clique_printer is a visitor that will print the vertices that comprise
// a clique. Note that the vertices are not given in any specific order.
template <typename OutputStream>
struct clique_printer
{
	clique_printer(OutputStream& stream)
		: os(stream)
	{ }

	template <typename Clique, typename Graph>
	void clique(const Clique& c, const Graph& g)
	{
		std::cout << "Clique: ";
		// Iterate over the clique and print each vertex within it.
		typename Clique::const_iterator i, end = c.end();
		for (i = c.begin(); i != end; ++i) {
			os << g[*i].name << " ";
		}
		os << endl;
	}
	OutputStream& os;
};

// The Actor type stores the name of each vertex in the graph.
struct Actor
{
	string name;
};

// Declare the graph type and its vertex and edge types.
typedef undirected_graph<Actor> Graph;
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef graph_traits<Graph>::edge_descriptor Edge;

// The name map provides an abstract accessor for the names of
// each vertex. This is used during graph creation.
typedef property_map<Graph, string Actor::*>::type NameMap;

template <typename Graph, typename NameMap, typename VertexMap>
typename boost::graph_traits<Graph>::vertex_descriptor
add_named_vertex(Graph& g, NameMap nm, const std::string& name, VertexMap& vm)
{
	typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;
	typedef typename VertexMap::iterator Iterator;

	Vertex v;
	Iterator iter;
	bool inserted;
	boost::tie(iter, inserted) = vm.insert(make_pair(name, Vertex()));
	if (inserted) {
		// The name was unique so we need to add a vertex to the graph
		v = add_vertex(g);
		iter->second = v;
		put(nm, v, name);      // store the name in the name map
	}
	else {
		// We had alread inserted this name so we can return the
		// associated vertex.
		v = iter->second;
	}
	return v;
}

template <typename Graph, typename NameMap, typename InputStream>
inline std::map<std::string, typename boost::graph_traits<Graph>::vertex_descriptor>
read_graph(Graph& g, NameMap nm, InputStream& is)
{
	typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;
	std::map<std::string, Vertex> verts;
	for (std::string line; std::getline(is, line); ) {
		if (line.empty()) break;
		std::size_t index = line.find_first_of(',');
		std::string first(line, 0, index);
		std::string second(line, index + 1);

		Vertex u = add_named_vertex(g, nm, first, verts);
		Vertex v = add_named_vertex(g, nm, second, verts);
		std::cout << "edge between " << first << " " << second << std::endl;
		add_edge(u, v, g);
	}
	return verts;
}

int
main(int argc, char *argv[])
{
	// Create the graph and and its name map accessor.
	Graph g;
	NameMap nm(get(&Actor::name, g));

	// Read the graph from standard input.
	//read_graph(g, nm, cin);

	// Instantiate the visitor for printing cliques
	clique_printer<ostream> vis(cout);

	std::map<std::string, Vertex> verts;

	Vertex v0 = add_named_vertex(g, nm, "0", verts);
	Vertex v1 = add_named_vertex(g, nm, "1", verts);
	Vertex v2 = add_named_vertex(g, nm, "2", verts);
	Vertex v3 = add_named_vertex(g, nm, "3", verts);
	Vertex v4 = add_named_vertex(g, nm, "4", verts);
	//Vertex v5 = add_named_vertex(g, nm, "5", verts);
	//Vertex v6 = add_named_vertex(g, nm, "6", verts);

	add_edge(v0, v1, g);
	add_edge(v1, v2, g);
	add_edge(v2, v3, g);
	add_edge(v3, v0, g);
	add_edge(v3, v1, g);
	add_edge(v2, v0, g);

	add_edge(v4, v0, g);
	add_edge(v4, v3, g);
	add_edge(v4, v1, g);


	// Use the Bron-Kerbosch algorithm to find all cliques, printing them
	// as they are found.
	bron_kerbosch_all_cliques(g, vis);

	int i = 0;
	std::cin >> i;

	return 0;
}
//]

#endif