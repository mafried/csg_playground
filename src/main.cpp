#define __N
#ifdef __N

#define BOOST_PARAMETER_MAX_ARITY 12

#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>

#include <Eigen/Core>
#include <iostream>
#include <tuple>
#include <chrono>

#include "mesh.h"
#include "ransac.h"
#include "pointcloud.h"
#include "collision.h"
#include "congraph.h"
#include "csgtree.h"
#include "tests.h"

#include "csgnode_evo.h"
#include "csgnode_helper.h"

#include "evolution.h"

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

	//RUN_TEST(CSGNodeTest);


	igl::opengl::glfw::Viewer viewer;

	// Initialize
	update(viewer);
	
	CSGNode node =

		op<Difference>(
		{
		op<Union>(
		{
			op<Union>(
			{
				geo<IFBox>(Eigen::Affine3d::Identity(), Eigen::Vector3d(0.6,0.6,0.6),2, "Box_0"),
				geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, -0.3, 0), 0.3, "Sphere_0")
			}),
			geo<IFCylinder>(Eigen::Affine3d::Identity(), 0.2, 1.0, "Cylinder_0"),
		}),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, 0.7, 0), 0.3, "Sphere_1")
		});

	CSGNode node2 =

		op<Union>(
	{
		geo<IFBox>(Eigen::Affine3d::Identity(), Eigen::Vector3d(0.5,0.5,0.5),2, "Box_0"),
		geo<IFCylinder>(Eigen::Affine3d::Identity(), 0.2, 1.0, "Cylinder_0")
		//geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, -0.2, 0), 0.2, "Sphere_0"),
		//geo<IFSphere>(Eigen::Affine3d::Identity(), 0.2, "Sphere_1")
	});


	//lmu::Mesh csgMesh = computeMesh(node, Eigen::Vector3i(50, 50, 50));
	//viewer.data().set_mesh(csgMesh.vertices, csgMesh.indices);

	//auto error = computeDistanceError(csgMesh.vertices, node, node2, true);
	//viewer.data().set_colors(error);
	
	auto pointCloud = lmu::computePointCloud(node, Eigen::Vector3i(40, 40, 40), 0.01, 0.01);
	
	viewer.data().point_size = 5.0;
	
	std::vector<std::shared_ptr<lmu::ImplicitFunction>> shapes; 
	for (const auto& geo : allGeometryNodePtrs(node))	
		shapes.push_back(geo->function());
	
	lmu::ransacWithSim(pointCloud.leftCols(3), pointCloud.rightCols(3), 0.001, shapes);

	int rows = 0; 
	for (const auto& shape : shapes)
	{
		std::cout << "Shape" << std::endl;

		rows += shape->points().rows();
	}
	
	Eigen::MatrixXd points(rows,6);
	int j = 0;
	int k = 0;

	Eigen::MatrixXd colors(16,3) ;
	colors.row(0) = Eigen::Vector3d(1, 0, 0);
	colors.row(1) = Eigen::Vector3d(0, 1, 0);
	colors.row(2) = Eigen::Vector3d(0, 0, 1);
	colors.row(3) = Eigen::Vector3d(1, 0, 1);
	colors.row(4) = Eigen::Vector3d(1, 1, 0);
	colors.row(5) = Eigen::Vector3d(0, 1, 1);
	colors.row(6) = Eigen::Vector3d(1, 1, 1);
	colors.row(7) = Eigen::Vector3d(0, 0, 0);

	colors.row(8) = Eigen::Vector3d(.5, 0, 0);
	colors.row(9) = Eigen::Vector3d(0, .5, 0);
	colors.row(10) = Eigen::Vector3d(0, 0, .5);
	colors.row(11) = Eigen::Vector3d(.5, 0, .5);
	colors.row(12) = Eigen::Vector3d(.5, .5, 0);
	colors.row(13) = Eigen::Vector3d(0, .5, .5);
	colors.row(14) = Eigen::Vector3d(.5, .5, .5);
	colors.row(15) = Eigen::Vector3d(0, 0, 0);

	
	for( auto& shape : shapes)
	{	
		for (int i = 0; i < shape->points().rows(); ++i)
		{
			auto row = shape->points().row(i);
			points.row(j) = row;
			//points.row(j)[3] = colors.row(k % colors.size())[0];
			//points.row(j)[4] = colors.row(k % colors.size())[1];
			//points.row(j)[5] = colors.row(k % colors.size())[2];

			j++;
		}

		std::cout << "Shape_" << std::to_string(k) << " Color: " << colors.row(k % colors.size()) << std::endl;

		k++;
	}

	 auto graph = lmu::createConnectionGraph(shapes);

	 lmu::writeConnectionGraph("graph.dot", graph);

	 auto cliques = lmu::getCliques(graph);

	 auto cliquesAndNodes = computeNodesForCliques(cliques, graph, ParallelismOptions::PerCliqueParallelism | ParallelismOptions::GAParallelism);
	
	 CSGNode recNode(nullptr);
	 try
	 {
		 recNode = mergeCSGNodeCliqueSimple(cliquesAndNodes);
	 }
	 catch (const std::exception& ex)
	 {
		 std::cout << "Could not merge. Reason: " << ex.what() << std::endl;
	 }

	 writeNode(recNode, "tree.dot");

	 auto treeMesh = computeMesh(recNode, Eigen::Vector3i(50, 50, 50));
	 viewer.data().set_mesh(treeMesh.vertices, treeMesh.indices);
	 	
	//viewer.core. = true;
	viewer.core.background_color = Eigen::Vector4f(0.3, 0.3, 0.3, 1.0);
	//viewer.core.point_size = 5.0;
	viewer.callback_key_down = &key_down;
	viewer.core.camera_dnear = 3.9;
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