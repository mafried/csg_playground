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

	//igl::readOFF(TUTORIAL_SHARED_PATH "/decimated-knight.off", VB, FB);
	// Plot the mesh with pseudocolors

	// Domain (Warning: Sphere_3 constructor uses squared radius !)
	/*Mesh_domain domain(sphere_function,
		K::Sphere_3(CGAL::ORIGIN, 2.));

	// Mesh criteria
	Spherical_sizing_field size;
	Mesh_criteria criteria(facet_angle = 30, facet_size = 0.1, facet_distance = 0.025,
		cell_radius_edge_ratio = 2, cell_size = size);

	// Mesh generation
	mesh = CGAL::make_mesh_3<C3t3>(domain, criteria, no_exude(), no_perturb());

	int a = 5;
	std::cout << a;

	// Output
	std::ofstream file("out.off");
	mesh.output_boundary_to_off(file);
	file.close();
	*/

	igl::opengl::glfw::Viewer viewer;

	// Initialize
	update(viewer);

	//lmu::CSGTreeGA ga;

	Eigen::Affine3d t = Eigen::Affine3d::Identity();
	//t = Eigen::AngleAxisd(20.0, Eigen::Vector3d::UnitZ());//
	t = Eigen::Translation3d(0.1, 0, 0);
	//rotate(Eigen::AngleAxisd(20.0, Eigen::Vector3d::UnitZ()));

	lmu::Mesh mesh1 = lmu::createBox(t, Eigen::Vector3d(0.5, 0.5, 0.5), 4); ;// lmu::createCylinder(t, 0.2, 0.2, 1.2, 10, 10);//
	lmu::Mesh mesh2 = lmu::createCylinder(Eigen::Affine3d::Identity(), 0.2, 0.2, 1.0, 30, 30);
	lmu::Mesh mesh3 = lmu::createSphere(Eigen::Affine3d::Identity(), 0.5, 30, 30);
	lmu::Mesh mesh;
	//lmu::Mesh mesh2 = lmu::createBox(Eigen::Affine3d::Identity(), Eigen::Vector3d(0.5, 0.5, 0.5));

	//igl::copyleft::cork::CSGTree meshTree = { { mesh1.vertices, mesh1.indices },{ mesh3.vertices, mesh3.indices }, igl::MeshBooleanType::MESH_BOOLEAN_TYPE_UNION };

	//igl::copyleft::cgal::mesh_boolean(mesh1.vertices, mesh1.indices, mesh3.vertices, mesh3.indices, igl::MeshBooleanType::MESH_BOOLEAN_TYPE_UNION, mesh.vertices, mesh.indices);

	//lmu::Mesh csgMesh(meshTree.cast_V<MatrixXd>(), meshTree.F());	

	//lmu::Mesh csgMesh(mesh.vertices, mesh.indices, mesh.normals);

	//auto csgMesh = lmu::fromOBJFile("mick.obj");// lmu::Mesh(meshTree.cast_V<MatrixXd>(), meshTree.F());//lmu::fromOBJFile("flower.obj");

	//auto pointCloud = lmu::pointCloudFromMesh(csgMesh, 0.001, 0.05, 0.005); //(csgMesh, 0.001, 0.1, 0.005); <= mick
	//auto pointCloud = lmu::readPointCloud("pt_001.dat");

	//lmu::writePointCloud("pt_001.dat", pointCloud);


	CSGNode node =
	
		op<Union>(
		{
			geo<IFSphere>(Eigen::Affine3d::Identity(), 0.2, "Sphere_0"),
			geo<IFBox>(t, Eigen::Vector3d(0.2,0.2,0.2),2, "Box_0"),
			geo<IFCylinder>(Eigen::Affine3d::Identity(), 0.2, 0.5, "Cylinder_0"),
		});

	CSGNode node2 =

		op<Union>(
	{
		geo<IFSphere>(Eigen::Affine3d::Identity(), 0.2, "Sphere_0"),
		geo<IFSphere>(Eigen::Affine3d::Identity(), 0.2, "Sphere_0")
	});


	lmu::Mesh csgMesh = computeMesh(node, Eigen::Vector3i(100, 10, 10));
	viewer.data().set_mesh(csgMesh.vertices, csgMesh.indices);

	auto error = computeDistanceError(csgMesh.vertices, node, node2, true);

	viewer.data().set_colors(error);
	//auto pointCloud = lmu::computePointCloud(node, Eigen::Vector3i(100, 100, 100), 0.01, 0.01);
	//viewer.data().set_points(pointCloud.leftCols(3), pointCloud.rightCols(3));
	//viewer.data().point_size = 0.01;
	
	//viewer.data().set_points(std::get<0>(data), Eigen::Vector3d(1,1,1));// .leftCols(3), pointCloud.rightCols(3));

	/*for (int i = 0; i < csgMesh.normals.rows(); ++i)
	{
		auto x = csgMesh.normals.row(i).x() < 0.0 ? 0.5 : csgMesh.normals.row(i).x();
		auto y = csgMesh.normals.row(i).y() < 0.0 ? 0.5 : csgMesh.normals.row(i).y();
		auto z = csgMesh.normals.row(i).z() < 0.0 ? 0.5 : csgMesh.normals.row(i).z();
				
		csgMesh.normals.row(i) = Eigen::RowVector3d(x,y,z);		
	}*/

	//viewer.data.set_colors(csgMesh.normals);

	std::vector<std::shared_ptr<lmu::ImplicitFunction>> shapes;// =
	//{
	//	std::make_shared<IFBox>(t, Eigen::Vector3d(0.5,0.5,0.5), "Box_0"),
	//	std::make_shared<IFCylinder>(Eigen::Affine3d::Identity(), 0.2, 1.0, "Cylinder_0")
	//};
	//lmu::ransacWithSim(pointCloud.leftCols(3), pointCloud.rightCols(3), 0.05, shapes);
	

	/*std::vector<std::shared_ptr<lmu::ImplicitFunction>> shapes;
	while (shapes.size() != 5)
	{
		shapes = lmu::ransacWithPCL(pointCloud.leftCols(3), pointCloud.rightCols(3));
	}*/
	
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

	for (const auto& shape : shapes)
	{	
		for (int i = 0; i < shape->points().rows() - 1; ++i)
		{
			auto row = shape->points().row(i);
			points.row(j) = row;
			points.row(j)[3] = colors.row(k % colors.size())[0];
			points.row(j)[4] = colors.row(k % colors.size())[1];
			points.row(j)[5] = colors.row(k % colors.size())[2];

			j++;
		}

		std::cout << "Shape_" << std::to_string(k) << " Color: " << colors.row(k % colors.size()) << std::endl;

		k++;
	}


	//viewer.data.set_points(points.leftCols(3), points.rightCols(3));
	//viewer.data.set_mesh(csgMesh.vertices, csgMesh.indices);

	//auto shapes = lmu::ransac(mesh1.vertices, mesh1.normals);

	//lmu::CSGTreeCreator c(shapes, 0.5, 0.7, 5);
	//c.create(10).write("tree.dot");
	
	
	//std::cout << "COLLIDE: " << lmu::collides(*shapes[0], *shapes[1]) << std::endl;

	//auto graph = lmu::createRandomConnectionGraph(30,0.5);
	 auto graph = lmu::createConnectionGraph(shapes);

	 lmu::writeConnectionGraph("graph.dot", graph);

	 auto cliques = lmu::getCliques(graph);

	 //if (cliques.size() != 3)
	 //{
	 //	 std::cout << "NOT ENOUGH CLIQUES!" << std::endl;
	//	 int i; 
	//	 std::cin >> i;
	 //} 

	//std::cout << "SIZE: " << cliques.size();
	 //int i; 
	 //std::cin >> i;
	

	 //auto tree = lmu::createCSGTreeWithGA(shapes, graph);

	 //auto tree = lmu::createCSGTreeTemplateFromCliques(cliques);

	 //tree.write("tree.dot");

	 //auto cliquesAndNodes = computeNodesForCliques(cliques, graph, ParallelismOptions::PerCliqueParallelism);
	 
	 //int i = 0;
	 //for (auto& can : cliquesAndNodes)
	 //{
	 //	 lmu::writeNode(std::get<1>(can), "tree" + std::to_string(i++) + ".dot");
	 //}
	 //std::cin >> i;
	 //return 0;
	
	 //lmu::CSGNodeCreator creator(shapes);
	 //auto node = lmu::createCSGNodeWithGA(shapes, true, graph); //creator.create(3);
	 
	 //auto node = mergeCSGNodeCliqueSimple(cliquesAndNodes);

	 //lmu::writeNode(node, "tree.dot");

	 /*try
	 {
		 //tree.childs[0].childs[0].childs[0].write("tree.dot");
		 auto treeMesh = node.mesh();
		 viewer.data.set_mesh(treeMesh.vertices, treeMesh.indices);
	 }
	 catch (const std::exception& ex)
	 {
		 std::cout << "Could not create CSG mesh. Reason: " << ex.what() << std::endl;
	 }*/

	 /*lmu::CSGTree tr3;
	 tr3.operation = lmu::OperationType::Union;
	 tr3.functions = { shapes[3], shapes[0] };

	 lmu::CSGTree tr2;
	 tr2.operation = lmu::OperationType::Union;
	 tr2.functions = { shapes[1], shapes[2] };
	 
	 lmu::CSGTree tr1;
	 tr1.operation = lmu::OperationType::Union;
	 tr1.childs = { tr2, tr3 };

	 lmu::CSGTree tr0;
	 tr0.operation = lmu::OperationType::Intersection;
	 tr0.functions = { shapes[0] };
	 tr0.childs = { tr1 };

	 int numPoints = 0;
	 for (const auto& shape : shapes)
		 numPoints += shape->points().rows();
	 double lambda = std::log(numPoints);
	 lmu::CSGTreeRanker ranker(lambda, shapes);
	 std::cout << "Rank: tr0 " << ranker.rank(tr0) << std::endl;
	 std::cout << "Rank: tr1 " << ranker.rank(tr1) << std::endl;
	 */
	 
	// auto tree = lmu::createCSGTreeWithGA(shapes);
	
	 //tree.write("tree_tmp.dot");
	
	//auto tree2 = lmu::CSGTreeCreator(shapes, 0.5, 0.7, 20).create();
	//tree2.write("tree.dot");
	
	//viewer.data.set_points(pointCloud.leftCols(3), pointCloud.rightCols(3));
	
	
	//MatrixXd D(shapes[0]->mesh().vertices.rows() + shapes[0]->points().rows(), shapes[0]->mesh().vertices.cols());
	//D << shapes[0]->mesh().vertices, shapes[0]->points();

	//viewer.data.set_points(csgMesh.vertices, Eigen::Vector3d(1,1,1));
	
	//viewer.data.set_points(shapes[1]->points().leftCols(3), shapes[1]->points().rightCols(3));//Eigen::Vector3d(1, 1, 1));

	
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