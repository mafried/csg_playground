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
#include "collision.h"
#include "congraph.h"
#include "csgtree.h"
#include "tests.h"

#include "csgnode_evo.h"
#include "csgnode_helper.h"

#include "evolution.h"

enum class ApproachType
{
	None = 0,
	BaselineGA, 
	Partition
};

ApproachType approachType = ApproachType::Partition;
ParallelismOptions paraOptions = ParallelismOptions::GAParallelism;
int sampling = 30;//35;
int nodeIdx = 3;

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


	bool interactiveMode = argc == 1;
	if (!interactiveMode)
	{
		if (argc != 5)
		{
			std::cerr << "Not enough arguments: " << argc << std::endl;
			return -1;
		}

		try
		{
			approachType = static_cast<ApproachType>(std::stoi(std::string(argv[1])));
			paraOptions = static_cast<ParallelismOptions>(std::stoi(std::string(argv[2])));
			sampling = std::stoi(std::string(argv[3]));
			nodeIdx = std::stoi(std::string(argv[4]));

			std::cout << "Start in batch mode. Approach Type: " << static_cast<int>(approachType) << " paraOptions: " << static_cast<int>(paraOptions) << " sampling: " << sampling << " nodeIdx: " << nodeIdx << std::endl;
			std::cout << "Per GA Parallelism: " << static_cast<int>((paraOptions & ParallelismOptions::GAParallelism)) << std::endl;
			std::cout << "Per Clique Parallelism: " << static_cast<int>((paraOptions & ParallelismOptions::PerCliqueParallelism)) << std::endl;
		}
		catch (const std::exception& ex)
		{
			std::cerr << "Unable to start app in noninteractive mode. Reason: " << ex.what() << std::endl;
			return -1;
		}
	}

	//RUN_TEST(CSGNodeTest);


	igl::opengl::glfw::Viewer viewer;
	viewer.mouse_mode = igl::opengl::glfw::Viewer::MouseMode::Rotation;

	// Initialize
	update(viewer);

	Eigen::AngleAxisd rot90x(M_PI / 2.0, Vector3d(0.0, 0.0, 1.0));


	CSGNode node(nullptr);

	if (nodeIdx == 0)
	{
		node =
			op<Union>(
		{
			op<Union>(
			{
				op<Union>(
				{
					op<Difference>({
						geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(-0.2, 0, -1)*rot90x), 0.2, 0.8, "Cylinder_2"),
						geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(-0.2, 0, -1)*rot90x), 0.1, 0.8, "Cylinder_3")
					}),

					op<Union>(
					{
						geo<IFBox>((Eigen::Affine3d)Eigen::Translation3d(0, 0, -0.5), Eigen::Vector3d(0.5,1.0,1.0),2, "Box_2"),
						geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(0, 0, -1)*rot90x), 0.5, 0.5, "Cylinder_0")
					})
				})
				,
				op<Union>(
				{
					op<Union>(
					{
						geo<IFBox>((Eigen::Affine3d)Eigen::Translation3d(-0.3, 0, -0.5), Eigen::Vector3d(0.2,0.8,0.9),2, "Box_3"),
						geo<IFBox>((Eigen::Affine3d)Eigen::Translation3d(0.3, 0, -0.5), Eigen::Vector3d(0.2,0.8,1.0),2, "Box_4")
					}),

					op<Union>(
					{
						geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(0.3, 0, -1)*rot90x), 0.4, 0.2, "Cylinder_1"),
						geo<IFBox>(Eigen::Affine3d::Identity(), Eigen::Vector3d(1.0,2.0,0.1),2, "Box_1")
					})
				})

			}),

			op<Union>(
			{
				geo<IFBox>((Eigen::Affine3d)Eigen::Translation3d(0, 0, -0.2), Eigen::Vector3d(0.8,1.8,0.2),2, "Box_0"),
				op<Union>(
				{
					op<Union>(
					{
						op<Difference>(
						{
							geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, 1.0, 0.2), 0.2, "Sphere_0"),
							geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, 1.0, 0.6), 0.4, "Sphere_1")
						}),
						op<Difference>(
						{
							geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, 1.0, 0.2), 0.2, "Sphere_2"),
							geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, 1.0, 0.6), 0.4, "Sphere_3")
						})
					}),
					op<Union>(
					{
						op<Difference>(
						{
							geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, -1.0, 0.2), 0.2, "Sphere_4"),
							geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, -1.0, 0.6), 0.4, "Sphere_5")
						}),
						op<Difference>(
						{
							geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, -1.0, 0.2), 0.2, "Sphere_6"),
							geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, -1.0, 0.6), 0.4, "Sphere_7")
						})
					})
				})
			})
		});
	}
	else if (nodeIdx == 1)
	{
		node =

			op<Difference>(
		{
		op<Union>(
		{
			op<Union>(
			{
				geo<IFBox>(Eigen::Affine3d::Identity(), Eigen::Vector3d(0.6,0.6,0.6),2, "Box_0", 2.0),
				geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, -0.3, 0), 0.3, "Sphere_0")
			}),
			geo<IFCylinder>(Eigen::Affine3d::Identity(), 0.2, 1.0, "Cylinder_0"),
		}),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, 0.7, 0), 0.4, "Sphere_1")
		});
	}
	else if (nodeIdx == 2)
	{
		node =
			op<Union>(
		{
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, 0.0, 0), 0.25, "Sphere_1"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, 0.3, 0), 0.25, "Sphere_2"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, 0.6, 0), 0.25, "Sphere_3"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, 0.9, 0), 0.25, "Sphere_4"),

			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.3, 0.0, 0), 0.25, "Sphere_7"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.3, 0.3, 0), 0.25, "Sphere_8"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.3, 0.6, 0), 0.25, "Sphere_9"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.3, 0.9, 0), 0.25, "Sphere_10"),

			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.6, 0.0, 0), 0.25, "Sphere_13"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.6, 0.3, 0), 0.25, "Sphere_14"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.6, 0.6, 0), 0.25, "Sphere_15"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.6, 0.9, 0), 0.25, "Sphere_16"),

			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.9, 0.0, 0), 0.25, "Sphere_19"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.9, 0.3, 0), 0.25, "Sphere_20"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.9, 0.6, 0), 0.25, "Sphere_21"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.9, 0.9, 0), 0.25, "Sphere_22"),

			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(1.2, 0.0, 0), 0.25, "Sphere_25"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(1.2, 0.3, 0), 0.25, "Sphere_26"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(1.2, 0.6, 0), 0.25, "Sphere_27"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(1.2, 0.9, 0), 0.25, "Sphere_28"),
		});
	}
	else if (nodeIdx == 3)
	{
		node = geometry(std::make_shared<IFNull>(""));
	}
	else
	{
		std::cerr << "Could not get node. Idx: " << nodeIdx << std::endl;
		return -1;
	}

	//lmu::Mesh csgMesh = computeMesh(node, Eigen::Vector3i(50, 50, 50));
	//viewer.data().set_mesh(csgMesh.vertices, csgMesh.indices);

	//auto error = computeDistanceError(csgMesh.vertices, node, node2, true);
	//viewer.data().set_colors(error);

	//high: lmu::computePointCloud(node, Eigen::Vector3i(120, 120, 120), 0.05, 0.01);
	//medium: lmu::computePointCloud(node, Eigen::Vector3i(75, 75, 75), 0.05, 0.01);
	//low: lmu::computePointCloud(node, Eigen::Vector3i(50, 50, 50), 0.05, 0.01);

	Eigen::MatrixXd pointCloud;
	std::vector<std::shared_ptr<lmu::ImplicitFunction>> shapes;
	double ransacShapeDist = 0.0;

	if (nodeIdx == 3)
	{
		float scaling = 0.1;
		pointCloud = readPointCloud("fayolle_data/body.xyz", scaling);
		shapes = fromFile("fayolle_data/body.fit", scaling);
		ransacShapeDist = scaling * 0.5;
	}
	else
	{
		pointCloud = lmu::computePointCloud(node, Eigen::Vector3i(sampling, sampling, sampling), 0.05, 0.01);
		for (const auto& geo : allGeometryNodePtrs(node))
			shapes.push_back(geo->function());
		ransacShapeDist = 0.05;
	}

	writeNode(node, "tree.dot");


	lmu::ransacWithSim(pointCloud.leftCols(3), pointCloud.rightCols(3), ransacShapeDist, shapes);

	int rows = 0;
	for (const auto& shape : shapes)
	{
		std::cout << "Shape" << std::endl;

		rows += shape->points().rows();
	}

	Eigen::MatrixXd points(rows, 6);
	int j = 0;
	int k = 0;

	Eigen::MatrixXd colors(16, 3);
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


	for (auto& shape : shapes)
	{
		for (int i = 0; i < shape->points().rows(); ++i)
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

	//viewer.data().set_points(points.leftCols(3), points.rightCols(3));
	//viewer.core.background_color = Eigen::Vector4f(1, 1, 1, 1);
	//viewer.data().point_size = 2.0;

	//if (interactiveMode)
	//	viewer.launch();
	
	//std::cout << "PointCloud: " << pointCloud.rows() << " Points: " << points.rows() << std::endl;
	std::ofstream f("pipeline_info.dat");
	f << "Approach Type: " << static_cast<int>(approachType) << std::endl;
	f << "Point cloud size: " << pointCloud.rows() << std::endl;

	const double alpha = M_PI / 18.0;
	const double epsilon = 0.01;

	f << "Input CSG tree: size: " << numNodes(node) << " depth: " << depth(node) << " geometry score: " << computeGeometryScore(node, epsilon, alpha, shapes) << std::endl;

	TimeTicker ticker;

	auto graph = lmu::createConnectionGraph(shapes);
	
	auto conGraphDur = ticker.tick();
	f << "Connection graph creation: duration: " << conGraphDur << std::endl;

	lmu::writeConnectionGraph("graph.dot", graph);

	ticker.tick();
	auto cliques = lmu::getCliques(graph);
	auto cliqueDur = ticker.tick();
	f << "Clique enumeration: #cliques: " << cliques.size() << " duration: " << cliqueDur << std::endl;

	CSGNode recNode(nullptr);
	try
	{
		ticker.tick();
		
		switch (approachType)
		{
			case ApproachType::BaselineGA:
				recNode = createCSGNodeWithGA(shapes, (paraOptions & ParallelismOptions::GAParallelism) == ParallelismOptions::GAParallelism, graph);
				f << "Full GA: duration: " << ticker.tick() << std::endl;

				break;

			case ApproachType::Partition:
				{
					auto cliquesAndNodes = computeNodesForCliques(cliques, paraOptions);
					optimizeCSGNodeClique(cliquesAndNodes, 100.0);

					auto cliqueCompDur = ticker.tick();
					f << "Per clique node computation: duration: " << cliqueCompDur;

					
					recNode = mergeCSGNodeCliqueSimple(cliquesAndNodes);

					auto mergeDur = ticker.tick();
					f <<  "Clique Merge: duration: " << mergeDur << std::endl;

					f << "Full Partition: duration: " << (conGraphDur + cliqueDur + cliqueCompDur + mergeDur) << std::endl;
				}
				break;
			default: 
				recNode = node;
				break;
		}
	}
	catch (const std::exception& ex)
	{
		std::cout << "Could not merge. Reason: " << ex.what() << std::endl;
		int i;
		std::cin >> i;
		return -1;
	}

	f << "Output CSG tree: size: " << numNodes(recNode) << " depth: " << depth(recNode) << " geometry score: " << computeGeometryScore(recNode, epsilon, alpha, shapes) << std::endl;


	f.close();

	writeNode(recNode, "tree.dot");

	auto treeMesh = computeMesh(recNode, Eigen::Vector3i(150, 150, 150));
	igl::writeOBJ("tree_mesh.obj", treeMesh.vertices, treeMesh.indices);

	viewer.data().set_mesh(treeMesh.vertices, treeMesh.indices);
	
	//viewer.core. = true;
	viewer.core.background_color = Eigen::Vector4f(1,1,1,1);
	viewer.data().point_size = 2.0;
	viewer.callback_key_down = &key_down;
	viewer.core.camera_dnear = 0.1;
	viewer.core.lighting_factor = 0;

	if(interactiveMode)
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