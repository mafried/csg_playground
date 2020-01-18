#ifndef OPTIMIZER_TEST_H
#define OPTIMIZER_TEST_H

#include "test.h"

#include "csgnode_helper.h"
#include "optimizer_red.h"
#include "optimizer_ga.h"
#include "optimizer_clustering.h"
#include "optimizer_py.h"
#include "red_inserter.h"
#include "cit.h"

// To verify CSG expressions
#include <igl/writeOBJ.h>
#include <pointcloud.h>

const std::string py_module_path = "C:/Projekte/dnf_opt/dnf_opt";

using namespace lmu;

// Helpers for defining a simple csg expression
CSGNode sphere(double x, double y, double z, double r, const std::string& name = "")
{
	return geo<IFSphere>((Eigen::Affine3d)(Eigen::Translation3d(x, y, z)), r, name);
}


// Helpers for defining a simple csg expression
CSGNode cys(double radius, double height, double d, double a, double dz, const std::string& name = "")
{
	Eigen::Affine3d tz = (Eigen::Affine3d)Eigen::Translation3d(0, 0, dz);

	Eigen::Affine3d td = (Eigen::Affine3d)Eigen::Translation3d(d, 0, 0);
	Eigen::Affine3d ra = (Eigen::Affine3d)Eigen::AngleAxisd(a*M_PI / 180.0, Eigen::Vector3d::UnitZ());
	Eigen::Affine3d rot90x = (Eigen::Affine3d)Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitX());
	Eigen::Affine3d t = tz * ra * td * rot90x;
	return geo<IFCylinder>(t, radius, height, name);
}

CSGNode cube1()
{
	const std::string name = "cube1";
	Eigen::Affine3d t = (Eigen::Affine3d)Eigen::Translation3d(0, 0, -1.5);
	Eigen::Vector3d size(10, 10, 3);
	return geo<IFBox>(t, size, 1, name);
}

CSGNode cys1()
{
	return cys(3, 3, 0, 0, 0, "cyl1");
}

CSGNode cys2()
{
	return cys(2.5, 1, 0, 0, 1.75, "cyl2");
}

CSGNode cys3()
{
	return cys(3, 0.5, 0, 0, 2.25, "cyl3");
}

CSGNode cys4()
{
	return cys(0.5, 10, 4.5, 30, 0, "cyl4");
}

CSGNode cys5()
{
	return cys(0.5, 10, 4.5, 120, 0, "cyl5");
}

CSGNode cys6()
{
	return cys(0.5, 10, 4.5, 210, 0, "cyl6");
}

CSGNode cys7()
{
	return cys(0.5, 10, 4.5, 300, 0, "cyl7");
}

CSGNode cys8()
{
	return cys(1.5, 10, 6.5, 45, 0, "cyl8");
}

CSGNode cys9()
{
	return cys(1.5, 10, 6.5, 135, 0, "cyl9");
}

CSGNode cys10()
{
	return cys(1.5, 10, 6.5, 225, 0, "cyl10");
}

CSGNode cys11()
{
	return cys(1.5, 10, 6.5, 315, 0, "cyl11");
}

CSGNode cys12()
{
	return cys(1.5, 3, 0, 0, 3, "cyl12");
}

CSGNode cys13()
{
	return cys(1, 6, 0, 0, 4, "cyl13");
}

CSGNode create_obj_0()
{
	return opDiff({
		opDiff({
		opDiff({
		opDiff({
		opDiff({
		opDiff({
		opDiff({
		opDiff({
		opDiff({
		opDiff({
		opUnion({
		opUnion({ opUnion({ cube1(), cys1() }), cys2() }),
		cys3() }),
		cys4() }),
		cys5() }),
		cys6() }),
		cys7() }),
		cys8() }),
		cys9() }),
		cys10() }),
		cys11() }),
		cys12() }),
		cys13() });
}


OptimizerGAParams get_std_ga_params()
{
	OptimizerGAParams params;

	params.ranker_params.geo_score_weight = 20.0;
	params.ranker_params.size_score_weight = 1.0;
	params.ranker_params.prox_score_weight = 2.0;

	params.ranker_params.gradient_step_size = 0.0001;
	params.ranker_params.position_tolerance = 0.1;
	params.ranker_params.sampling_params.errorSigma = 0.00000001;
	params.ranker_params.sampling_params.samplingStepSize = 0.1;
	params.ranker_params.sampling_params.maxDistance = 0.1;
	params.ranker_params.max_sampling_points = 250;

	params.creator_params.create_new_prob = 0.3;
	params.creator_params.subtree_prob = 0.3;
	params.creator_params.initial_population_dist = { 0.1,0.8,0.1 };

	params.man_params.max_delta = 0.2;

	params.ga_params.crossover_rate = 0.4;
	params.ga_params.mutation_rate = 0.3;
	params.ga_params.in_parallel = true;
	params.ga_params.max_iterations = 50;
	params.ga_params.num_best_parents = 2;
	params.ga_params.population_size = 100;
	params.ga_params.tournament_k = 2;
	params.ga_params.use_caching = true;

	return params;
}

TEST(JSONTest)
{
	auto node = create_obj_0();

	toJSONFile(node, "test_node.json");

}

TEST(OptimizerRedundancyTest)
{	
	//s1 does overlap with s2, s3 does neither overlap with s1 nor with s2.
	auto s1 = sphere(0, 0, 0, 1);
	auto s2 = sphere(1, 0, 0, 1, "s2");
	auto s3 = sphere(3, 0, 0, 1);

	const double sampling = 0.1;
	EmptySetLookup esl;

	ASSERT_TRUE(is_empty_set(opInter({ s1, s3 }), sampling, esl));
	ASSERT_TRUE(!is_empty_set(opInter({ s1, s2 }), sampling, esl));
	
	auto node_with_redun = opUnion({ s2, opInter({ s1, s3 }) });
	auto node_without_redun = remove_redundancies(node_with_redun, sampling);
	ASSERT_TRUE(
		numNodes(node_without_redun) == 1,
		node_without_redun.type() == CSGNodeType::Geometry,
		node_without_redun.name() == "s2"
	);

	ASSERT_TRUE(numNodes(remove_redundancies(opUnion({}), sampling)) == 1);
	ASSERT_TRUE(numNodes(remove_redundancies(opUnion({s1}), sampling)) == 2);
	ASSERT_TRUE(numNodes(remove_redundancies(opInter({}), sampling)) == 1);
	ASSERT_TRUE(numNodes(remove_redundancies(opInter({s1}), sampling)) == 2);
	ASSERT_TRUE(numNodes(remove_redundancies(opDiff({}), sampling)) == 1);
	ASSERT_TRUE(numNodes(remove_redundancies(opDiff({ s1 }), sampling)) == 2);
	ASSERT_TRUE(numNodes(remove_redundancies(opComp({}), sampling)) == 1);

	//=========================

	auto node = create_obj_0();
	auto inflated_node = inflate_node(node, 10, { inserter(InserterType::SubtreeCopy, 1.0) });
	auto red_opt_node = remove_redundancies(inflated_node, sampling);
	
	std::cout << "Node: " << numNodes(node) << std::endl;
	std::cout << "Inflated Node: " << numNodes(inflated_node) << std::endl;
	std::cout << "Optimized Node: " << numNodes(red_opt_node) << std::endl;

	ASSERT_TRUE(numNodes(red_opt_node) < numNodes(inflated_node));
	ASSERT_TRUE(
		is_empty_set(opDiff({ node, red_opt_node }), sampling, esl) &&
		is_empty_set(opDiff({ red_opt_node, node }), sampling, esl)
	);

	//=========================

	node = opUnion({ cys1(), opDiff({opDiff({opUnion({cys1(), opDiff({opUnion({cys1(), opDiff({opUnion({opInter({cube1(), cys7()}),opDiff({opUnion({opInter({cube1(), opComp({cys11()})}), cys3()}), cys10()})}), cys4()})}), cys6()})}),cys5() }),cys8() }) });
	red_opt_node = remove_redundancies(node, sampling);

	std::cout << "Node: " << numNodes(node) << std::endl;
	std::cout << "Optimized Node: " << numNodes(red_opt_node) << std::endl;
	
	//auto mesh = lmu::computeMesh(node, Eigen::Vector3i(200, 200, 200));
	//igl::writeOBJ("csg_mesh.obj", mesh.vertices, mesh.indices);
	
	//mesh = lmu::computeMesh(red_opt_node, Eigen::Vector3i(200, 200, 200));
	//igl::writeOBJ("csg_mesh_red.obj", mesh.vertices, mesh.indices);

}

TEST(OptimizerPISetTest)
{
	auto s1 = sphere(0, 0, 0, 1, "s1");
	auto s2 = sphere(1, 0, 0, 1, "s2");
	auto s3 = sphere(0.5, 1, 0, 1, "s3");
	auto s4 = sphere(0.5, -1, 0, 1, "s4");
	auto s5 = sphere(2.5, 0, 0, 1, "s5");
	auto s6 = sphere(0, 0, 0, 0.05, "s6");

	CITSets sets = generate_cit_sets(opUnion({ s6, opUnion({opDiff({ opUnion({ s1, s2 }), opUnion({ s3, s4 }) }), s5 }) }), 0.02);

	std::cout << sets;
}

TEST(OptimizerGA)
{
	auto s1 = sphere(0, 0, 0, 1, "s1");
	auto s2 = sphere(1, 0, 0, 1, "s2");
	auto s3 = sphere(0.5, 1, 0, 1, "s3");
	auto s4 = sphere(0.5, -1, 0, 1, "s4");
	auto s5 = sphere(2.5, 0, 0, 1, "s5");
	auto s6 = sphere(0, 0, 0, 0.2, "s6");

	auto node = opUnion({s6, opUnion({ opDiff({ opInter({opUnion({ s1, s2 }),opUnion({ s1, s2 })}), opUnion({ s3, s4 }) }), s5 }) });

	OptimizerGAParams params = get_std_ga_params(); 
		
	//auto opt_node_ga = optimize_with_ga(node, params, std::cout).node;
	
	auto opt_node_rr = remove_redundancies(node, params.ranker_params.sampling_params.samplingStepSize);
	
	PythonInterpreter interpreter(py_module_path);

	auto opt_node_sc = optimize_pi_set_cover(opt_node_rr, params.ranker_params.sampling_params.samplingStepSize, interpreter);
	
	std::cout << "Node: " << numNodes(node) << " red: " << numNodes(opt_node_rr) << " sc: " << numNodes(opt_node_sc) << std::endl;

	writeNode(node, "n.gv");
	//writeNode(opt_node_ga, "opt_ga.gv");
	writeNode(opt_node_rr, "opt_rr.gv");
	writeNode(opt_node_sc, "opt_sc.gv");
}

TEST(Cluster_Optimizer)
{
	auto s1 = sphere(0, 0, 0, 1, "s1");
	auto s2 = sphere(1, 0, 0, 1, "s2");
	auto s3 = sphere(0.5, 1, 0, 1, "s3");
	auto s4 = sphere(0.5, -1, 0, 1, "s4");
	auto s5 = sphere(2.5, 0, 0, 1, "s5");

	auto node = opUnion({ opDiff({ opInter({ opUnion({ s1, s2 }),opUnion({ s1, s2 }) }), opUnion({ s3, s4 }) }), s5 });

	const double sampling_grid_size = 0.1;

	PythonInterpreter interpreter(py_module_path);

	auto opt_node = apply_per_cluster_optimization
	(
		cluster_union_paths(node), 

		[sampling_grid_size, &interpreter](const CSGNode& n) { return optimize_pi_set_cover(n, sampling_grid_size, interpreter); },
		/*[](const CSGNode& n) { return optimize_with_ga(n, get_std_ga_params(), std::cout).node; },*/

		union_merge
	);

	auto red_opt_node = remove_redundancies(opt_node, sampling_grid_size);
	
	writeNode(red_opt_node, "red_opt_cluster.gv");
}

TEST(Primitive_Cluster_Optimizer)
{
	auto s1 = sphere(0, 0, 0, 1, "s1");
	auto s2 = sphere(1, 0, 0, 1, "s2");
	auto s3 = sphere(0.5, 1, 0, 1, "s3");
	auto s4 = sphere(0.5, -1, 0, 1, "s4");
	auto s5 = sphere(2.5, 0, 0, 1, "s5");

	auto node = opUnion({ opDiff({ opInter({ opUnion({ s1, s2 }),opUnion({ s1, s2 }) }), opUnion({ s3, s4 }) }), s5 });

	const double sampling_grid_size = 0.1;

	auto dom_prims = find_dominating_prims(node, sampling_grid_size);
	for (const auto& dp : dom_prims)
		std::cout << "DP: " << dp->name() << std::endl;

	auto opt_node = apply_per_cluster_optimization
	(
		cluster_with_dominating_prims(node, dom_prims),

		/*[sampling_grid_size, &node](const PrimitiveCluster& c) { return optimize_pi_set_cover(node, sampling_grid_size, c); },*/
		[&node](const PrimitiveCluster& c) { return optimize_with_ga(node, get_std_ga_params(), std::cout, c).node; },

		union_merge
	);

	// TODO: GA: count only those points for ranking that are close to one of the primitives. 

	auto red_opt_node = remove_redundancies(opt_node, sampling_grid_size);

	writeNode(red_opt_node, "red_opt_cluster.gv");
}


TEST(Proximity_Score)
{
	auto s1 = sphere(0, 0, 0, 1, "s1");
	auto s2 = sphere(1, 0, 0, 1, "s2");
	auto s3 = sphere(3, 0, 0, 1, "s3");

	const double sampling_grid_size = 0.1;

	ASSERT_TRUE(compute_local_proximity_score(opUnion({ s1, s2 }), sampling_grid_size) == 1.0);
	ASSERT_TRUE(compute_local_proximity_score(opUnion({ s1, s3 }), sampling_grid_size) == 0.0);
}

TEST(Python_Parser)
{
	std::string str = "Or(Symbol('s5'), And(Symbol('s1'), Not(Symbol('s3')), Not(Symbol('s4'))), And(Not(Symbol('s3')), Not(Symbol('s4')), Symbol('s2')))";

	auto res_tokenize = tokenize_py_string(str);
	for (const auto t : res_tokenize.tokens)
	{
		std::cout << (int)t.type << " " << t.value << std::endl;
	}

	try
	{
		auto node = parse_py_string(str, { std::make_shared<IFNull>("s1"), std::make_shared<IFNull>("s2"), std::make_shared<IFNull>("s3") , std::make_shared<IFNull>("s4") ,
			std::make_shared<IFNull>("s5") });

		PythonInterpreter interpreter(py_module_path);

		auto opt_node = optimize_with_python(node, SimplifierMethod::SIMPY_TO_DNF, interpreter);

		std::cout << espresso_expression(node) << std::endl;

		std::cout << espresso_expression(opt_node) << std::endl;

		writeNode(node, "parsed_node.gv");
	}
	catch (const CSGNodeParseException& ex)
	{
		std::cout << ex.msg << " token pos: " << ex.error_pos << std::endl;
	}
}

TEST(RedInserter)
{
	auto s1 = sphere(0, 0, 0, 1, "s1");
	auto s2 = sphere(1, 0, 0, 1, "s2");
	auto s3 = sphere(0.5, 1, 0, 1, "s3");
	auto s4 = sphere(0.5, -1, 0, 1, "s4");
	auto s5 = sphere(2.5, 0, 0, 1, "s5");

	auto node = opUnion({ opDiff({ opUnion({ s1, s2 }), opUnion({ s3, s4 }) }), s5 });

	auto inflated_node = inflate_node(node, 10, { inserter(InserterType::SubtreeCopy, 1.0) });

	writeNode(inflated_node, "inflated_node.gv");

	const double sampling = 0.01;
	EmptySetLookup esl;

	ASSERT_TRUE(is_empty_set(opDiff({ node, inflated_node }), sampling, esl) && is_empty_set(opDiff({ inflated_node, node }), sampling, esl));
}

TEST(DominantPrimDecomposer)
{
	auto s1 = sphere(0, 0, 0, 1, "s1");
	auto s2 = sphere(1, 0, 0, 1, "s2");
	auto s3 = sphere(0.5, 1, 0, 1, "s3");
	auto s4 = sphere(0.5, -1, 0, 1, "s4");
	auto s5 = sphere(2.5, 0, 0, 1, "s5");

	auto node_0 = s1;
	auto node_1 = opUnion({ s1, s2 });
	auto node_2 = opDiff({ s1, s2 });
	auto node_3 = opUnion({ opDiff({ opUnion({ s1, s2 }), opUnion({ s3, s4 }) }), s5 });
	
	const double sampling = 0.01;
	const bool use_diff_op = true;

	auto res_0 = dom_prim_decomposition(node_0, sampling, use_diff_op);
	auto res_1 = dom_prim_decomposition(node_1, sampling, use_diff_op);
	auto res_2 = dom_prim_decomposition(node_2, sampling, use_diff_op);
	auto res_3 = dom_prim_decomposition(node_3, sampling, use_diff_op);

	ASSERT_TRUE(res_0.already_complete());
	ASSERT_TRUE(res_1.already_complete());
	ASSERT_TRUE(res_2.already_complete());

	ASSERT_FALSE(res_3.already_complete());
	ASSERT_TRUE(nodePtrAt(res_3.node, res_3.noop_node_idx)->operationType() == CSGNodeOperationType::Noop);

	writeNode(res_0.node, "decomp_node_0.gv");
	writeNode(res_1.node, "decomp_node_1.gv");
	writeNode(res_2.node, "decomp_node_2.gv");
	writeNode(res_3.node, "decomp_node_3.gv");
}

TEST(DominantPrimOptimizer)
{
	auto s1 = sphere(0, 0, 0, 1, "s1");
	auto s2 = sphere(1, 0, 0, 1, "s2");
	auto s3 = sphere(0.5, 1, 0, 1, "s3");
	auto s4 = sphere(0.5, -1, 0, 1, "s4");
	auto s5 = sphere(2.5, 0, 0, 1, "s5");
	
	auto node = opUnion({ opDiff({ opUnion({ s1, s2 }), opUnion({ s3, s4 }) }), s5 });

	const double sampling = 0.01;
	const bool use_diff_op = true;

	auto params = get_std_ga_params();

	auto opt_node = optimize_with_decomposition(node, sampling, use_diff_op,
		[&params](const CSGNode& node, const PrimitiveCluster& prims)
	{
		return optimize_with_ga(node, params, std::cout, prims).node;
	});

	writeNode(node, "node.gv");
	writeNode(opt_node, "decomp_node_1.gv");	
}

// Experiments to compare different approach on a simple CSG expression 
TEST(CSGExpr1)
{
	auto s1 = sphere(0, 0, 0, 1, "s1");
	auto s2 = sphere(1, 0, 0, 1, "s2");
	auto s3 = sphere(0.5, 1, 0, 1, "s3");
	auto s4 = sphere(0.5, -1, 0, 1, "s4");
	auto s5 = sphere(2.5, 0, 0, 1, "s5");

	auto node = opUnion({ opDiff({ opUnion({ s1, s2 }), opUnion({ s3, s4 }) }), s5 });


	// artificially create a more complex expression
	auto inflated_node = inflate_node(node, 10, { inserter(InserterType::SubtreeCopy, 1.0) });
	writeNode(inflated_node, "inflated_node.gv");


	const double sampling = 0.01;
	const bool use_diff_op = true;

	auto params = get_std_ga_params();

	auto opt_node = optimize_with_decomposition(inflated_node, sampling, use_diff_op,
		[&params](const CSGNode& node, const PrimitiveCluster& prims)
	{
		return optimize_with_ga(node, params, std::cout, prims).node;
	});

	writeNode(opt_node, "decomp_optim_node.gv");


	// For comparison GA + remove redundancy
	auto opt_node_ga = optimize_with_ga(inflated_node, params, std::cout).node;
	auto red_opt_node_ga = remove_redundancies(opt_node_ga, sampling);
	writeNode(red_opt_node_ga, "ga_optim.gv");


	// Clustering + GA 
	auto dom_prims = find_dominating_prims(inflated_node, sampling);
	auto opt_node_cluster = apply_per_cluster_optimization
	(
		cluster_with_dominating_prims(inflated_node, dom_prims),
		[&inflated_node](const PrimitiveCluster& c) { return optimize_with_ga(inflated_node, get_std_ga_params(), std::cout, c).node; },
		union_merge
	);

	auto red_opt_node = remove_redundancies(opt_node_cluster, sampling);
	writeNode(red_opt_node, "red_cluster_ga_optim.gv");
}


// Comment to avoid generating the meshes corresponding to each CSG expression
#define GEN_MESHES 

TEST(CSGExpr2)
{
	auto aabb = cys10().function()->aabb();	
	auto bb = geo<IFBox>(cys10().function()->transform(), Eigen::Vector3d(6,4,2), 1, "");
	
	//auto node = opUnion({ bb, cys10() });
	
	auto node = create_obj_0();
	

#ifdef GEN_MESHES
	// Verify that the object is correct
	auto mesh = lmu::computeMesh(node, Eigen::Vector3i(200, 200, 200));
	igl::writeOBJ("csgexpr2_mesh.obj", mesh.vertices, mesh.indices);

	//writePointCloud("node.xyz", computePointCloud(node, CSGNodeSamplingParams(0.2, 0.5, 0.0)));
#endif


	// Artificially create a more complex expression
	auto inflated_node = inflate_node(node, 3, { inserter(InserterType::SubtreeCopy, 1.0) });
	writeNode(inflated_node, "inflated_node.gv");
	/*
#ifdef GEN_MESHES
	//mesh = lmu::computeMesh(inflated_node, Eigen::Vector3i(200, 200, 200));
	//igl::writeOBJ("csgexpr2_inflated_mesh.obj", mesh.vertices, mesh.indices);

#endif

*/
	// Dominant primitive decomposition
	const double sampling = 0.1;
	const bool use_diff_op = true;

	auto params = get_std_ga_params();

	auto ga_opt_node = optimize_with_ga(inflated_node, params, std::cout).node;
	
	writeNode(ga_opt_node, "ga_opt_node.gv");

#ifdef GEN_MESHES
	mesh = lmu::computeMesh(ga_opt_node, Eigen::Vector3i(200, 200, 200));
	igl::writeOBJ("ga_opt_node_mesh.obj", mesh.vertices, mesh.indices);

#endif
/*


	auto opt_node = optimize_with_decomposition(inflated_node, sampling, use_diff_op,
		[&params](const CSGNode& node, const PrimitiveCluster& prims)
	{
		return optimize_with_ga(node, params, std::cout, prims).node;
	});

	writeNode(opt_node, "decomp_optim_node.gv");

#ifdef GEN_MESHES
	mesh = lmu::computeMesh(opt_node, Eigen::Vector3i(200, 200, 200));
	igl::writeOBJ("csgexpr2_decomp_mesh.obj", mesh.vertices, mesh.indices);
#endif


	// GA + remove redundancy
	auto opt_node_ga = optimize_with_ga(inflated_node, params, std::cout).node;
	auto red_opt_node_ga = remove_redundancies(opt_node_ga, sampling);
	writeNode(opt_node_ga, "ga_optim.gv");
	writeNode(red_opt_node_ga, "red_ga_optim.gv");

#ifdef GEN_MESHES
	auto mesh = lmu::computeMesh(opt_node_ga, Eigen::Vector3i(200, 200, 200));
	igl::writeOBJ("csgexpr2_ga_mesh.obj", mesh.vertices, mesh.indices);

	mesh = lmu::computeMesh(red_opt_node_ga, Eigen::Vector3i(200, 200, 200));
	igl::writeOBJ("csgexpr2_red_ga_mesh.obj", mesh.vertices, mesh.indices);
#endif
*/


	// Clustering + GA 
/*	auto dom_prims = find_dominating_prims(inflated_node, sampling);

	std::cout << "DOM PRIMS: " << std::endl;
	for (const auto& dp : dom_prims)
	{
		std::cout << "DP: " << dp->name() << std::endl;
	}
*/

	//auto opt_node_cluster = apply_per_cluster_optimization
	//(
	//	cluster_with_dominating_prims(inflated_node, dom_prims),
	//	[&inflated_node](const PrimitiveCluster& c) { return optimize_with_ga(inflated_node, get_std_ga_params(), std::cout, c).node; },
	//	union_merge
	//);

	//auto red_opt_node = remove_redundancies(opt_node_cluster, sampling);
	//writeNode(red_opt_node, "red_cluster_ga_optim.gv");

#ifdef GEN_MESHES
	//auto mesh = lmu::computeMesh(red_opt_node, Eigen::Vector3i(200, 200, 200));
	//igl::writeOBJ("csgexpr2_red_cluster_ga_mesh.obj", mesh.vertices, mesh.indices);
#endif

}


#endif
