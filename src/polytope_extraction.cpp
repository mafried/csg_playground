#include "polytope_extraction.h"
#include "primitive_extraction.h"
#include "point_vis.h"

#ifdef _DEBUG
#undef _DEBUG
#include <python.h>
#define _DEBUG
#else
#include <python.h>
#endif


lmu::ConvexCluster::ConvexCluster(const IntermediateConvexCluster& icc) : 
	pc(lmu::pointCloudFromVector(icc.points)),
	planes(icc.planes.begin(), icc.planes.end())
{
}

std::vector<lmu::ConvexCluster> lmu::get_convex_clusters(lmu::PlaneGraph& pg, double max_point_dist, const std::string& python_script)
{	
	auto pc = pg.plane_points();
	lmu::PointCloud debug_pc;

	writePointCloud("C:/Projekte/csg_playground_build/RelWithDebInfo/pc_af.dat", pc);

	auto aff_mat = lmu::get_affinity_matrix(pc, pg.planes(), max_point_dist, debug_pc);

	auto n = std::to_string(aff_mat.rows());
	std::string afm_path = "C:/Projekte/csg_playground_build/RelWithDebInfo/af.dat";

	std::cout << n << " " << afm_path << std::endl;
	
	lmu::write_affinity_matrix(afm_path, aff_mat);

	std::cout << "AM was written." << std::endl;

	// Call Python clustering script.

	std::cout << "Before init." << std::endl;
	
	/*
	Py_SetPath(L"C:/ProgramData/Anaconda3/Lib");
	Py_Initialize();

	std::cout << "After init." << std::endl;
	
	PyObject* od_method_name = PyUnicode_FromString((char*)"clustering");

	PyRun_SimpleString(("import sys\nsys.path.append('" + python_script + "')").c_str());

	PyObject* od_module = PyImport_Import(od_method_name);
	PyObject* od_dict = PyModule_GetDict(od_module);
	PyObject* od_method = PyDict_GetItemString(od_dict, (char*)"get_clusters_and_write_to_file");

	std::cout << "Before call" << std::endl;

	if (PyCallable_Check(od_method))
	{
		PyErr_Print();
		
		PyObject_CallObject(od_method, Py_BuildValue("(z, z)", (char*)afm_path.c_str(), (char*)n.c_str()));
		PyErr_Print();
	}
	else
	{
		PyErr_Print();
	}
	
	Py_DECREF(od_module);
	Py_DECREF(od_method_name);

	Py_Finalize();

	std::cout << "After call" << std::endl;
	*/
	int foo;
	std::cin >> foo;
	
	// Clustering result is in a file. Load it and create clusters.
	std::string cluster_file = python_script + "/clusters.dat";
	
	std::vector<int> per_point_cluster_ids;
	int num_clusters;
		
	std::ifstream cf(cluster_file);
	cf >> num_clusters;
	while (!cf.eof())
	{	
		int cluster_id;
		cf >> cluster_id;
		per_point_cluster_ids.push_back(cluster_id);
	}
	cf.close();

	std::vector<IntermediateConvexCluster> im_clusters(num_clusters);

	auto planes = pg.planes();
	int point_idx = 0;
	for (const auto& plane : planes)
	{
		for (int i = 0; i < plane->pc.rows(); ++i)
		{
			int cluster_id = per_point_cluster_ids[point_idx + i];

			im_clusters[cluster_id].points.push_back(plane->pc.row(i));
			im_clusters[cluster_id].planes.insert(plane);
		}

		point_idx += plane->pc.rows();
	}

	std::vector<ConvexCluster> clusters;
	std::transform(im_clusters.begin(), im_clusters.end(), std::back_inserter(clusters), [](const auto& ic) { return ConvexCluster(ic); });

	std::cout << "Clusters: " << clusters.size() << std::endl;

	return clusters;
}

bool is_mesh_out_of_range(const lmu::Mesh& mesh)
{
	Eigen::Vector3d min = mesh.vertices.colwise().minCoeff();
	Eigen::Vector3d max = mesh.vertices.colwise().maxCoeff();

	return (max - min).norm() > std::sqrt(3.0); //TODO make this variable.
}

lmu::Primitive polytope_from_planes(const lmu::ManifoldSet& planes, const Eigen::Vector3d& center)
{
	std::vector<Eigen::Vector3d> p;
	std::vector<Eigen::Vector3d> n;

	for (int i = 0; i < planes.size(); ++i)
	{
		auto new_plane = std::make_shared<lmu::Manifold>(*planes[i]);

		// Flip plane normal if inside_point would be outside.
		double d = (center - new_plane->p).dot(new_plane->n);
		if (d > 0.0)
		{
			new_plane->n = -1.0 * new_plane->n;
		}

		n.push_back(new_plane->n);
		p.push_back(new_plane->p);
	}

	auto polytope = std::make_shared<lmu::IFPolytope>(Eigen::Affine3d::Identity(), p, n, "");

	if (polytope->empty() || is_mesh_out_of_range(polytope->meshCRef()))
	{
		return lmu::Primitive::None();
	}

	return lmu::Primitive(polytope, planes, lmu::PrimitiveType::Polytope);

}

lmu::Primitive generate_polytope_with_ga(const lmu::ConvexCluster convex_cluster, const lmu::PlaneGraph& plane_graph,
	const lmu::PrimitiveGaParams& params, std::ofstream& s, const std::shared_ptr<lmu::PrimitiveSetRanker>& ranker, 
	const Eigen::Vector3d& polytope_center, int max_primitive_set_size)
{
	double angle_t = M_PI / 9.0;
	

	lmu::PrimitiveSetGA::Parameters ga_params(50, 2, 0.4, 0.4, true, lmu::Schedule(), lmu::Schedule(), true);

	lmu::PrimitiveSetTournamentSelector selector(2);
	lmu::PrimitiveSetIterationStopCriterion criterion(params.max_count, lmu::PrimitiveSetRank(0.00001), params.max_iterations);
	lmu::PrimitiveSetCreator creator(plane_graph, 0.0, { 0.40, 0.15, 0.15, 0.15, 0.15 }, 1, 1, max_primitive_set_size, angle_t, 0.001,
		params.polytope_prob, params.min_polytope_planes, params.max_polytope_planes, polytope_center, convex_cluster.planes);

	lmu::PrimitiveSetPopMan popMan(*ranker, max_primitive_set_size, params.geo_weight, params.per_prim_geo_weight, params.size_weight, 
		params.num_elite_injections);
	
	lmu::PrimitiveSetGA ga;

	auto res = ga.run(ga_params, selector, creator, *ranker, criterion, popMan);

	res.statistics.save(s);

	// Filter primitives
	lmu::ThresholdOutlierDetector od(params.filter_threshold);
	lmu::SimilarityFilter sf(params.similarity_filter_epsilon, params.similarity_filter_voxel_size, params.similarity_filter_similarity_only,
		params.similarity_filter_perfectness_t);

	auto polytopes = res.population[0].creature;

	polytopes = polytopes.without_duplicates();

	polytopes = od.remove_outliers(polytopes, *ranker);

	polytopes = sf.filter(polytopes, *ranker);

	// Get best polytope
	auto geo_scores = ranker->rank(polytopes).per_primitive_geo_scores;	
	int max_score_idx = 0;
	double max_score = -std::numeric_limits<double>::max();
	for (int i = 0; i < polytopes.size(); ++i)
	{
		if (geo_scores[i] > max_score)
		{
			max_score = geo_scores[i];
			max_score_idx = i;
		}
	}
	auto polytope = polytopes.size() > max_score_idx ? polytopes[max_score_idx] : lmu::Primitive::None();
	
	if (!polytope.isNone())
	{
		std::cout << "RANK: " << res.population[0].rank.per_primitive_geo_scores[0] << std::endl;
	}

	return polytope;
}

#include <igl/writeOBJ.h>

lmu::Primitive generate_polytope(const lmu::ConvexCluster convex_cluster, const lmu::PlaneGraph& plane_graph, 
	const lmu::PrimitiveGaParams& params, std::ofstream& s)
{
	static int i = 0;

	std::cout << "----------------------------" << std::endl;
	std::cout << "Points: " << convex_cluster.pc.rows() << std::endl;
	std::cout << "Planes: " << convex_cluster.planes.size() << std::endl;

	// Compute polytope center.
	Eigen::Vector3d center;
	double num_points = 0.0;
	for (int  i = 0; i < convex_cluster.pc.rows(); ++i)
	{		
		center += Eigen::Vector3d(convex_cluster.pc.row(i).x(), convex_cluster.pc.row(i).y(), convex_cluster.pc.row(i).z());
		num_points += 1.0;
		
	}
	center /= num_points;
	std::cout << "Center: " << center.transpose() << std::endl;

	// Compute model sdf. 
	auto model_sdf = std::make_shared<lmu::ModelSDF>(convex_cluster.pc, params.sdf_voxel_size, s);
	
	igl::writeOBJ("p_mesh_" + std::to_string(i++) + ".obj", model_sdf->surface_mesh.vertices, model_sdf->surface_mesh.indices);

	// Create polytope ranker.
	int max_primitive_set_size = 2;
	auto ranker = std::make_shared<lmu::PrimitiveSetRanker>(
		lmu::farthestPointSampling(convex_cluster.pc, params.num_geo_score_samples),
		params.max_dist, max_primitive_set_size, params.ranker_voxel_size, params.allow_cube_cutout, model_sdf,
		params.geo_weight, params.per_prim_geo_weight, params.size_weight);

	// Try to create a polytope with all planes in the convex cluster.
	// If not possible, use ga. 
	auto polytope = polytope_from_planes(convex_cluster.planes, center);
	lmu::PrimitiveSet ps; ps.push_back(polytope);
	double polytope_score = ranker->rank(ps).per_primitive_geo_scores[0];

	if (!polytope.isNone() && polytope_score == 1.0)
	{		
		return polytope;
	}
	else
	{
		std::cout << "Polytope is not valid or its score is not perfect. Score: " << polytope_score << std::endl;

		return generate_polytope_with_ga(convex_cluster, plane_graph, params, s, ranker, center, max_primitive_set_size);
	}
}

lmu::PrimitiveSet lmu::generate_polytopes(const std::vector<ConvexCluster>& convex_clusters, const PlaneGraph& plane_graph,
	const lmu::PrimitiveGaParams& params, std::ofstream& s)
{
	lmu::PrimitiveSet ps;

	// Initialize polytope creator.
	initializePolytopeCreator();

	for (const auto& cc : convex_clusters)
	{
		auto polytope = generate_polytope(cc, plane_graph, params, s);
		if (!polytope.isNone())
			ps.push_back(polytope);
	}

	std::cout << "Created polytopes: " << ps.size() << std::endl;

	return ps;
}
