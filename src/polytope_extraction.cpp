#include "polytope_extraction.h"
#include "primitive_extraction.h"
#include "point_vis.h"

#include <boost/math/special_functions/erf.hpp>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/compute_average_spacing.h>
#include <CGAL/remove_outliers.h>

typedef CGAL::Simple_cartesian<double> K;


#ifdef _DEBUG
#undef _DEBUG
#include <python.h>
#define _DEBUG
#else
#include <python.h>
#endif

inline double median(std::vector<double> v)
{
	std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
	return v[v.size() / 2];
}

std::tuple<double, double> scaled3MADAndMedian(const lmu::PointCloud& pc, const lmu::ModelSDF& msdf)
{
	std::vector<double> values(pc.rows());

	for (int j = 0; j < pc.rows(); ++j)
	{
		Eigen::Vector3d p = pc.row(j).leftCols(3);
		values[j] = std::abs(msdf.distance(p));
	}

	double med = median(values);

	std::transform(values.begin(), values.end(), values.begin(), [med](double v) -> double { return std::abs(v - med); });

	const double c = -1.0 / (std::sqrt(2.0)*boost::math::erfc_inv(3.0 / 2.0));

	return std::make_tuple(c * median(values) * 3.0, med);
}


lmu::ConvexCluster::ConvexCluster(const IntermediateConvexCluster& icc, bool rem_outliers) : 
	pc(lmu::pointCloudFromVector(icc.points)),
	planes(icc.planes.begin(), icc.planes.end())
{
	if (rem_outliers)
		remove_outliers();
}

void lmu::ConvexCluster::remove_outliers()
{
	typedef std::pair<K::Point_3, K::Vector_3> PointNormal;

	std::vector<PointNormal> points;
	points.reserve(pc.rows());
	for (int i = 0; i < pc.rows(); ++i)
		points.push_back(std::make_pair(
			K::Point_3(pc.coeff(i, 0), pc.coeff(i, 1), pc.coeff(i, 2)), K::Vector_3(pc.coeff(i, 3), pc.coeff(i, 4), pc.coeff(i, 5))));

	const int nb_neighbors = 24; 
	auto average_spacing = CGAL::compute_average_spacing<CGAL::Sequential_tag>(points, nb_neighbors, 
		CGAL::parameters::point_map(CGAL::First_of_pair_property_map<PointNormal>()));
	
	auto first_to_remove = CGAL::remove_outliers(points.begin(), points.end(), CGAL::First_of_pair_property_map<PointNormal>(), nb_neighbors,  100., 2. * average_spacing);

	std::cout << "Removed " << (100. * std::distance(first_to_remove, points.end()) / (double)(points.size())) << "% of the points." << std::endl;

	points.erase(first_to_remove, points.end());

	pc = lmu::PointCloud(points.size(), 6);
	for (int i = 0; i < points.size(); ++i)
		pc.row(i) << points[i].first.x(), points[i].first.y(), points[i].first.z(), points[i].second.x(), points[i].second.y(), points[i].second.z();
		
}

Eigen::Vector3d lmu::ConvexCluster::compute_center(const lmu::ModelSDF& msdf) const
{
	auto mad_and_median = scaled3MADAndMedian(pc, msdf);
	
	Eigen::Vector3d center;
	double num_points = 0.0;
	for (int i = 0; i < pc.rows(); ++i)
	{
		Eigen::Vector3d p(pc.row(i).x(), pc.row(i).y(), pc.row(i).z());

		if (std::abs(std::abs(msdf.distance(p)) - std::get<1>(mad_and_median)) < std::get<0>(mad_and_median))
		{		
			center += p;
			num_points += 1.0;
		}
	}
	center /= num_points;

	return center;
}

std::vector<lmu::ConvexCluster> lmu::get_convex_clusters(lmu::PlaneGraph& pg, const std::string& python_script, double am_clustering_param)
{
	std::string cluster_file = "clusters.dat";
	std::string afm_path = "af.dat";
	std::string pcaf_path = "pc_af.dat";

	auto pc = pg.plane_points();
	lmu::PointCloud debug_pc;

	writePointCloud(pcaf_path, pc);

	auto aff_mat = lmu::get_affinity_matrix_with_triangulation(pc, pg.planes(), true);//lmu::get_affinity_matrix(pc, pg.planes(), true, debug_pc);

	auto n = std::to_string(aff_mat.rows());
	std::cout << n << " " << afm_path << std::endl;

	auto am_clustering_param_str = std::to_string(am_clustering_param);

	lmu::write_affinity_matrix(afm_path, aff_mat);

	std::cout << "AM was written." << std::endl;

	// Call Python clustering script.

	std::cout << "Before init." << std::endl;

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

		PyObject_CallObject(od_method, Py_BuildValue("(z, z, z, z)", (char*)afm_path.c_str(), (char*)n.c_str(), (char*)cluster_file.c_str(), (char*)am_clustering_param_str.c_str()));
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

	// Clustering result is in a file. Load it and create clusters.

	std::vector<int> per_point_cluster_ids;
	int num_clusters;

	std::ifstream cf(cluster_file);

	std::cout << "Stream is open: " << cf.is_open() << std::endl;

	cf >> num_clusters;
	std::cout << "Num Clusters: " << num_clusters << std::endl;
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

	std::cout << "Planes: " << planes.size() << std::endl;
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
	std::transform(im_clusters.begin(), im_clusters.end(), std::back_inserter(clusters), [](const auto& ic) { return ConvexCluster(ic, true); });

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
	const Eigen::Vector3d& polytope_center)
{
	double angle_t = M_PI / 9.0;
	

	lmu::PrimitiveSetGA::Parameters ga_params(50, 2, 0.4, 0.4, true, lmu::Schedule(), lmu::Schedule(), true);

	lmu::PrimitiveSetTournamentSelector selector(2);
	lmu::PrimitiveSetIterationStopCriterion criterion(params.max_count, lmu::PrimitiveSetRank(0.00001), params.max_iterations);
	lmu::PrimitiveSetCreator creator(plane_graph, 0.0, { 0.40, 0.15, 0.15, 0.15, 0.15 }, 1, 1, params.maxPrimitiveSetSize, angle_t, 0.001,
		params.polytope_prob, params.min_polytope_planes, params.max_polytope_planes, polytope_center, convex_cluster.planes);

	lmu::PrimitiveSetPopMan popMan(*ranker, params.maxPrimitiveSetSize, params.geo_weight, params.per_prim_geo_weight, params.size_weight,
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
	
	// Compute model sdf. 
	std::shared_ptr<lmu::ModelSDF> model_sdf = nullptr; 
	try
	{
		model_sdf = std::make_shared<lmu::ModelSDF>(convex_cluster.pc, params.sdf_voxel_size, s);
	}
	catch (const std::runtime_error& ex)
	{
		std::cout << "Could not generate polytope. Reason: " << std::string(ex.what()) << std::endl;
		return lmu::Primitive::None();
	}

	//igl::writeOBJ("p_mesh_" + std::to_string(i++) + ".obj", model_sdf->surface_mesh.vertices, model_sdf->surface_mesh.indices);

	// Compute polytope center.
	Eigen::Vector3d center = convex_cluster.compute_center(*model_sdf);
	std::cout << "Center: " << center.transpose() << std::endl;
	std::cout << "GA Threshold: " << params.ga_threshold << std::endl;

	// Create polytope ranker.
	auto ranker = std::make_shared<lmu::PrimitiveSetRanker>(
		lmu::farthestPointSampling(convex_cluster.pc, params.num_geo_score_samples),
		params.max_dist, params.maxPrimitiveSetSize, params.ranker_voxel_size, params.allow_cube_cutout, model_sdf,
		params.geo_weight, params.per_prim_geo_weight, params.size_weight);

	// Try to create a polytope with all planes in the convex cluster.
	// If not possible, use ga. 
	auto polytope = polytope_from_planes(convex_cluster.planes, center);
	lmu::PrimitiveSet ps; ps.push_back(polytope);
	double polytope_score = ranker->rank(ps).per_primitive_geo_scores[0];

	if (!polytope.isNone() && polytope_score >= params.ga_threshold)
	{		
		return polytope;
	}
	else
	{
		std::cout << "Polytope is not valid or its score is not perfect. Score: " << polytope_score << std::endl;

		return generate_polytope_with_ga(convex_cluster, plane_graph, params, s, ranker, center);
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
		if (cc.planes.size() < 4 || cc.pc.rows() == 0)
		{
			std::cout << "cluster skipped. Planes: " << cc.planes.size() << " Points: " << cc.pc.rows() << std::endl;
			continue;
		}

		auto polytope = generate_polytope(cc, plane_graph, params, s);
		if (!polytope.isNone())
			ps.push_back(polytope);
	}

	std::cout << "Created polytopes: " << ps.size() << std::endl;

	name_primitives(ps);

	return ps;
}

lmu::Primitive merge_to_single_polytope(const lmu::PrimitiveSet& ps)
{
	if (ps.size() == 1)
		return ps[0];

	//Collect all planes from all polytopes.
	std::vector<Eigen::Vector3d> n;
	std::vector<Eigen::Vector3d> pos;
	std::set<lmu::ManifoldPtr> manifolds;
	for (const auto& p : ps)
	{
		if (p.type == lmu::PrimitiveType::Polytope)
		{
			auto p_ptr = (lmu::IFPolytope*)p.imFunc.get();
			auto _n = p_ptr->n();
			auto _pos = p_ptr->p();
			n.insert(n.end(), _n.begin(), _n.end());
			pos.insert(pos.end(), _pos.begin(), _pos.end());

			for(const auto& m : p.ms)
				manifolds.insert(m);
		}
	}

	//Merge that are duplicates and filter out double planes. 
	// Double planes are planes that exist twice but with normals in opposite directions.
	// Double planes are completely removed since they mark surface regions that need to be open for the merge.
	std::vector<Eigen::Vector3d> f_n;
	std::vector<Eigen::Vector3d> f_pos;
	std::unordered_set<int> double_planes; 
	for (int i = 0; i < n.size(); ++i)
	{
		bool duplicate = false;
		for (int j = i+1; j < n.size(); ++j)
		{
			K::Plane_3 p_i(K::Point_3(pos[i].x(), pos[i].y(), pos[i].z()), K::Vector_3(n[i].x(), n[i].y(), n[i].z()));
			K::Plane_3 p_j(K::Point_3(pos[j].x(), pos[j].y(), pos[j].z()), K::Vector_3(n[j].x(), n[j].y(), n[j].z()));

			// If the intersection of the two planes i,j is a plane, then plane i is a duplicate.
			auto result = CGAL::intersection(p_i, p_j);
			if (result)
			{
				if (boost::get<K::Plane_3>(&*result))
				{
					if (p_i.orthogonal_vector() == -p_j.orthogonal_vector())
						double_planes.insert(j);

					duplicate = true; 
					break;
				}
			}
		}
		if (!duplicate && double_planes.find(i) == double_planes.end())
		{
			f_n.push_back(n[i]);
			f_pos.push_back(pos[i]);
		}
	}
	
	std::stringstream ss; 
	for (const auto& p : ps)
		ss << p.imFunc->name() << "_";
	auto new_name = ss.str().substr(0, ss.str().size() - 1);

	// Create polytope. 
	return lmu::Primitive(
		std::make_shared<lmu::IFPolytope>(Eigen::Affine3d::Identity(), f_pos, f_n, new_name), 
		lmu::ManifoldSet(manifolds.begin(), manifolds.end()), 
		lmu::PrimitiveType::Polytope
	);
}

bool can_be_merged(const lmu::Primitive& p0, const lmu::Primitive& p1, double am_quality_threshold)
{
	auto afm = lmu::get_affinity_matrix(p0.imFunc->meshCRef(), p1.imFunc->meshCRef());

	double s = ((double)afm.sum() / (double)afm.size());

	return s >= am_quality_threshold;
}

lmu::PrimitiveSet lmu::merge_polytopes(const lmu::PrimitiveSet& ps, double am_quality_threshold)
{
	lmu::PrimitiveSet candidates = ps; 	

	std::cout << "Candidates: " << candidates.size() << std::endl;
	std::cout << "AM Quality Threshold: " << am_quality_threshold << std::endl;

	bool something_was_merged = true;
	while (something_was_merged)
	{
		something_was_merged = false;
		
		for (int i = 0; i < candidates.size() && !something_was_merged; ++i)
		{
			for (int j = i + 1; j < candidates.size(); ++j)
			{
				if (can_be_merged(candidates[i], candidates[j], am_quality_threshold))
				{
					something_was_merged = true;

					PrimitiveSet to_merge;
					to_merge.push_back(candidates[i]);
					to_merge.push_back(candidates[j]);

					std::cout << to_merge[0].imFunc->name() << " and " << to_merge[1].imFunc->name() << " are merged into ";

					candidates[i] = merge_to_single_polytope(to_merge);

					std::cout << candidates[i].imFunc->name() << "." << std::endl;

					candidates.erase(candidates.begin() + j);

					break;
				}
			}
		}
	}
	
	return candidates;
}