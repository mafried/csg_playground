#include "primitive_extraction.h"
#include "primitive_helper.h"
#include "csgnode.h"
#include "csgnode_helper.h"
#include "helper.h"

#include "igl/signed_distance.h"
#include <igl/per_vertex_normals.h>
#include <igl/per_edge_normals.h>
#include <igl/per_face_normals.h>

lmu::PrimitiveSetRank const lmu::PrimitiveSetRank::Invalid = lmu::PrimitiveSetRank(-std::numeric_limits<double>::max());

std::ostream& lmu::operator<<(std::ostream& out, const lmu::DHType& t)
{
	switch (t)
	{
	case DHType::INSIDE:
		out << "Inside";
		break;
	case DHType::OUTSIDE:
		out << "Outside";
		break;
	case DHType::NONE:
		out << "None";
		break;
	}

	return out;
}

lmu::Mesh computeMeshFromPrimitives(const lmu::PrimitiveSet& ps, int primitive_idx = -1)
{
	if (ps.empty())
		return lmu::Mesh();

	lmu::PrimitiveSet filtered_ps;
	if (primitive_idx < 0)
		filtered_ps = ps;
	else
		filtered_ps.push_back(ps[primitive_idx]);

	int vRows = 0;
	int iRows = 0;
	for (const auto& p : filtered_ps)
	{
		auto mesh = p.imFunc->createMesh();
		vRows += mesh.vertices.rows();
		iRows += mesh.indices.rows();
	}

	Eigen::MatrixXi indices(iRows, 3);
	Eigen::MatrixXd vertices(vRows, 3);
	int vOffset = 0;
	int iOffset = 0;
	for (const auto& p : filtered_ps)
	{
		auto mesh = p.imFunc->createMesh();

		Eigen::MatrixXi newIndices(mesh.indices.rows(), 3);
		newIndices << mesh.indices;

		newIndices.array() += vOffset;

		indices.block(iOffset, 0, mesh.indices.rows(), 3) << newIndices;
		vertices.block(vOffset, 0, mesh.vertices.rows(), 3) << mesh.vertices;

		vOffset += mesh.vertices.rows();
		iOffset += mesh.indices.rows();
	}

	return lmu::Mesh(vertices, indices);
}

#include <igl/writeOBJ.h>

std::ostream & lmu::operator<<(std::ostream & out, const PrimitiveSetRank & r)
{
	out << "{ \"combined\": " << r.combined << ", \"geo\": " << r.geo << ", \"size\": " << r.size << ", \"per_prim_geo_sum\": " << r.per_prim_geo_sum << 
		", \"size_unnormalized\": " << r.size_unnormalized << ", \"geo_unnormalized\":" << r.geo_unnormalized << ", \"per_prim_mean_score\": " << r.per_primitive_mean_score << ", \"per_prim_coverage_sum_score\": " << r.per_prim_coverage_sum << "}";
	return out;
}

void lmu::name_primitives(const lmu::PrimitiveSet& ps)
{
	std::unordered_map<lmu::PrimitiveType, int> counter;

	for (const auto& p : ps)
		p.imFunc->setName(lmu::primitiveTypeToString(p.type) + std::to_string((counter[p.type]++)));
}

lmu::PrimitiveSet lmu::extractNonPlanarPrimitives(const lmu::ManifoldSet& manifolds)
{
	lmu::PrimitiveSet primitives;

	for (const auto& manifold : manifolds)
	{
		if (manifold->type == lmu::ManifoldType::Sphere)
		{
			primitives.push_back(lmu::createSpherePrimitive(manifold));
		}
		else if (manifold->type == lmu::ManifoldType::Cylinder) {
			lmu::ManifoldSet planes;
			primitives.push_back(lmu::createCylinderPrimitive(manifold, planes));
		}

	}

	return primitives;
}

lmu::GAResult lmu::extractPolytopePrimitivesWithGA(const PlaneGraph& plane_graph, const std::shared_ptr<ModelSDF>& model_sdf, 
	const PrimitiveGaParams& params, std::ostream& stream)
{	
	double distT = 0.02;
	double angleT = M_PI / 9.0;
	int maxPrimitiveSetSize = params.maxPrimitiveSetSize;
	
	double size_weight = params.size_weight;
	double geo_weight = params.geo_weight;
	double per_prim_geo_weight = params.per_prim_geo_weight;
	double per_prim_coverage_weight = params.per_prim_coverage_weight;

	lmu::PrimitiveSetGA::Parameters paramsGA1(50, 2, 0.4, 0.4, true, Schedule(), Schedule(), true);

	// Initialize polytope creator.
	initializePolytopeCreator();
	
	// get union of all plane pointclouds.
	auto plane_pointcloud = plane_graph.plane_points();
		 
	// First GA for candidate box generation.
	PrimitiveSetTournamentSelector selector(2);
	PrimitiveSetIterationStopCriterion criterion(params.max_count, PrimitiveSetRank(0.00001), params.max_iterations);
	PrimitiveSetCreator creator(plane_graph, 0.0, { 0.40, 0.15, 0.15, 0.15, 0.15 }, 1, 1, maxPrimitiveSetSize, angleT, 0.001, 
		params.polytope_prob, params.neighbor_prob, params.min_polytope_planes, params.max_polytope_planes, Eigen::Vector3d(0,0,0), ManifoldSet());
	
	auto ranker = std::make_shared<PrimitiveSetRanker>(farthestPointSampling(plane_pointcloud, params.num_geo_score_samples),
		params.max_dist, maxPrimitiveSetSize, params.ranker_voxel_size, params.allow_cube_cutout, model_sdf, 
		geo_weight, per_prim_geo_weight, per_prim_coverage_weight, size_weight);

	PrimitiveSetPopMan popMan(*ranker, maxPrimitiveSetSize, geo_weight, per_prim_geo_weight, per_prim_coverage_weight, size_weight, params.num_elite_injections);
	PrimitiveSetGA ga;

	std::cout << "Start GA" << std::endl;
	auto res = ga.run(paramsGA1, selector, creator, *ranker, criterion, popMan);
	
	res.statistics.save(stream);

	// ================ TMP ================
	/*std::cout << "Serialize meshes" << std::endl;
	std::string basename = "mid_out_mesh
	for (int i = 0; i < primitives.size(); ++i) {
	auto mesh = computeMeshFromPrimitives(primitives, i);
	if (!mesh.empty()) {
	std::string mesh_name = basename + std::to_string(i) + ".obj";
	igl::writeOBJ(mesh_name, mesh.vertices, mesh.indices);
	}
	}*/
	// ================ TMP ================

	auto polytopes = res.population[0].creature;

	name_primitives(polytopes);

	GAResult result;
	result.polytopes = polytopes;
	result.plane_graph = plane_graph;
	result.ranker = ranker;

	return result;	
}

// ==================== CREATOR ====================

lmu::PrimitiveSetCreator::PrimitiveSetCreator(const PlaneGraph& plane_graph, double intraCrossProb,
	const std::vector<double>& mutationDistribution, int maxMutationIterations, int maxCrossoverIterations,
	int maxPrimitiveSetSize, double angleEpsilon, double minDistanceBetweenParallelPlanes, double polytope_prob, double neighbor_prob, int min_polytope_planes,
	int max_polytope_planes, const Eigen::Vector3d& polytope_center, const ManifoldSet& cluster_planes) :
	plane_graph(plane_graph),
	intraCrossProb(intraCrossProb),
	mutationDistribution(mutationDistribution),
	maxMutationIterations(maxMutationIterations),
	maxCrossoverIterations(maxCrossoverIterations),
	maxPrimitiveSetSize(maxPrimitiveSetSize),
	angleEpsilon(angleEpsilon),
	minDistanceBetweenParallelPlanes(minDistanceBetweenParallelPlanes),
	polytope_prob(polytope_prob),
	neighbor_prob(neighbor_prob),
	min_polytope_planes(min_polytope_planes), 
	max_polytope_planes(max_polytope_planes),
	polytope_center(polytope_center),
	cluster_planes(cluster_planes)
{
	rndEngine.seed(rndDevice());
}

int lmu::PrimitiveSetCreator::getRandomPrimitiveIdx(const PrimitiveSet& ps) const
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	return du(rndEngine, parmu_t{ 0, (int)ps.size() - 1 });
}

lmu::PrimitiveSet lmu::PrimitiveSetCreator::mutate(const PrimitiveSet& ps) const
{
	static std::discrete_distribution<int> dd{ mutationDistribution.begin(), mutationDistribution.end() };
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	MutationType mt = (MutationType)dd(rndEngine);

	if (mt == MutationType::NEW || ps.empty())
	{
		//std::cout << "Mutation New" << std::endl;
		return create();
	}
	else
	{
		auto newPS = ps;

		for (int i = 0; i < du(rndEngine, parmu_t{ 1, (int)maxMutationIterations }); ++i)
		{
			switch (mt)
			{
			case MutationType::REPLACE:
			{
				//std::cout << "Mutation Replace" << std::endl;

				int idx = getRandomPrimitiveIdx(newPS);
				if (idx != -1)
				{
					auto newP = createPrimitive();
					newPS[idx] = newP.isNone() ? newPS[idx] : newP;
				}

				break;
			}
			case MutationType::MODIFY:
			{
				//std::cout << "Mutation Modify" << std::endl;

				int idx = getRandomPrimitiveIdx(newPS);
				auto newP = mutatePrimitive(newPS[idx], angleEpsilon);
				newPS[idx] = newP.isNone() ? newPS[idx] : newP;

				break;
			}
			case MutationType::REMOVE:
			{
				//std::cout << "Mutation Remove" << std::endl;

				int idx = getRandomPrimitiveIdx(newPS);
				newPS.erase(newPS.begin() + idx);

				break;
			}
			case MutationType::ADD:
			{
				//std::cout << "Mutation Add" << std::endl;

				auto newP = createPrimitive();
				if (!newP.isNone() && newPS.size() < maxMutationIterations)
					newPS.push_back(newP);

				break;
			}
			default:
				std::cout << "Warning: Unknown mutation type." << std::endl;
			}
		}

		return newPS;
	}
}

std::vector<lmu::PrimitiveSet> lmu::PrimitiveSetCreator::crossover(const PrimitiveSet& ps1, const PrimitiveSet& ps2) const
{
	//std::cout << "Crossover" << std::endl;

	static std::bernoulli_distribution db{};
	using parmb_t = decltype(db)::param_type;

	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	PrimitiveSet newPS1 = ps1;
	PrimitiveSet newPS2 = ps2;

	for (int i = 0; i < du(rndEngine, parmu_t{ 1, (int)maxCrossoverIterations }); ++i)
	{
		bool intra = db(rndEngine, parmb_t{ intraCrossProb });

		if (intra)
		{
			//TODO (if it makes sense).
		}
		else
		{
			if (!ps1.empty() && !ps2.empty())
			{
				int idx1 = getRandomPrimitiveIdx(ps1);
				int idx2 = getRandomPrimitiveIdx(ps2);

				if (idx1 != -1 && idx2 != -1)
				{
					//newPS1[idx1] = ps2[idx2];
					//newPS2[idx2] = ps1[idx1];
					for (int j = idx2; j < std::min(newPS1.size(), ps2.size()); ++j) {
						newPS1[j] = ps2[j];
					}

					for (int j = idx1; j < std::min(ps1.size(), newPS2.size()); ++j) {
						newPS2[j] = ps1[j];
					}
				}
			}
		}
	}

	return { newPS1, newPS2 };
}

lmu::PrimitiveSet lmu::PrimitiveSetCreator::create() const
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	static std::bernoulli_distribution db{};
	using parmb_t = decltype(db)::param_type;

	int setSize = du(rndEngine, parmu_t{ 1, (int)maxPrimitiveSetSize });

	PrimitiveSet ps;

	// Fill primitive set with randomly created primitives. 
	while (ps.size() < setSize)
	{
		//std::cout << "try to create primitive" << std::endl;
		auto p = createPrimitive();
		if (!p.isNone())
		{
			ps.push_back(p);

			//std::cout << "Added Primitive" << std::endl;
		}
		else
		{
			//std::cout << "Added None" << std::endl;
		}
	}

	//std::cout << "PS SIZE: " << ps.size() << std::endl;

	return ps;
}

std::string lmu::PrimitiveSetCreator::info() const
{
	return std::string();
}

lmu::ManifoldPtr lmu::PrimitiveSetCreator::getManifold(ManifoldType type, const Eigen::Vector3d& direction,
	const ManifoldSet& alreadyUsed, double angleEpsilon, bool ignoreDirection,
	const Eigen::Vector3d& point, double minimumPointDistance, bool ignorePointDistance, bool use_cluster_planes_only) const
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	ManifoldSet candidates;
	const double cos_e = std::cos(angleEpsilon);

	auto all_planes = use_cluster_planes_only ? cluster_planes : plane_graph.planes();

	// Filter manifold list.
	std::copy_if(all_planes.begin(), all_planes.end(), std::back_inserter(candidates),
		[type, &alreadyUsed, &direction, cos_e, ignoreDirection, &point, minimumPointDistance, ignorePointDistance](const ManifoldPtr& m)
	{
		//std::cout << (direction.norm() || ignoreDirection) << " " << m->n.norm() << std::endl;

		return
			m->type == type &&												// same type.
			std::find_if(alreadyUsed.begin(), alreadyUsed.end(),			// not already used.
				[&m, minimumPointDistance](const ManifoldPtr& alreadyUsedM) {
			return lmu::manifoldsEqual(*m, *alreadyUsedM, 0.0001);
		}) == alreadyUsed.end() &&
			(ignoreDirection || std::abs(direction.dot(m->n)) > cos_e) &&	// same direction (or flipped).
			(ignorePointDistance || std::abs((point - m->p).dot(m->n)) > minimumPointDistance);	// distance between point and plane  
	});

	if (candidates.empty())
		return nullptr;

	return candidates[du(rndEngine, parmu_t{ 0, (int)candidates.size() - 1 })];
}

lmu::ManifoldPtr lmu::PrimitiveSetCreator::getNeighborPlane(const ManifoldPtr& plane, const ManifoldSet& already_used) const
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	auto n_planes = plane_graph.connected(plane);
	ManifoldSet unused_n_planes;

	// Filter out those planes that are already in use.
	std::copy_if(n_planes.begin(), n_planes.end(), std::back_inserter(unused_n_planes), [&already_used](const auto& e)
	{
		return std::find(already_used.begin(), already_used.end(), e) == already_used.end();
	});

	// If no neighbor plane left, return any unused plane.
	if (unused_n_planes.empty())
	{
		return getManifold(ManifoldType::Plane, Eigen::Vector3d(0, 0, 0), already_used, 0.0, true);
	}
	else // Else return a randomly selected plane neighbor plane.
	{
		//std::cout << "neighbor plane found" << std::endl;
		return unused_n_planes[du(rndEngine, parmu_t{ 0, (int)unused_n_planes.size() - 1 })];
	}
}

lmu::ManifoldPtr lmu::PrimitiveSetCreator::getPerpendicularPlane(const std::vector<ManifoldPtr>& planes,
	const ManifoldSet& alreadyUsed, double angleEpsilon) const
{
	//std::cout << "perp ";

	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	ManifoldSet candidates;
	const double cos_e = std::cos(angleEpsilon);

	auto all_planes = plane_graph.planes();

	// Filter manifold list.
	std::copy_if(all_planes.begin(), all_planes.end(), std::back_inserter(candidates),
		[&alreadyUsed, &planes, cos_e](const ManifoldPtr& m)
	{
		if (m->type != ManifoldType::Plane)
		{
			return false;
		}

		if (std::find_if(alreadyUsed.begin(), alreadyUsed.end(), [&m](const ManifoldPtr& alreadyUsedM)
		{ return lmu::manifoldsEqual(*m, *alreadyUsedM, 0.0001); }) != alreadyUsed.end())
		{
			return false;
		}

		for (const auto& plane : planes)
		{
			if (std::abs(plane->n.dot(m->n)) >= cos_e) // enforce perpendicular direction.
				return false;
		}

		return true;

	});

	if (candidates.empty())
		return nullptr;

	return candidates[du(rndEngine, parmu_t{ 0, (int)candidates.size() - 1 })];
}

lmu::ManifoldPtr lmu::PrimitiveSetCreator::getParallelPlane(const ManifoldPtr& plane, const ManifoldSet & alreadyUsed,
	double angleEpsilon, double minDistanceToParallelPlane) const
{
	auto foundPlane = getManifold(ManifoldType::Plane, plane->n, alreadyUsed, angleEpsilon, false, plane->p, minDistanceToParallelPlane);
	return foundPlane;
}

std::unordered_set<lmu::ManifoldType> lmu::PrimitiveSetCreator::getAvailableManifoldTypes(const ManifoldSet & ms) const
{
	std::unordered_set<lmu::ManifoldType> amt;

	std::transform(ms.begin(), ms.end(), std::inserter(amt, amt.begin()),
		[](const auto& m) -> lmu::ManifoldType { return m->type; });

	return amt;
}

lmu::PrimitiveType lmu::PrimitiveSetCreator::getRandomPrimitiveType() const
{
	static std::bernoulli_distribution db{};
	using parmb_t = decltype(db)::param_type;

	return db(rndEngine, parmb_t{ polytope_prob }) ? PrimitiveType::Polytope : PrimitiveType::Box;
}

lmu::Primitive lmu::PrimitiveSetCreator::createPrimitive() const
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	static std::bernoulli_distribution db{};
	using parmb_t = decltype(db)::param_type;

	const auto anyDirection = Eigen::Vector3d(0, 0, 0);

	auto primitiveType = getRandomPrimitiveType();
	//std::cout << "Primitive Type: " << (int)primitiveType << std::endl;
	auto primitive = Primitive::None();

	switch (primitiveType)
	{
	case PrimitiveType::Box:
	{
		ManifoldSet planes;

		auto plane = getManifold(ManifoldType::Plane, anyDirection, {}, 0.0, true);
		if (!plane)
			break;
		planes.push_back(plane);

		plane = getParallelPlane(plane, planes, angleEpsilon, minDistanceBetweenParallelPlanes);
		if (!plane)
			break;
		planes.push_back(plane);

		plane = getPerpendicularPlane(planes, planes, angleEpsilon);
		if (!plane)
			break;
		planes.push_back(plane);

		plane = getParallelPlane(plane, planes, angleEpsilon, minDistanceBetweenParallelPlanes);
		if (!plane)
			break;
		planes.push_back(plane);

		plane = getPerpendicularPlane(planes, planes, angleEpsilon);
		if (!plane)
			break;
		planes.push_back(plane);

		plane = getParallelPlane(plane, planes, angleEpsilon, minDistanceBetweenParallelPlanes);
		if (!plane)
			break;
		planes.push_back(plane);
		
		primitive = createBoxPrimitive(planes);
	}
	break;

	case PrimitiveType::Polytope:
	{
		ManifoldSet planes;
		ManifoldSet usable_cluster_planes = cluster_planes;
		
		std::bernoulli_distribution bdist(neighbor_prob);


		int max_planes = du(rndEngine, parmu_t{ min_polytope_planes, max_polytope_planes });
		int max_fixed_planes = bdist(rndEngine) ? du(rndEngine, parmu_t{ 1, (int)usable_cluster_planes.size() }) : (int)usable_cluster_planes.size();
		int num_fixed_planes = 0;
		ManifoldPtr cur_plane;
		for( int i = 0; i < max_planes; ++i)
		{
			if (num_fixed_planes < max_fixed_planes)
			{
				int idx = du(rndEngine, parmu_t{ 0, (int)usable_cluster_planes.size()-1 });

				cur_plane = usable_cluster_planes[idx];
				usable_cluster_planes.erase(usable_cluster_planes.begin() + idx);
				num_fixed_planes++;
			}
			else
			{
				cur_plane = getNeighborPlane(cur_plane, planes); // getManifold(ManifoldType::Plane, anyDirection, planes, 0.0, true, Eigen::Vector3d(0, 0, 0), 0.0, true);
			}

			planes.push_back(cur_plane);
		}
		//std::cout << "planes: " << planes.size() << " of " << max_planes << std::endl;
		
		primitive = createPolytopePrimitive(planes, polytope_center);		
	}
	break;

	case PrimitiveType::Cylinder:
	{
		auto cyl = getManifold(ManifoldType::Cylinder, anyDirection, {}, 0.0, true);
		if (cyl)
		{
			ManifoldSet planes;

			auto numPlanesToSelect = du(rndEngine, parmu_t{ 0, 2 });

			for (int i = 0; i < numPlanesToSelect; ++i)
			{
				auto p = getManifold(ManifoldType::Plane, cyl->n, planes, angleEpsilon);
				if (p)
					planes.push_back(p);
			}
			primitive = createCylinderPrimitive(cyl, planes);
		}
	}
	break;

	case PrimitiveType::Sphere:
	{
		auto sphere = getManifold(ManifoldType::Sphere, anyDirection, {}, 0.0, true);
		if (sphere)
		{
			primitive = createSpherePrimitive(sphere);
		}
	}
	break;
	}

	primitive.cutout = db(rndEngine, parmb_t{ 0.5 });
	//std::cout << "PC0: " << primitive.cutout << std::endl;


	return primitive;
}

lmu::Primitive lmu::PrimitiveSetCreator::mutatePrimitive(const Primitive& p, double angleEpsilon) const
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	static std::bernoulli_distribution db{};
	using parmb_t = decltype(db)::param_type;

	const auto anyDirection = Eigen::Vector3d(0, 0, 0);

	auto primitive = p;

	switch (primitive.type)
	{
	case PrimitiveType::Box:
	{
		// Find a new parallel plane to a randomly chosen plane (parallel planes come in pairs).
		int planePairIdx = du(rndEngine, parmu_t{ 0, 2 }) * 2;
		auto newPlane = getParallelPlane(p.ms[planePairIdx], p.ms, angleEpsilon, minDistanceBetweenParallelPlanes);
		if (newPlane)
		{
			auto newPlanes = ManifoldSet(p.ms);
			newPlanes[planePairIdx + 1] = newPlane;

			primitive = createBoxPrimitive(newPlanes);
		}

		break;
	}

	case PrimitiveType::Polytope:
	{
		int plane_idx = du(rndEngine, parmu_t{ 0, (int)p.ms.size() - 1 });
		auto new_plane = getManifold(ManifoldType::Plane, anyDirection, p.ms, 0.0, true, Eigen::Vector3d(0, 0, 0), 0.0, true, false); 
		if (new_plane)
		{
			auto new_planes = ManifoldSet(p.ms);
			new_planes[plane_idx] = new_plane;

			primitive = createPolytopePrimitive(new_planes, polytope_center);
		}

		break;
	}

	case PrimitiveType::Cylinder:

		ManifoldSet planes;
		auto numPlanesToSelect = du(rndEngine, parmu_t{ 0, 2 });
		auto cyl = p.ms[0]; //First element in manifold set is always the cylinder.
		for (int i = 0; i < numPlanesToSelect; ++i)
		{
			auto m = getManifold(ManifoldType::Plane, cyl->n, planes, angleEpsilon);
			if (m)
				planes.push_back(m);
		}

		primitive = createCylinderPrimitive(cyl, planes);

		break;
	}

	primitive.cutout = db(rndEngine, parmb_t{ 0.5 });
	//std::cout << "PC1: " << primitive.cutout << std::endl;

	return primitive;
}

// ==================== RANKER ====================

lmu::PrimitiveSetRanker::PrimitiveSetRanker(const PointCloud& pc,
	double distanceEpsilon, int maxPrimitiveSetSize, double cell_size, bool allow_cube_cutout, const std::shared_ptr<ModelSDF>& model_sdf,
	double geo_weight, double per_prim_geo_weight, double per_prim_coverage_weight, double size_weight) :
	pc(pc),
	distanceEpsilon(distanceEpsilon),
	cell_size(cell_size),
	model_sdf(model_sdf),
	maxPrimitiveSetSize(maxPrimitiveSetSize),
	allow_cube_cutout(allow_cube_cutout),
	geo_weight(geo_weight),
	per_prim_geo_weight(per_prim_geo_weight),
	per_prim_coverage_weight(per_prim_coverage_weight),
	size_weight(size_weight)
{
}

std::tuple<double, std::vector<double>> lmu::PrimitiveSetRanker::get_geo_score(const lmu::PrimitiveSet& ps) const
{
	const double delta = 0.001;
	int validPoints = 0;
	int checkedPoints = 0;

	std::vector<double> per_polytope_coverage_score(ps.size(), 0.0);

	for (int i = 0; i < pc.rows(); ++i)
	{
		Eigen::Vector3d point = pc.block<1, 3>(i, 0);
		Eigen::Vector3d n = pc.block<1, 3>(i, 3);

		// Get distance to closest primitive. 
		double min_d = std::numeric_limits<double>::max();
		Eigen::Vector3d min_normal;
		int min_d_polytope_idx = -1;

		for (int i = 0; i < ps.size(); ++i)
		{
			if (ps[i].isNone())
				continue;

			auto dg = ps[i].imFunc->signedDistanceAndGradient(point);
			double d = dg[0];

			Eigen::Vector3d g = dg.bottomRows(3);
			//if (p.cutout)
			//	g = -dg.bottomRows(3);

			if (min_d > d)
			{
				min_d = d;
				min_normal = g.normalized();
				min_d_polytope_idx = i;
			}
		}

		//if (std::abs(min_d) < delta && n.dot(min_normal) > 0.9) validPoints++;
		if (std::abs(min_d) < delta)
		{
			validPoints++;

			if (min_d_polytope_idx != -1)
				per_polytope_coverage_score[min_d_polytope_idx] += 1.0;
		}

		checkedPoints++;
	}
	
	for (auto& v : per_polytope_coverage_score)
		v /= (double)checkedPoints;

	return std::make_tuple((double)validPoints / (double)checkedPoints, per_polytope_coverage_score);
}

void iterate_over_prim_volume(const lmu::Primitive& prim, double cell_size, std::function<void(const Eigen::Vector3d&)> f)
{
	static const std::array<std::array<int, 3>, 35> indices =
	{
		std::array<int, 3>({ 0, 1, 2 }),
		std::array<int, 3>({ 0, 1, 3 }),
		std::array<int, 3>({ 0, 1, 4 }),
		std::array<int, 3>({ 0, 1, 5 }),
		std::array<int, 3>({ 0, 1, 6 }),
		std::array<int, 3>({ 0, 2, 3 }),
		std::array<int, 3>({ 0, 2, 4 }),
		std::array<int, 3>({ 0, 2, 5 }),
		std::array<int, 3>({ 0, 2, 6 }),
		std::array<int, 3>({ 0, 3, 4 }),
		std::array<int, 3>({ 0, 3, 5 }),
		std::array<int, 3>({ 0, 3, 6 }),
		std::array<int, 3>({ 0, 4, 5 }),
		std::array<int, 3>({ 0, 4, 6 }),
		std::array<int, 3>({ 0, 5, 6 }),
		std::array<int, 3>({ 1, 2, 3 }),
		std::array<int, 3>({ 1, 2, 4 }),
		std::array<int, 3>({ 1, 2, 5 }),
		std::array<int, 3>({ 1, 2, 6 }),
		std::array<int, 3>({ 1, 3, 4 }),
		std::array<int, 3>({ 1, 3, 5 }),
		std::array<int, 3>({ 1, 3, 6 }),
		std::array<int, 3>({ 1, 4, 5 }),
		std::array<int, 3>({ 1, 4, 6 }),
		std::array<int, 3>({ 1, 5, 6 }),
		std::array<int, 3>({ 2, 3, 4 }),
		std::array<int, 3>({ 2, 3, 5 }),
		std::array<int, 3>({ 2, 3, 6 }),
		std::array<int, 3>({ 2, 4, 5 }),
		std::array<int, 3>({ 2, 4, 6 }),
		std::array<int, 3>({ 2, 5, 6 }),
		std::array<int, 3>({ 3, 4, 5 }),
		std::array<int, 3>({ 3, 4, 6 }),
		std::array<int, 3>({ 3, 5, 6 }),
		std::array<int, 3>({ 4, 5, 6 })
	};

	auto aabb = prim.imFunc->aabb();
	auto max = aabb.c + aabb.s;
	auto min = aabb.c - aabb.s;
	auto vertices = Eigen::Matrix<double, 8, 3>();
	vertices.row(0) << Eigen::Vector3d(min).transpose();
	vertices.row(1) << Eigen::Vector3d(max.x(), min.y(), min.z()).transpose();
	vertices.row(2) << Eigen::Vector3d(min.x(), max.y(), min.z()).transpose();
	vertices.row(3) << Eigen::Vector3d(min.x(), min.y(), max.z()).transpose();
	vertices.row(4) << Eigen::Vector3d(max.x(), max.y(), min.z()).transpose();
	vertices.row(5) << Eigen::Vector3d(max.x(), min.y(), max.z()).transpose();
	vertices.row(6) << Eigen::Vector3d(min.x(), max.y(), max.z()).transpose();
	vertices.row(7) << Eigen::Vector3d(max).transpose();

	std::array<Eigen::RowVector3d, 7> v;
	auto p0 = vertices.row(0).transpose();
	for (int i = 1; i < vertices.rows(); ++i)
		v[i - 1] = vertices.row(i).transpose() - p0;


	// Find axes of coordinate system induced by box vertices.  
	Eigen::Vector3d v0, v1, v2;
	double smallest_sum = std::numeric_limits<double>::max();
	for (const auto& index : indices)
	{	
		double sum =
			std::abs(v[index[0]].dot(v[index[1]])) +
			std::abs(v[index[0]].dot(v[index[2]])) +
			std::abs(v[index[1]].dot(v[index[2]]));

		if (sum < smallest_sum)
		{
			smallest_sum = sum;
			v0 = v[index[0]]; v1 = v[index[1]]; v2 = v[index[2]];
		}
	}

	const double max_len = 2.0;

	double v0_len, v1_len, v2_len;
	v0_len = std::min(v0.norm(), max_len);
	v1_len = std::min(v1.norm(), max_len);
	v2_len = std::min(v2.norm(), max_len);
	
	//std::cout << "len: " << v0_len << " " << v1_len << " " << v2_len << std::endl;

	v0.normalize(); v1.normalize(); v2.normalize();

	double cell_size_x = cell_size;
	double cell_size_y = cell_size;
	double cell_size_z = cell_size;
	bool last_x = false;
	bool last_y = false;
	bool last_z = false;

	for (double x = 0.0;; x += cell_size_x)
	{
		for (double y = 0.0;; y += cell_size_y)
		{
			for (double z = 0.0;; z += cell_size_z)
			{
				// Compute voxel pos in world coordinates.
				Eigen::Vector3d p = p0 + v0 * x + v1 * y + v2 * z;

				//if (prim.imFunc->signedDistance(p) <= 0.0)
					f(p);

				if (last_z)
				{
					last_z = false;
					cell_size_z = cell_size;
					break;
				}
				else if (z + cell_size_z > v2_len)
				{
					last_z = true;
					cell_size_z = v2_len - z;
				}
			}

			if (last_y)
			{
				last_y = false;
				cell_size_y = cell_size;
				break;
			}
			else if (y + cell_size_y > v1_len)
			{
				last_y = true;
				cell_size_y = v1_len - y;
			}
		}

		if (last_x)
		{
			last_x = false;
			cell_size_x = cell_size;
			break;
		}
		else if (x + cell_size_x > v0_len)
		{
			last_x = true;
			cell_size_x = v0_len - x;
		}
	}
}

std::vector<double> lmu::PrimitiveSetRanker::get_per_prim_geo_score(const PrimitiveSet& ps, std::vector<Eigen::Matrix<double, 1, 6>>& points, bool debug) const
{
	std::vector<double> scores;
	for (const auto& prim : ps)
	{
		if (per_prim_geo_weight == 0.0 || prim.isNone())
		{
			scores.push_back(0.0);
			continue;
		}

		double inside_voxels = 0.0;
		double outside_voxels = 0.0;
		int all_voxels = 0;
	
		iterate_over_prim_volume(prim, cell_size, [&prim, this, &inside_voxels, &outside_voxels, &all_voxels, &points](const Eigen::Vector3d& p)
		{
			Eigen::Matrix<double, 1, 6 > pn;

			if (prim.imFunc->signedDistance(p) <= 0)
			{
				auto sdf = model_sdf->sdf_value(p);
								
				if (sdf.d < distanceEpsilon)
				{
					inside_voxels += 1.0;
					pn.row(0) << p.transpose(), 0.0, 1.0, 0;
				}
				else
				{
					outside_voxels -= 1.0;
					pn.row(0) << p.transpose(), 1.0, 0, 0;
				}
				all_voxels++;
			}
			else
			{
				pn.row(0) << p.transpose(), 0.5,0.0,0.5;
			}

			points.push_back(pn);
		});

		const int lower_voxel_bound = 1;
		
		double score = 0.0;

		/*
		if (!allow_cube_cutout && prim.type == PrimitiveType::Box)
		{
			score = all_voxels >= lower_voxel_bound ? inside_voxels / (double)all_voxels : 0.0;
		}
		else
		{
			score = all_voxels >= lower_voxel_bound ? std::max(inside_voxels, outside_voxels) / (double)all_voxels : 0.0;
		}
		*/
		score = all_voxels >= lower_voxel_bound ? inside_voxels / (double)all_voxels : 0.0;

		//std::cout << "S: " << score << " " << all_voxels << std::endl;
		
		scores.push_back(score);
	}

	return scores;
}


lmu::PrimitiveSetRank lmu::PrimitiveSetRanker::rank(const PrimitiveSet& ps) const
{
	if (ps.empty())
		return PrimitiveSetRank::Invalid;

	// Geometry and per primitive coverage score
	auto geo_score = geo_weight == 0.0 && per_prim_coverage_weight == 0.0 ? 
		std::make_tuple(0.0, std::vector<double>(ps.size(), 0.0)) : get_geo_score(ps);

	auto per_prim_coverage_scores = std::get<1>(geo_score);

	// Size score
	double size_score = size_weight == 0.0 ? 0.0 : (double)ps.size() / (double)maxPrimitiveSetSize;

	// Per prim score
	std::vector<Eigen::Matrix<double, 1, 6>> points;
	auto per_prim_geo_scores = get_per_prim_geo_score(ps, points, false);

	auto per_prim_geo_score_sum = std::accumulate(per_prim_geo_scores.begin(), per_prim_geo_scores.end(), 0.0);
	auto per_prim_coverage_score_sum = std::accumulate(per_prim_coverage_scores.begin(), per_prim_coverage_scores.end(), 0.0);

	auto score = PrimitiveSetRank(std::get<0>(geo_score), per_prim_geo_score_sum, per_prim_coverage_score_sum, size_score, 
		0.0 /*computed later*/, per_prim_geo_scores, per_prim_coverage_scores);

	score.capture_score_stats();

	return score;
}

lmu::PrimitiveSetRank lmu::PrimitiveSetRanker::rank(const PrimitiveSet& ps, std::vector<Eigen::Matrix<double, 1, 6>>& debug_points) const
{
	if (ps.empty())
		return PrimitiveSetRank::Invalid;

	// Geometry and per primitive coverage score
	auto geo_score = geo_weight == 0.0 && per_prim_coverage_weight == 0.0 ? 
		std::make_tuple(0.0, std::vector<double>(ps.size(), 0.0)) : get_geo_score(ps);

	auto per_prim_coverage_scores = std::get<1>(geo_score);

	// Size score
	double size_score = size_weight == 0.0 ? 0.0 : (double)ps.size() / (double)maxPrimitiveSetSize;

	// Per prim score
	
	auto per_prim_geo_scores = get_per_prim_geo_score(ps, debug_points, false);

	auto per_prim_geo_score_sum = std::accumulate(per_prim_geo_scores.begin(), per_prim_geo_scores.end(), 0.0);
	auto per_prim_coverage_score_sum = std::accumulate(per_prim_coverage_scores.begin(), per_prim_coverage_scores.end(), 0.0);

	auto score = PrimitiveSetRank(std::get<0>(geo_score), per_prim_geo_score_sum, per_prim_coverage_score_sum, size_score, 
		0.0 /*computed later*/, per_prim_geo_scores, per_prim_coverage_scores);

	score.capture_score_stats();

	return score;
}


std::string lmu::PrimitiveSetRanker::info() const
{
	return std::string();
}

// ==================== Population Manipulator ====================

lmu::PrimitiveSetPopMan::PrimitiveSetPopMan(const PrimitiveSetRanker& ranker, int maxPrimitiveSetSize,
	double geoWeight, double perPrimGeoWeight, double perPrimCoverageWeight, double sizeWeight,
	int num_elite_injections) :
	ranker(&ranker),
	maxPrimitiveSetSize(maxPrimitiveSetSize),
	geoWeight(geoWeight),
	perPrimGeoWeight(perPrimGeoWeight),
	sizeWeight(sizeWeight),
	perPrimCoverageWeight(perPrimCoverageWeight),
	num_elite_injections(num_elite_injections)
{
}

void lmu::PrimitiveSetPopMan::manipulateBeforeRanking(std::vector<RankedCreature<PrimitiveSet, PrimitiveSetRank>>& population) const
{
	std::vector<RankedCreature<PrimitiveSet, PrimitiveSetRank>> filtered_pop;

	// Remove duplicate primitives in each creature. 
	
	for (const auto& c : population)
	{	
		filtered_pop.push_back(RankedCreature<PrimitiveSet, PrimitiveSetRank>(c.creature.without_duplicates(), PrimitiveSetRank()));		
	}

	population = filtered_pop;
	
}

void lmu::PrimitiveSetPopMan::manipulateAfterRanking(std::vector<RankedCreature<PrimitiveSet, PrimitiveSetRank>>& population) const
{
	// Add a primitive set consisting of primitives with best per pri score score.
	if (num_elite_injections > 0)
	{
		int max_primitives = maxPrimitiveSetSize;
		std::vector<std::pair<const Primitive*, double>> all_primitives;
		for (const auto& ps : population)
		{			
			if (ps.rank == PrimitiveSetRank::Invalid)
				continue;
		
			for (int i = 0; i < ps.creature.size(); ++i)
			{
				auto score = perPrimGeoWeight * ps.rank.per_primitive_geo_scores[i] + perPrimCoverageWeight * ps.rank.per_primitive_coverage_scores[i];
				all_primitives.push_back(std::make_pair(&(ps.creature[i]), score));
			}
		}

		if (!all_primitives.empty())
		{
			std::vector<std::pair<const Primitive*, double>> n_best_primitives(std::min(all_primitives.size(), (size_t)maxPrimitiveSetSize));
			std::partial_sort_copy(all_primitives.begin(), all_primitives.end(), n_best_primitives.begin(), n_best_primitives.end(),
				[](const std::pair<const Primitive*, double>& a, const std::pair<const Primitive*, double>& b)
			{
				return a.second > b.second;
			});

			PrimitiveSet best_primitives;
			for (const auto& p : n_best_primitives)
				best_primitives.push_back(*p.first);
		

			auto rank = ranker->rank(best_primitives);

			for(int i = 0; i < num_elite_injections; ++i)
				population.push_back(RankedCreature<PrimitiveSet, PrimitiveSetRank>(best_primitives, rank));
		}
	}


	// Re-normalize scores and compute combined score. 
	
	PrimitiveSetRank max_r(-std::numeric_limits<double>::max()), min_r(std::numeric_limits<double>::max());
	for (auto& ps : population)
	{
		if (ps.rank.geo < 0.0 || ps.rank.size < 0.0)
			continue;

		
		max_r.geo = max_r.geo < ps.rank.geo ? ps.rank.geo : max_r.geo;
		max_r.per_prim_geo_sum = max_r.per_prim_geo_sum < ps.rank.per_prim_geo_sum ? ps.rank.per_prim_geo_sum : max_r.per_prim_geo_sum;
		max_r.per_prim_coverage_sum = max_r.per_prim_coverage_sum < ps.rank.per_prim_coverage_sum ? ps.rank.per_prim_coverage_sum : max_r.per_prim_coverage_sum;
		max_r.size = max_r.size < ps.rank.size ? ps.rank.size : max_r.size;

		min_r.geo = min_r.geo > ps.rank.geo ? ps.rank.geo : min_r.geo;
		min_r.per_prim_geo_sum = min_r.per_prim_geo_sum > ps.rank.per_prim_geo_sum ? ps.rank.per_prim_geo_sum : min_r.per_prim_geo_sum;
		min_r.per_prim_coverage_sum = min_r.per_prim_coverage_sum > ps.rank.per_prim_coverage_sum ? ps.rank.per_prim_coverage_sum : min_r.per_prim_coverage_sum;
		min_r.size = min_r.size > ps.rank.size ? ps.rank.size : min_r.size;
			
	}
	auto diff_r = max_r - min_r;

	//std::cout << "DIFF: " << diff_r << std::endl;
	//std::cout << "MAX: " << max_r << std::endl;
	//std::cout << "MIN: " << min_r << std::endl;
	

	for (auto& ps : population)
	{
		//std::cout << "Rank Before: " << ps.rank << std::endl;

		// Un-normalized 
		//ps.rank.geo = std::max(0.0, ps.rank.geo);
		//ps.rank.per_prim_geo_sum = std::max(0.0, ps.rank.per_prim_geo_sum);
		//ps.rank.size = std::max(0.0, ps.rank.size);
		
		// Normalized
		ps.rank.geo              = ps.rank.geo < 0.0 || diff_r.geo == 0.0 ? 0.0 : (ps.rank.geo - min_r.geo) / diff_r.geo;
		ps.rank.per_prim_geo_sum = ps.rank.per_prim_geo_sum < 0.0 || diff_r.per_prim_geo_sum == 0.0 ? 0.0 : (ps.rank.per_prim_geo_sum - min_r.per_prim_geo_sum) / diff_r.per_prim_geo_sum;
		ps.rank.per_prim_coverage_sum = ps.rank.per_prim_coverage_sum < 0.0 || diff_r.per_prim_coverage_sum == 0.0 ? 0.0 : (ps.rank.per_prim_coverage_sum - min_r.per_prim_coverage_sum) / diff_r.per_prim_coverage_sum;
		ps.rank.size             = ps.rank.size < 0.0 || diff_r.size == 0.0 ? 0.0 : (ps.rank.size - min_r.size) / diff_r.size;

		ps.rank.combined = ps.rank.geo * geoWeight + ps.rank.per_prim_geo_sum * perPrimGeoWeight + ps.rank.per_prim_coverage_sum * perPrimCoverageWeight - ps.rank.size * sizeWeight;
		
		
		//std::cout << "Combined: " << ps.rank.combined << std::endl;
		
		/*
		std::cout << "Rank: geo: " << ps.rank.geo << " per prim geo sum: " << ps.rank.per_prim_geo_sum << " size: " << ps.rank.size << std::endl;
		std::cout << "Rank: " << ps.rank << std::endl;
		std::cout << "Weights: geo: " << geoWeight << " per prim geo sum: " << perPrimGeoWeight << " size: " << sizeWeight << std::endl;
		std::cout << "----" << std::endl;
		*/
	}
}

std::string lmu::PrimitiveSetPopMan::info() const
{
	return std::string();
}

bool mesh_out_of_range(const lmu::Mesh& mesh)
{
	Eigen::Vector3d min = mesh.vertices.colwise().minCoeff();
	Eigen::Vector3d max = mesh.vertices.colwise().maxCoeff();

	return (max - min).norm() > std::sqrt(3.0); //TODO make this variable.
}

// Find normal to point-cloud by majority voting
Eigen::Vector3d findNormalByMajority(const lmu::PointCloud& pc)
{
	Eigen::Vector3d n1(0.0, 0.0, 0.0);
	int count1 = 0;
	Eigen::Vector3d n2(0.0, 0.0, 0.0);
	int count2 = 0;

	int np = pc.rows();

	if (np == 0) {
		std::cout << "Empty point-cloud" << std::endl;
		return n1;
	}

	n1[0] = pc(0, 3);
	n1[1] = pc(0, 4);
	n1[2] = pc(0, 5);
	count1++;

	for (int i = 1; i < np; ++i) {
		Eigen::Vector3d cn(pc(i, 3), pc(i, 4), pc(i, 5));
		double d = n1.dot(cn);
		if (d < 0) {
			if (!(n2[0] == 0.0 && n2[1] == 0.0 && n2[2] == 0.0)) {
				n2[0] = pc(i, 3);
				n2[1] = pc(i, 4);
				n2[2] = pc(i, 5);
			}
			count2++;
		}
		else {
			count1++;
		}
	}

	if (count1 > count2) {
		return n1;
	}
	else {
		return n2;
	}
}

lmu::Primitive lmu::createPolytopePrimitive(const ManifoldSet& planes, const Eigen::Vector3d& cluster_center)
{
	if (planes.size() < 4)
	{
		return Primitive::None();
	}
		
	std::vector<Eigen::Vector3d> p;
	std::vector<Eigen::Vector3d> n;

	static std::bernoulli_distribution db{};
	using parmb_t = decltype(db)::param_type;

	/*
	for (int i = 0; i < planes.size(); ++i)
	{
		auto new_plane = std::make_shared<Manifold>(*planes[i]);
		
		if (db(rndEngine(), parmb_t{ 0.5 }))
		{
			new_plane->n = -1.0 * new_plane->n;
		}

		n.push_back(new_plane->n);
		p.push_back(new_plane->p);
	}
	*/

	/*
	for (int i = 0; i < planes.size(); ++i)
	{
		auto new_plane = std::make_shared<Manifold>(*planes[i]);

		// Find the normal to the plane by majority voting
		Eigen::Vector3d nm = findNormalByMajority(planes[i]->pc);

		// Flip plane normal if it disagrees with the point-cloud normal
		double d = nm.dot(new_plane->n);
		if (d < 0.0)
		{
			new_plane->n = -1.0 * new_plane->n;
		}

		n.push_back(new_plane->n);
		p.push_back(new_plane->p);
	}*/

	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;


	if (db(rndEngine(), parmb_t{ 0.5 })) // Here is the problem! Maybe we need a better plane normal estimation.
	{	
		for (const auto& plane : planes)
		{
			int point_idx = du(rndEngine(), parmu_t{ 0, (int)plane->pc.rows()-1 });
						
			Eigen::Vector3d point_n(plane->pc.row(point_idx).rightCols(3));
			n.push_back(point_n);
			p.push_back(plane->p);			
		}
	}
	else
	{

		// Get point that is guaranteed to be inside of the polytope. 
		// The point is the center of all points stemming from pointclouds of the plane manifolds (but it is enough to just take a single point per plane point cloud).


		for (int i = 0; i < planes.size(); ++i)
		{
			auto new_plane = std::make_shared<Manifold>(*planes[i]);

			// Flip plane normal if inside_point would be outside.
			double d = (cluster_center - new_plane->p).dot(new_plane->n);
			if (d > 0.0)
			{
				new_plane->n = -1.0 * new_plane->n;
			}

			n.push_back(new_plane->n);
			p.push_back(new_plane->p);
		}
	}
	
	auto polytope = std::make_shared<IFPolytope>(Eigen::Affine3d::Identity(), p, n, "");

	if (polytope->empty() || mesh_out_of_range(polytope->meshCRef()))
	{
		return Primitive::None();
	}

	return Primitive(polytope, planes, PrimitiveType::Polytope);
}

lmu::Primitive lmu::createBoxPrimitive(const ManifoldSet& planes)
{
	bool strictlyParallel = false;

	if (planes.size() != 6)
	{
		return Primitive::None();
	}

	std::vector<Eigen::Vector3d> p;
	std::vector<Eigen::Vector3d> n;
	ManifoldSet ms;
	for (int i = 0; i < planes.size() / 2; ++i)
	{
		auto newPlane1 = std::make_shared<Manifold>(*planes[i * 2]);
		auto newPlane2 = std::make_shared<Manifold>(*planes[i * 2 + 1]);

		Eigen::Vector3d p1 = newPlane1->p;
		Eigen::Vector3d n1 = newPlane1->n;
		Eigen::Vector3d p2 = newPlane2->p;
		Eigen::Vector3d n2 = newPlane2->n;

		// Check plane orientation and correct if necessary.
		double d1 = (p2 - p1).dot(n2) / n1.dot(n2);
		double d2 = (p1 - p2).dot(n1) / n2.dot(n1);
		if (d1 >= 0.0)
			newPlane1->n = newPlane1->n * -1.0;
		if (d2 >= 0.0)
			newPlane2->n = newPlane2->n * -1.0;

		ms.push_back(newPlane1);
		ms.push_back(newPlane2);

		n.push_back(newPlane1->n);

		if (strictlyParallel)
			n.push_back(newPlane1->n * -1.0);
		else
			n.push_back(newPlane2->n);

		p.push_back(newPlane1->p);
		p.push_back(newPlane2->p);
	}

	auto box = std::make_shared<IFPolytope>(Eigen::Affine3d::Identity(), p, n, "");
	if (box->empty() || mesh_out_of_range(box->meshCRef()))
	{
		return Primitive::None();
	}

	return Primitive(box, ms, PrimitiveType::Box);
}

lmu::Primitive lmu::createSpherePrimitive(const lmu::ManifoldPtr& m)
{
	if (!m || m->type != ManifoldType::Sphere)
		return Primitive::None();

	Eigen::Affine3d t = Eigen::Affine3d::Identity();
	t.translate(m->p);

	auto sphereIF = std::make_shared<IFSphere>(t, m->r.x(), ""); //TODO: Add name.

	return Primitive(sphereIF, { m }, PrimitiveType::Sphere);
}

lmu::Primitive lmu::createCylinderPrimitive(const ManifoldPtr& m, ManifoldSet& planes)
{
	double const height_epsilon = 0.0;

	switch (planes.size())
	{
	case 1: //estimate the second plane and go on as if there existed two planes.		
		planes.push_back(lmu::estimateSecondCylinderPlaneFromPointCloud(*m, *planes[0]));
	case 2:
	{
		// Cylinder height is distance between both parallel planes.
		// double height = std::abs((planes[0]->p - planes[1]->p).dot(planes[0]->n));

		// Get intersection points of cylinder ray and plane 0 and 1.		
		Eigen::Vector3d p0 = planes[0]->p;
		Eigen::Vector3d l0 = m->p;
		Eigen::Vector3d l = m->n;
		Eigen::Vector3d n = planes[0]->n;

		double d = (p0 - l0).dot(n) / l.dot(n);
		Eigen::Vector3d i0 = d * l + l0;

		p0 = planes[1]->p;
		n = planes[1]->n;
		d = (p0 - l0).dot(n) / l.dot(n);
		Eigen::Vector3d i1 = d * l + l0;

		double height = (i0 - i1).norm() + height_epsilon;
		Eigen::Vector3d pos = i0 + (0.5 * (i1 - i0));

		// Compute cylinder transform.
		Eigen::Matrix3d rot = getRotationMatrix(m->n);
		Eigen::Affine3d t = (Eigen::Affine3d)(Eigen::Translation3d(pos) * rot);

		// Create primitive. 
		auto cylinderIF = std::make_shared<IFCylinder>(t, m->r.x(), height, "");

		return Primitive(cylinderIF, { m, planes[0], planes[1] }, PrimitiveType::Cylinder);
	}
	case 0:	//Estimate cylinder height and center position using the point cloud only since no planes exist.
	{
		auto height = lmu::estimateCylinderHeightFromPointCloud(*m) + height_epsilon;
		auto pos = m->p;
		std::cout << "POS: " << m->p << std::endl;
		Eigen::Matrix3d rot = getRotationMatrix(m->n);
		Eigen::Affine3d t = (Eigen::Affine3d)(Eigen::Translation3d(pos) * rot);

		auto cylinderIF = std::make_shared<IFCylinder>(t, m->r.x(), height, "");

		return Primitive(cylinderIF, { m }, PrimitiveType::Cylinder);
	}
	default:
		return Primitive::None();
	}
}

double lmu::estimateCylinderHeightFromPointCloud(const Manifold& m)
{
	// Get matrix for transform to identity rotation.
	double min_t = std::numeric_limits<double>::max();
	double max_t = -std::numeric_limits<double>::max();

	for (int i = 0; i < m.pc.rows(); ++i)
	{
		Eigen::Vector3d p = m.pc.row(i).leftCols(3).transpose();
		Eigen::Vector3d a = m.p;
		Eigen::Vector3d ab = m.n;
		Eigen::Vector3d ap = p - a;

		//A + dot(AP, AB) / dot(AB, AB) * AB
		Eigen::Vector3d proj_p = a + ap.dot(ab) / ab.dot(ab) * ab;

		// proj_p = m.p + m.n *t 
		double t = (proj_p.x() - m.p.x()) / m.n.x();

		min_t = t < min_t ? t : min_t;
		max_t = t > max_t ? t : max_t;
	}

	Eigen::Vector3d min_p = m.p + m.n * min_t;
	Eigen::Vector3d max_p = m.p + m.n * max_t;

	return (max_p - min_p).norm();
}

lmu::ManifoldPtr lmu::estimateSecondCylinderPlaneFromPointCloud(const Manifold& m, const Manifold& firstPlane)
{
	Eigen::Vector3d minPos = (Eigen::Vector3d)(m.pc.leftCols(3).colwise().minCoeff());
	Eigen::Vector3d maxPos = (Eigen::Vector3d)(m.pc.leftCols(3).colwise().maxCoeff());

	//Take the point of the point cloud's min-max points which is farer away from the first plane as the second plane's point.
	Eigen::Vector3d p = (firstPlane.p - minPos).norm() > (firstPlane.p - maxPos).norm() ? minPos : maxPos;

	auto secondPlane =
		std::make_shared<Manifold>(ManifoldType::Plane, p, -firstPlane.n, Eigen::Vector3d(0, 0, 0), PointCloud());

	return secondPlane;
}


lmu::ModelSDF::ModelSDF(const PointCloud& pc, double voxel_size, std::ofstream& s) :
	data(nullptr),
	voxel_size(voxel_size)
{
	const double border_factor = 4.0;

	Eigen::Vector3d border(voxel_size, voxel_size, voxel_size);
	border *= border_factor;

	Eigen::Vector3d dims = computeAABBDims(pc) + border * 2.0;
	origin = Eigen::Vector3d(pc.leftCols(3).colwise().minCoeff()) - border;

	std::cout << "Border: " << border.transpose() << std::endl;
	std::cout << "Dims: " << dims.transpose() << std::endl;
	std::cout << "Voxel size: " << voxel_size << std::endl;

	grid_size = Eigen::Vector3i(std::ceil(dims.x() / voxel_size), std::ceil(dims.y() / voxel_size), std::ceil(dims.z() / voxel_size));
	if (grid_size.x() > 1000 || grid_size.y() > 1000 || grid_size.z() > 1000)
		std::cout << "Too large model size: " << grid_size.transpose() << std::endl;
	std::cout << "Grid size: " << grid_size.transpose() << std::endl;

	n = grid_size.x() * grid_size.y() * grid_size.z();

	size = Eigen::Vector3d((double)grid_size.x() * voxel_size, (double)grid_size.y() * voxel_size, (double)grid_size.z() * voxel_size);
	
	std::cout << "Create mesh" << std::endl;

	lmu::TimeTicker t;

	t.tick();
	auto mesh = lmu::createFromPointCloud(pc);
	//s << "Poisson Reconstruction=" << t.tick() << std::endl;

	if (!mesh.empty())
	{
		recreate_from_mesh(mesh);
		s << "SDF Creation=" << t.tick() << std::endl;
	}
	else
	{
		throw std::runtime_error("Unable to create ModelSDF. Poisson mesh is empty.");
	}
}

lmu::ModelSDF::~ModelSDF()
{
	delete[] data;
}

void lmu::ModelSDF::recreate_from_mesh(const Mesh& m)
{
	delete[] data;
	data = new SDFValue[n];
	
	surface_mesh = m;

	tree.init(surface_mesh.vertices, surface_mesh.indices);

	igl::per_face_normals(surface_mesh.vertices, surface_mesh.indices, fn);
	igl::per_vertex_normals(surface_mesh.vertices, surface_mesh.indices, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, fn, vn);
	igl::per_edge_normals(surface_mesh.vertices, surface_mesh.indices, igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM, fn, en, e, emap);

	std::cout << "Fill with signed distance values. " << std::endl;
	std::cout << "N: " << n << std::endl;
	std::cout << "GridSize: " << grid_size.transpose() << std::endl;

	Eigen::MatrixXd points(n, 3);
	int idx = 0;
	for (int x = 0; x < grid_size.x(); ++x)
	{
		for (int y = 0; y < grid_size.y(); ++y)
		{
			for (int z = 0; z < grid_size.z(); ++z)
			{
				int idx = x + grid_size.x() * y + grid_size.x() * grid_size.y() * z;

				Eigen::Vector3d p = Eigen::Vector3d(x, y, z) * voxel_size + origin;

				points.row(idx++) << p.transpose();
			}
		}
	}

	Eigen::VectorXd d;
	Eigen::VectorXi i;
	Eigen::MatrixXd norm, c;

	std::cout << "Get sd and normal. ";

	igl::signed_distance_pseudonormal(points, surface_mesh.vertices, surface_mesh.indices, tree, fn, vn, en, emap, d, i, c, norm);

	std::cout << "Done." << std::endl;

	for (int j = 0; j < n; ++j)
	{
		float sd = d.coeff(j, 0);
		Eigen::Vector3f n = norm.row(j).transpose().cast<float>();

		data[j] = SDFValue(sd, n);
	}

	std::cout << "Mesh re-creation done." << std::endl;
}

lmu::SDFValue lmu::ModelSDF::sdf_value(const Eigen::Vector3d& p) const
{
	// Eigen::Vector3i p_int = ((p - origin) / voxel_size).array().round().cast<int>();
	Eigen::Vector3i p_int = ((p - origin) / voxel_size).cast<int>();

	int idx = p_int.x() + grid_size.x() * p_int.y() + grid_size.x() * grid_size.y() * p_int.z();

	return 
		idx < n &&
		p.x() >= origin.x() && p.x() < origin.x() + size.x() &&
		p.y() >= origin.y() && p.y() < origin.y() + size.y() &&
		p.z() >= origin.z() && p.z() < origin.z() + size.z()
		? data[idx] : SDFValue(std::numeric_limits<double>::max(), Eigen::Vector3f());
}

double lmu::ModelSDF::distance(const Eigen::Vector3d& p) const
{
	return sdf_value(p).d;
}

#include <igl/copyleft/marching_cubes.h>

lmu::Mesh lmu::ModelSDF::to_mesh() const
{
	int num = grid_size.x()* grid_size.y()* grid_size.z();
	Eigen::MatrixXd sampling_points(num, 3);
	Eigen::VectorXd sampling_values(num);

	for (int x = 0; x < grid_size.x(); ++x)
	{
		for (int y = 0; y < grid_size.y(); ++y)
		{
			for (int z = 0; z < grid_size.z(); ++z)
			{
				int idx = x + grid_size.x() * y + grid_size.x() * grid_size.y() * z;

				Eigen::Vector3d p = Eigen::Vector3d(x, y, z) * voxel_size + origin;

				sampling_points.row(idx) = p;
				sampling_values(idx) = data[idx].d;
			}
		}
	}

	Mesh mesh;
	igl::copyleft::marching_cubes(sampling_values, sampling_points, grid_size.x(), grid_size.y(), grid_size.z(), mesh.vertices, mesh.indices);
	return mesh;
}

lmu::PointCloud lmu::ModelSDF::to_pc() const
{
	std::vector<Eigen::Matrix<double, 1, 6>> points;

	for (int x = 0; x < grid_size.x(); ++x)
	{
		for (int y = 0; y < grid_size.y(); ++y)
		{
			for (int z = 0; z < grid_size.z(); ++z)
			{
				int idx = x + grid_size.x() * y + grid_size.x() * grid_size.y() * z;

				Eigen::Vector3d p = Eigen::Vector3d(x, y, z) * voxel_size + origin;

				auto v = sdf_value(p);

				if ( v.d != -1.0/* && std::abs(v.d) < voxel_size*/)
				{
					//std::cout << " " << v.v;
					Eigen::Matrix<double, 1, 6> point;
					point << p.transpose(), v.n.transpose().cast<double>(); //std::abs(v.d) * 1.0, std::abs(v.d) * 1.0, std::abs(v.d) * 1.0;
					points.push_back(point);
				}
			}
		}
	}

	return pointCloudFromVector(points);
}

lmu::DHType lmu::ModelSDF::get_dh_type(const Primitive & p, double t_inside, double t_outside, double v_size, std::vector<Eigen::Matrix<double, 1, 6>>& debug_points, bool debug) const
{
	if (v_size == 0.0)
		v_size = voxel_size;

	std::vector<Eigen::Vector3d> points; 
	iterate_over_prim_volume(p, v_size, [&points](const Eigen::Vector3d& p) { if(!std::isnan(p.x())) points.push_back(p); });


	Eigen::MatrixXd point_mat(points.size(), 3);
	for (int i = 0; i < points.size(); ++i)
	{
		point_mat.row(i) << points[i].transpose();
	}

	
	std::cout << "Points: " << points.size() << std::endl;

	Eigen::VectorXd d;
	Eigen::VectorXi i;
	Eigen::MatrixXd norm, c;

	igl::signed_distance_pseudonormal(point_mat, surface_mesh.vertices, surface_mesh.indices, tree, fn, vn, en, emap, d, i, c, norm);

	//std::cout << "0" << std::endl;

	int num_inside_points = 0;
	int num_points = 0;

	for (int i = 0; i < points.size(); ++i)
	{

		Eigen::Matrix<double, 1, 6 > pn;

		auto dg = p.imFunc->signedDistanceAndGradient(points[i]);

		Eigen::Vector3d mesh_n = norm.row(i).transpose();
		Eigen::Vector3d prim_n = dg.bottomRows(3);
		double mesh_sd = d.coeff(i, 0);
		double prim_sd = dg.x();

		if (prim_sd <= 0.0)
		{
			if (mesh_sd <= v_size && mesh_n.dot(prim_n) >= 0.0)
			{
				num_inside_points++;

				pn.row(0) << points[i].transpose(), 0.0, 1.0, 0.0;
			}
			else
			{
				pn.row(0) << points[i].transpose(), 1.0, 0.0, 0.0;
			}

			debug_points.push_back(pn);

			num_points++;
		}
	}

	std::cout << p.imFunc->name() << std::endl;

	double inside_score = num_points > 0 ? (double)num_inside_points / (double)num_points : 0;
	std::cout << "inside score: " << inside_score << " points: " << num_points << std::endl;

	auto geo = geometry(p.imFunc);

	bool inside = inside_score >= t_inside;

	auto type =  inside_score >= t_inside ? DHType::INSIDE : inside_score <= t_outside ? DHType::OUTSIDE : DHType::NONE;

	std::cout << "t inside: " << t_inside << " t outside: " << t_outside << std::endl;
	std::cout << "type: " << type << std::endl;

	return type;
}

inline lmu::SDFValue::SDFValue() :
	SDFValue(0.0, Eigen::Vector3f(0.0,0.0,0.0))
{
}

inline lmu::SDFValue::SDFValue(float d, Eigen::Vector3f& n) :
	d(d),
	n(n)
{
}

const float lmu::SDFValue::max_distance{16.0f};


#ifdef _DEBUG
#undef _DEBUG
#include <python.h>
#define _DEBUG
#else
#include <python.h>
#endif

lmu::OutlierDetector::OutlierDetector(const std::string & python_module_path)
{
	Py_Initialize();

	od_method_name = PyUnicode_FromString((char*)"outlier_detection");
	
	PyRun_SimpleString(("import sys\nsys.path.append('" + python_module_path + "')").c_str());

	od_module = PyImport_Import(od_method_name);
	od_dict = PyModule_GetDict(od_module);
	od_method = PyDict_GetItemString(od_dict, (char*)"detect_outliers");
}

lmu::OutlierDetector::~OutlierDetector()
{
	Py_DECREF(od_module);
	Py_DECREF(od_method_name);

	Py_Finalize();
}

lmu::PrimitiveSet lmu::OutlierDetector::remove_outliers(const PrimitiveSet& ps, const PrimitiveSetRank& psr) const
{
	if (ps.size() <= 1)
		return ps;

	PyObject *arg, *res;

	std::ostringstream oss;
	for (auto v : psr.per_primitive_geo_scores)
	{
		oss << v << ";";
	}
	auto scores = oss.str();
	scores = scores.substr(0, scores.size() - 1);

	std::cout << "VALUES: " << scores << std::endl;

	if (PyCallable_Check(od_method))
	{
		arg = Py_BuildValue("(z)", (char*)scores.c_str());
		PyErr_Print();
		res = PyObject_CallObject(od_method, arg);
		PyErr_Print();
	}
	else
	{
		PyErr_Print();
	}

	std::string res_str = PyUnicode_AsUTF8(res);

	std::stringstream ss(res_str);

	std::vector<int> outliers;
	for (int i; ss >> i;) {
		outliers.push_back(i);
		if (ss.peek() == ';')
			ss.ignore();
	}
	std::cout << "OUTLIERS: ";
	for (int i : outliers)
		std::cout << " " << i;
	std::cout << std::endl;

	PrimitiveSet filtered_ps;
	for (int i = 0; i < ps.size(); ++i)
		if (outliers[i] == 0)
		{
			filtered_ps.push_back(ps[i]);
		}
		else
		{
			std::cout << "Filtered Primitive at " << i << std::endl;
		}

	Py_DECREF(res);

	return filtered_ps;
}

lmu::ThresholdOutlierDetector::ThresholdOutlierDetector(double threshold) : 
	threshold(threshold)
{
}

lmu::PrimitiveSet lmu::ThresholdOutlierDetector::remove_outliers(const PrimitiveSet& ps, const PrimitiveSetRanker& ranker) const
{
	auto psr = ranker.rank(ps);

	PrimitiveSet filtered_ps;
	for (int i = 0; i < ps.size(); ++i)
	if (psr.per_primitive_geo_scores[i] >= threshold)
	{
		filtered_ps.push_back(ps[i]);
	}
	else
	{
		std::cout << "Filtered Primitive at " << i << ". Type: " << primitiveTypeToString(ps[i].type) << " Score: " << psr.per_primitive_geo_scores[i] << " Threshold: " << threshold << std::endl;
	}

	return filtered_ps;
}

#include "optimizer_red.h"

lmu::SimilarityFilter::SimilarityFilter(double epsilon, double voxel_size, bool similarity_only, double perfectness_t) :
	epsilon(epsilon),
	voxel_size(voxel_size),
	similarity_only(similarity_only),
	perfectness_t(perfectness_t)
{
}

bool fully_contains(const lmu::Primitive& p1, const lmu::Primitive& p2, double voxel_size, double epsilon)
{
	int num_points = 0;
	int num_outside_points = 0;
	bool contains = true;
	iterate_over_prim_volume(p1, voxel_size, [&p1, &p2, &contains, &num_points, &num_outside_points, epsilon](const Eigen::Vector3d& p) {
		if (p1.imFunc->signedDistance(p) < 0.0 && p2.imFunc->signedDistance(p) > epsilon )
		{
			contains = false;
			num_outside_points++;
		}
		num_points++;
	});

	std::cout << "(" << num_outside_points << "|" << num_points <<  ")";
	
	return contains;
}

bool are_similar_or_contain_one_another(const lmu::Primitive& p1, const lmu::Primitive& p2, double voxel_size, const lmu::PrimitiveSetRanker& ranker, double perfectness_t, double epsilon)
{
	static lmu::EmptySetLookup esLookup;

	// Check if p2 is perfect.
	std::vector<Eigen::Matrix<double, 1, 6>> pts;
	lmu::PrimitiveSet ps;
	ps.push_back(p2);
	double score = ranker.get_per_prim_geo_score(ps, pts)[0];
	bool container_is_perfect_primitive = score > perfectness_t;

	if (fully_contains(p1, p2, voxel_size, epsilon))
	{
		if (container_is_perfect_primitive)
		{
			std::cout << p2.imFunc->name() << " is perfect and fully contains " << p1.imFunc->name() << std::endl;
			return true;
		}
		else if(fully_contains(p2, p1, voxel_size, epsilon))
		{
			std::cout << p2.imFunc->name() << " is equal to " << p1.imFunc->name() << std::endl;
			return true;
		}
		else
		{
			std::cout << p2.imFunc->name() << " is NOT equal to " << p1.imFunc->name() << std::endl;
		}
	}
	else
	{ 
		std::cout << p2.imFunc->name() << " does not fully contain " << p1.imFunc->name() << std::endl;
	}
	
	return false;
}

lmu::PrimitiveSet lmu::SimilarityFilter::filter(const PrimitiveSet& ps, const PrimitiveSetRanker& ranker)
{
	PrimitiveSet filtered_prims;
	std::set<int> already_removed_indices;

	for (int i = 0; i < ps.size(); ++i)
	{
		bool add = true;

		std::cout << ps[i].imFunc->name() << ": " << std::endl;

		for(int j = 0; j < ps.size(); ++j)
		{
			if (i == j || already_removed_indices.find(j) != already_removed_indices.end()) continue;

			if (are_similar_or_contain_one_another(ps[i], ps[j], voxel_size, ranker, perfectness_t, epsilon))
			{
				std::cout << "filtered redundant primitive " << ps[i].imFunc->name() << std::endl;
				already_removed_indices.insert(i);
				add = false;
				break;
			}
		}
		if (add)
		{
			filtered_prims.push_back(ps[i]);
		}

		std::cout << std::endl;
	}
	
	return filtered_prims;
}

void lmu::PrimitiveSetRank::capture_score_stats()
{
	per_primitive_mean_score = accumulate(per_primitive_geo_scores.begin(), per_primitive_geo_scores.end(), 0.0) / (double)per_primitive_geo_scores.size();
	geo_unnormalized = geo;
	size_unnormalized = size;
}

lmu::CapOptimizer::CapOptimizer(double cap_plane_adjustment_max_dist) : 
	cap_plane_adjustment_max_dist(cap_plane_adjustment_max_dist)
{
}

lmu::CSGNode lmu::CapOptimizer::optimize_caps(const PrimitiveSet& ps, const CSGNode& inp_node)
{
	auto res_node = inp_node;

	// Collect planes.
	std::vector<std::shared_ptr<IFPlane>> planes;
	for (const auto& prim : ps)
	{
		std::vector<Eigen::Vector3d> _p;
		std::vector<Eigen::Vector3d> _n;

		if (prim.type == PrimitiveType::Polytope || prim.type == PrimitiveType::Box)
		{
			auto poly = dynamic_cast<IFPolytope*>(prim.imFunc.get());
			auto p_n = poly->n();
			auto p_p = poly->p();
			_n.insert(_n.end(), p_n.begin(), p_n.end());
			_p.insert(_p.end(), p_p.begin(), p_p.end());
		}
		else if (prim.type == PrimitiveType::Cylinder)
		{
			for (const auto m : prim.ms)
			{
				if (m->type == ManifoldType::Plane)
				{
					_n.push_back(m->n);
					_p.push_back(m->p);
				}
			}
		}
		
		for (int i = 0; i < _n.size(); ++i)
		{			
			auto plane = std::make_shared<IFPlane>(_p[i], _n[i], "");
			planes.push_back(plane);
		}
	}

	// Find closest planes for capped primitives.
	for (const auto& p : ps)
	{
		if (p.type == PrimitiveType::Cylinder)
		{
			std::shared_ptr<IFPlane> closest_top_plane;
			std::shared_ptr<IFPlane> closest_bottom_plane;
			double closest_top_plane_d = std::numeric_limits<double>::max();
			double closest_bottom_plane_d = std::numeric_limits<double>::max();
			
			auto cyl = dynamic_cast<IFCylinder*>(p.imFunc.get());
			const auto max_height = 2.0;

			// point on the top and bottom of the cylinder.
			Eigen::Vector3d bottom_p = cyl->transform() * Eigen::Vector3d(0, -cyl->height() / 2, 0);
			Eigen::Vector3d top_p = cyl->transform() * Eigen::Vector3d(0, cyl->height() / 2, 0);
					
			// find suitable bottom and top cap planes.
			for (const auto& plane : planes)
			{	
				auto bottom_plane = plane->signedDistanceAndGradient(bottom_p);
				auto top_plane = plane->signedDistanceAndGradient(top_p);

				auto bottom_cyl = cyl->signedDistanceAndGradient(bottom_p);
				auto top_cyl = cyl->signedDistanceAndGradient(top_p);

				auto bottom_d = std::abs(bottom_plane.x());
				auto top_d = std::abs(top_plane.x());

				if (closest_bottom_plane_d > bottom_d)// && bottom_plane.bottomRows(3).dot(bottom_cyl.bottomRows(3)) > 0.0)
				{
					closest_bottom_plane_d = bottom_d;
					closest_bottom_plane = plane;
				}

				if (closest_top_plane_d > top_d)// && top_plane.bottomRows(3).dot(top_cyl.bottomRows(3)) > 0.0)
				{
					closest_top_plane_d = top_d;
					closest_top_plane = plane;

					//std::cout << " topd: " << top_d;
				}
				//std::cout << std::endl;
			}

			std::cout << "closest top d: " << closest_top_plane_d << " closest bottom d: " << closest_bottom_plane_d << std::endl;

			if (!closest_top_plane || !closest_bottom_plane)
			{
				std::cout << "Could not find better top and bottom plane for primitive." << std::endl;
				continue;
			}
			if (closest_top_plane.get() == closest_bottom_plane.get())
			{
				std::cout << "Both planes are the same which is not possible." << std::endl;
				continue;
			}

			if (closest_top_plane_d > cap_plane_adjustment_max_dist && closest_bottom_plane_d > cap_plane_adjustment_max_dist)
			{
				std::cout << "Both planes are farther away than allowed (allowed: " << cap_plane_adjustment_max_dist << ")." << std::endl;
				continue;
			}
			else if (closest_top_plane_d > cap_plane_adjustment_max_dist)
			{
				auto top_cyl = cyl->signedDistanceAndGradient(top_p);
				auto top_n = Eigen::Vector3d(top_cyl.bottomRows(3)).normalized();
				closest_top_plane = std::make_shared<IFPlane>(top_p, top_n, cyl->name() + "_Top");
			}
			else if (closest_bottom_plane_d > cap_plane_adjustment_max_dist)
			{
				auto bottom_cyl = cyl->signedDistanceAndGradient(bottom_p);
				auto bottom_n = Eigen::Vector3d(bottom_cyl.bottomRows(3)).normalized();
				closest_bottom_plane = std::make_shared<IFPlane>(bottom_p, bottom_n, cyl->name() + "_Bottom");
			}


			lmu::visit(res_node, [&closest_top_plane, &closest_bottom_plane, cyl, max_height](CSGNode& n) 
			{
				if (n.function().get() == cyl)
				{
					Eigen::Vector3d cyl_center = cyl->transform() * Eigen::Vector3d(0, 0, 0);
					
					// Check if normal needs to be flipped.
					Eigen::Vector3d top_n = closest_top_plane->n() * (closest_top_plane->signedDistance(cyl_center) < 0.0 ? -1.0 : 1.0);
					Eigen::Vector3d bottom_n = closest_bottom_plane->n() * (closest_bottom_plane->signedDistance(cyl_center) < 0.0 ? -1.0 : 1.0);

					//std::cout << "cyl center: " << cyl_center.transpose() << std::endl;
					//std::cout << top_p.transpose() << " | " << bottom_p.transpose() << std::endl;
					//std::cout << top_n.transpose() << " | " << bottom_n.transpose() << std::endl;
					//std::cout << std::endl;

					// Create new cylinder primitive with top and bottom planes and replace old one with it.
					std::cout << "Create new cylinder for " << cyl->name() << std::endl;
					n = lmu::opPrim(
					{ 						
						geo<IFCylinder>(cyl->transform(), cyl->radius(), max_height, cyl->name()),
						geo<IFPlane>(closest_top_plane->p(), top_n, cyl->name() + "_Top"),
						geo<IFPlane>(closest_bottom_plane->p(), bottom_n, cyl->name() + "_Bottom")
					});
				}
			});
		}
	}

	return res_node;
}
