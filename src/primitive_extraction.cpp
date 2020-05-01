#include "primitive_extraction.h"
#include "primitive_helper.h"
#include "csgnode.h"
#include "csgnode_helper.h"
#include "helper.h"

lmu::PrimitiveSetRank const lmu::PrimitiveSetRank::Invalid = lmu::PrimitiveSetRank(-std::numeric_limits<double>::max());

std::tuple<lmu::PrimitiveSet, lmu::ManifoldSet> extractStaticManifolds(const lmu::ManifoldSet& manifolds)
{
	lmu::PrimitiveSet primitives;
	lmu::ManifoldSet restManifolds;

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
		else
		{
			restManifolds.push_back(manifold);
		}

		// All manifolds are non-static.
		//restManifolds.push_back(manifold);

	}

	return std::make_tuple(primitives, restManifolds);
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
	out << r.geo << " " << r.size << std::endl;
	return out;
}

lmu::GAResult lmu::extractPrimitivesWithGA(const RansacResult& ransacRes)
{
	double distT = 0.02;
	double angleT = M_PI / 9.0;
	int maxPrimitiveSetSize = 75;
	
	double sizeWeightGA = 0.1;
	double geoWeightGA = 1.0;
	double perPrimGeoWeightGA = 0.1;
	
	lmu::PrimitiveSetGA::Parameters paramsGA1(150, 2, 0.4, 0.4, false, Schedule(), Schedule(), true);

	// Initialize polytope creator.
	initializePolytopeCreator();

	// static primitives are not changed in the GA process but used.
	auto staticPrimsAndRestManifolds = extractStaticManifolds(ransacRes.manifolds);
	auto manifoldsForCreator = std::get<1>(staticPrimsAndRestManifolds);
	auto staticPrimitives = std::get<0>(staticPrimsAndRestManifolds);

	// get union of all non-static manifold pointclouds.
	std::vector<PointCloud> pointClouds;
	std::transform(manifoldsForCreator.begin(), manifoldsForCreator.end(), std::back_inserter(pointClouds),
		[](const ManifoldPtr m) {return m->pc; });
	auto non_static_pointcloud = lmu::mergePointClouds(pointClouds);

	auto model_sdf = std::make_shared<ModelSDF>(non_static_pointcloud, 0.05, 0.1);

	// First GA for candidate box generation.
	PrimitiveSetTournamentSelector selector(2);
	PrimitiveSetIterationStopCriterion criterion(25, PrimitiveSetRank(0.00001), 10);
	PrimitiveSetCreator creator(manifoldsForCreator, 0.0, { 0.55, 0.15, 0.15, 0.0, 0.15 }, 1, 1, maxPrimitiveSetSize, angleT, 0.001);
	PrimitiveSetRanker ranker(non_static_pointcloud, ransacRes.manifolds, staticPrimitives, 0.005, maxPrimitiveSetSize, 0.01, model_sdf);
	PrimitiveSetPopMan popMan(ranker, maxPrimitiveSetSize, geoWeightGA, perPrimGeoWeightGA, sizeWeightGA, true);
	PrimitiveSetGA ga;
	auto res = ga.run(paramsGA1, selector, creator, ranker, criterion, popMan);
	auto primitives = res.population[0].creature;

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
	
	GAResult result;
	result.primitives = primitives;
	result.primitives.insert(result.primitives.end(), staticPrimitives.begin(), staticPrimitives.end());
	result.manifolds = ransacRes.manifolds;

	return result;
	
}

// ==================== CREATOR ====================

lmu::PrimitiveSetCreator::PrimitiveSetCreator(const ManifoldSet& ms, double intraCrossProb,
	const std::vector<double>& mutationDistribution, int maxMutationIterations, int maxCrossoverIterations,
	int maxPrimitiveSetSize, double angleEpsilon, double minDistanceBetweenParallelPlanes) :
	ms(ms),
	intraCrossProb(intraCrossProb),
	mutationDistribution(mutationDistribution),
	maxMutationIterations(maxMutationIterations),
	maxCrossoverIterations(maxCrossoverIterations),
	maxPrimitiveSetSize(maxPrimitiveSetSize),
	angleEpsilon(angleEpsilon),
	availableManifoldTypes(getAvailableManifoldTypes(ms)),
	minDistanceBetweenParallelPlanes(minDistanceBetweenParallelPlanes)
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
		std::cout << "Mutation New" << std::endl;
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
				std::cout << "Mutation Replace" << std::endl;

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
				std::cout << "Mutation Modify" << std::endl;

				int idx = getRandomPrimitiveIdx(newPS);
				auto newP = mutatePrimitive(newPS[idx], angleEpsilon);
				newPS[idx] = newP.isNone() ? newPS[idx] : newP;

				break;
			}
			case MutationType::REMOVE:
			{
				std::cout << "Mutation Remove" << std::endl;

				//int idx = getRandomPrimitiveIdx(newPS);
				//newPS.erase(newPS.begin() + idx);

				break;
			}
			case MutationType::ADD:
			{
				std::cout << "Mutation Add" << std::endl;

				auto newP = createPrimitive();
				if (!newP.isNone())
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
	std::cout << "Crossover" << std::endl;

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
	const Eigen::Vector3d& point, double minimumPointDistance) const
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	ManifoldSet candidates;
	const double cos_e = std::cos(angleEpsilon);

	// Filter manifold list.
	std::copy_if(ms.begin(), ms.end(), std::back_inserter(candidates),
		[type, &alreadyUsed, &direction, cos_e, ignoreDirection, &point, minimumPointDistance](const ManifoldPtr& m)
	{
		//std::cout << (direction.norm() || ignoreDirection) << " " << m->n.norm() << std::endl;

		return
			m->type == type &&												// same type.
			std::find_if(alreadyUsed.begin(), alreadyUsed.end(),			// not already used.
				[&m, minimumPointDistance](const ManifoldPtr& alreadyUsedM) {
			return lmu::manifoldsEqual(*m, *alreadyUsedM, 0.0001);
		}) == alreadyUsed.end() &&
			(ignoreDirection || std::abs(direction.dot(m->n)) > cos_e) &&	// same direction (or flipped).
			std::abs((point - m->p).dot(m->n)) > minimumPointDistance;		// distance between point and plane  
	});

	if (candidates.empty())
		return nullptr;

	return candidates[du(rndEngine, parmu_t{ 0, (int)candidates.size() - 1 })];
}

lmu::ManifoldPtr lmu::PrimitiveSetCreator::getPerpendicularPlane(const std::vector<ManifoldPtr>& planes,
	const ManifoldSet& alreadyUsed, double angleEpsilon) const
{
	//std::cout << "perp ";

	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	ManifoldSet candidates;
	const double cos_e = std::cos(angleEpsilon);

	// Filter manifold list.
	std::copy_if(ms.begin(), ms.end(), std::back_inserter(candidates),
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

	//std::cout << "found ";


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
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	auto n = du(rndEngine, parmu_t{ 0, (int)availableManifoldTypes.size() - 1 });
	auto it = std::begin(availableManifoldTypes);
	std::advance(it, n);

	switch (*it)
	{
	case ManifoldType::Plane:
		return PrimitiveType::Box;
	case ManifoldType::Cylinder:
		return PrimitiveType::Cylinder;
	default:
		return PrimitiveType::None;
	}
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

		/*std::cout << "####################################" << std::endl;
		if (planes.size() == 6)
		{
		for (int i = 0; i < planes.size(); ++i)
		{
		std::cout
		<< "p: " << planes[i]->p.x() << " " << planes[i]->p.y() << " " << planes[i]->p.z()
		<< " n: " << planes[i]->n.x() << " " << planes[i]->n.y() << " " << planes[i]->n.z() << std::endl;
		}
		}*/

		primitive = createBoxPrimitive(planes);
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

lmu::PrimitiveSetRanker::PrimitiveSetRanker(const PointCloud& pc, const ManifoldSet& ms, const PrimitiveSet& staticPrims,
	double distanceEpsilon, int maxPrimitiveSetSize, double cell_size, const std::shared_ptr<ModelSDF>& model_sdf) :
	pc(pc),
	ms(ms),
	staticPrimitives(staticPrims),
	distanceEpsilon(distanceEpsilon),
	cell_size(cell_size),
	model_sdf(model_sdf),
	maxPrimitiveSetSize(maxPrimitiveSetSize)
{
}

double lmu::PrimitiveSetRanker::get_geo_score(const lmu::PrimitiveSet& ps) const
{
	const double delta = 0.001;
	int validPoints = 0;
	int checkedPoints = 0;

	for (int i = 0; i < pc.rows(); ++i)
	{
		Eigen::Vector3d point = pc.block<1, 3>(i, 0);
		Eigen::Vector3d n = pc.block<1, 3>(i, 3);

		// Get distance to closest primitive. 
		double min_d = std::numeric_limits<double>::max();
		Eigen::Vector3d min_normal;

		for (const auto& p : ps)
		{
			auto dg = p.imFunc->signedDistanceAndGradient(point);
			double d = dg[0];

			Eigen::Vector3d g = dg.bottomRows(3);
			//if (p.cutout)
			//	g = -dg.bottomRows(3);

			if (min_d > d)
			{
				min_d = d;
				min_normal = g.normalized();
			}
		}

		//if (std::abs(min_d) < delta && n.dot(min_normal) > 0.9) validPoints++;
		if (std::abs(min_d) < delta) validPoints++;

		checkedPoints++;
	}

	return (double)validPoints / (double)checkedPoints;
}

std::vector<double> lmu::PrimitiveSetRanker::get_per_prim_geo_score(const PrimitiveSet& ps, double cell_size, double distance_epsilon, const ModelSDF& model_sdf, std::vector<Eigen::Matrix<double, 1, 6>>& points) const
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

	std::vector<double> scores;
	for (const auto& prim : ps)
	{
		if (prim.type != PrimitiveType::Box || prim.imFunc->meshCRef().vertices.rows() != 8)
		{
			scores.push_back(0.0);
			continue;
		}

		auto vertices = prim.imFunc->meshCRef().vertices;

		std::array<Eigen::RowVector3d, 7> v;
		auto p0 = vertices.row(0).transpose();
		for (int i = 1; i < vertices.rows(); ++i)
			v[i-1] = vertices.row(i).transpose() - p0;

		
		// Find axes of coordinate system induced by box vertices.  
		Eigen::Vector3d v0, v1, v2;
		double smallest_sum = std::numeric_limits<double>::max();
		for (const auto& index : indices)
		{
			//std::cout << "01: " << (v[index[0]].dot(v[index[1]])) << " 02: " << (v[index[0]].dot(v[index[2]])) << " 12: " << (v[index[1]].dot(v[index[2]])) << std::endl;

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
		double v0_len, v1_len, v2_len;
		v0_len = std::min(v0.norm(), 0.5);
		v1_len = std::min(v1.norm(), 0.5);
		v2_len = std::min(v2.norm(), 0.5);

		v0.normalize(); v1.normalize(); v2.normalize();
				
		//std::cout << "v0: " << v0.transpose() << " v1: " << v1.transpose() << " v2: " << v2.transpose() << std::endl;
		
		//Compute score 
		std::cout << "Compute score (" << v0_len << ", " << v1_len << ", " << v2_len << ")" << std::endl;
		int inside_voxels = 0;
		int all_voxels = 0;
		double delta_x = cell_size; 
		double delta_y = cell_size;
		double delta_z = cell_size;
		const double e = 0.0000000000000001;
		for (double x = 0.0; x <= v0_len; x += std::max(e, std::min(cell_size, v0_len - x)))
		{
			for (double y = 0.0; y <= v1_len; y += std::max(e, std::min(cell_size, v1_len - y)))
			{
				for (double z = 0.0; z <= v2_len; z += std::max(e, std::min(cell_size, v2_len - z)))
				{
					// Compute voxel pos in world coordinates.
					Eigen::Vector3d p = p0 + v0 * x + v1 * y + v2 * z;

					Eigen::Matrix<double, 1, 6 > pn;
					pn.row(0) << p.transpose(), 1, 0, 0;
					points.push_back(pn);

					if (model_sdf.distance(p) < distance_epsilon)
						inside_voxels++;

					all_voxels++;
				}
			}
		}
		double score = all_voxels > 0 ? (double)inside_voxels / (double)all_voxels : 0.0;
		std::cout << "Done. Score: " << score << std::endl;
	
		scores.push_back(score);
	}

	return scores;
}


lmu::PrimitiveSetRank lmu::PrimitiveSetRanker::rank(const PrimitiveSet& ps) const
{
	if (ps.empty())
		return PrimitiveSetRank::Invalid;	
	

	// Geometry score
	double geo_score = get_geo_score(ps);

	// Size score
	double size_score = (double)ps.size() / (double)maxPrimitiveSetSize;

	// Per prim score
		
	std::vector<Eigen::Matrix<double, 1, 6>> points;

	auto per_prim_geo_score = get_per_prim_geo_score(ps, cell_size, distanceEpsilon, *model_sdf, points);

	return PrimitiveSetRank(geo_score, size_score, 0.0 /*computed later*/, per_prim_geo_score);
}


std::string lmu::PrimitiveSetRanker::info() const
{
	return std::string();
}

// ==================== Population Manipulator ====================

lmu::PrimitiveSetPopMan::PrimitiveSetPopMan(const PrimitiveSetRanker& ranker, int maxPrimitiveSetSize,
	double geoWeight, double perPrimGeoWeight, double sizeWeight,
	bool do_elite_optimization) :
	ranker(&ranker),
	maxPrimitiveSetSize(maxPrimitiveSetSize),
	geoWeight(geoWeight),
	perPrimGeoWeight(perPrimGeoWeight),
	sizeWeight(sizeWeight),
	do_elite_optimization(do_elite_optimization)
{
}

void lmu::PrimitiveSetPopMan::manipulateBeforeRanking(std::vector<RankedCreature<PrimitiveSet, PrimitiveSetRank>>& population) const
{

}

void lmu::PrimitiveSetPopMan::manipulateAfterRanking(std::vector<RankedCreature<PrimitiveSet, PrimitiveSetRank>>& population) const
{
}

std::string lmu::PrimitiveSetPopMan::info() const
{
	return std::string();
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
	if (box->empty())
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

		double height = (i0 - i1).norm();
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
		auto height = lmu::estimateCylinderHeightFromPointCloud(*m);
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

lmu::ModelSDF::ModelSDF(const PointCloud& pc, double voxel_size, double block_radius) : 
	data(nullptr),
	voxel_size(voxel_size)
{
	Eigen::Vector3d border(voxel_size, voxel_size, voxel_size);
	auto _pc = to_canonical_frame(pc);
	Eigen::Vector3d dims = computeAABBDims(_pc) + border;
	origin = Eigen::Vector3d(_pc.leftCols(3).colwise().minCoeff()) - border;
	int int_block_radius = (int)std::ceil(block_radius / voxel_size);

	std::cout << "Border: " << border.transpose() << std::endl;
	std::cout << "Dims: " << dims.transpose() << std::endl;
	std::cout << "Voxel size: " << voxel_size << std::endl;
	std::cout << "Block radius (int): " << int_block_radius << std::endl;

	grid_size = Eigen::Vector3i(std::ceil(dims.x() / voxel_size), std::ceil(dims.y() / voxel_size), std::ceil(dims.z() / voxel_size));
	if (grid_size.x() > 1000 || grid_size.y() > 1000 || grid_size.z() > 1000)
		std::cout << "Too large model size: " << grid_size.transpose() << std::endl;
	std::cout << "Grid size: " << grid_size.transpose() << std::endl;

	n = grid_size.x() * grid_size.y() * grid_size.z();
	data = new SDFValue[n];

	size = Eigen::Vector3d((double)grid_size.x() * voxel_size, (double)grid_size.y() * voxel_size, (double)grid_size.z() * voxel_size);

	// Write SDF values
	float max_v = -std::numeric_limits<float>::max();
	float min_v = std::numeric_limits<float>::max();
	for (int i = 0; i < pc.rows(); ++i)
		fill_block(pc.row(i).leftCols(3), pc.row(i).rightCols(3).normalized(), int_block_radius, min_v, max_v);

	// Normalize weights
	for (int i = 0; i < n; ++i)
		data[i].w = (data[i].w - min_v) / (max_v - min_v);
}

lmu::ModelSDF::~ModelSDF()
{
	delete[] data;
}

lmu::SDFValue lmu::ModelSDF::sdf_value(const Eigen::Vector3d& p) const
{
	Eigen::Vector3i p_int = ((p - origin) / voxel_size).array().round().cast<int>();
	int idx = p_int.x() + grid_size.x() * p_int.y() + grid_size.x() * grid_size.y() * p_int.z();

	return 
		p.x() >= origin.x() && p.x() <= origin.x() + size.x() &&
		p.y() >= origin.y() && p.y() <= origin.y() + size.y() &&
		p.z() >= origin.z() && p.z() <= origin.z() + size.z()
		? data[idx] : SDFValue();
}

double lmu::ModelSDF::distance(const Eigen::Vector3d& p) const
{
	return sdf_value(p).v;
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
				sampling_values(idx) = data[idx].v;
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

				if ( v.w != -1.0 )// && v.v <= voxel_size)
				{
					//std::cout << " " << v.v;
					Eigen::Matrix<double, 1, 6> point;
					point << p.transpose(), v.w, v.w, v.w;
					points.push_back(point);
				}
			}
		}
	}

	return pointCloudFromVector(points);
}

void lmu::ModelSDF::fill_block(const Eigen::Vector3d& p, const Eigen::Vector3d& n, int block_radius, float& min_v, float& max_v)
{	
	Eigen::Vector3i p_int = ((p - origin) / voxel_size).array().round().cast<int>();
	
	Eigen::Vector3i br_min, br_max;

	br_min.x() = std::min(p_int.x(), block_radius);
	br_min.y() = std::min(p_int.y(), block_radius);
	br_min.z() = std::min(p_int.z(), block_radius);

	br_max.x() = p_int.x() + block_radius > grid_size.x() ? grid_size.x() - p_int.x() : block_radius;
	br_max.y() = p_int.y() + block_radius > grid_size.y() ? grid_size.y() - p_int.y() : block_radius;
	br_max.z() = p_int.z() + block_radius > grid_size.z() ? grid_size.z() - p_int.z() : block_radius;
	
	for (int x = -br_min.x(); x < br_max.x(); ++x)
	{
		for (int y = -br_min.y(); y < br_max.y(); ++y)
		{
			for (int z = -br_min.z(); z < br_max.z(); ++z)
			{
				Eigen::Vector3i pi_int = p_int + Eigen::Vector3i(x, y, z);
				Eigen::Vector3d pi = origin + (pi_int.cast<double>() * voxel_size);
				
				double vi = std::copysign((pi - p).norm(), (pi - p).dot(n));
				const double wi = 1.0;

				int idx = pi_int.x() + grid_size.x() * pi_int.y() + grid_size.x() * grid_size.y() * pi_int.z();

				data[idx].v = std::abs(data[idx].v) < std::abs(vi) ? data[idx].v : vi;

				min_v = std::min(data[idx].v, min_v);
				max_v = std::max(data[idx].v, max_v);

				data[idx].w = data[idx].v;
			}
		}
	}
}

inline lmu::SDFValue::SDFValue() :
	SDFValue(max_distance, -1.0)
{
}

inline lmu::SDFValue::SDFValue(float v, float w) :
	v(v),
	w(w)
{
}

const float lmu::SDFValue::max_distance{16.0f};
