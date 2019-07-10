#include "primitive_extraction.h"
#include "primitive_helper.h"
#include "csgnode.h"
#include "csgnode_helper.h"

lmu::GAResult lmu::extractPrimitivesWithGA(const RansacResult& ransacRes)
{
	GAResult result;

	PrimitiveSetTournamentSelector selector(2);

	PrimitiveSetIterationStopCriterion criterion(100);

	int maxPrimitiveSetSize = 5;

	PrimitiveSetCreator creator(ransacRes.manifolds, 0.0, 0.5, 0.3, 1, 1, maxPrimitiveSetSize, /*M_PI / 18.0*/ M_PI / 9.0);

	PrimitiveSetRanker ranker(ransacRes.pc, ransacRes.manifolds, 0.2, maxPrimitiveSetSize);

	lmu::PrimitiveSetGA::Parameters params(150, 2, 0.7, 0.7, true, Schedule(), Schedule(), false);
	PrimitiveSetGA ga;

	auto res = ga.run(params, selector, creator, ranker, criterion);

	result.primitives = ranker.bestPrimitiveSet();//res.population[0].creature;
	result.manifolds = ransacRes.manifolds;

	//std::cout << "BEST RANK: " << ranker.rank(ranker.bestPrimitiveSet());

	return result;
}

// ==================== CREATOR ====================

lmu::PrimitiveSetCreator::PrimitiveSetCreator(const ManifoldSet& ms, double intraCrossProb, double intraMutationProb, double createNewMutationProb, int maxMutationIterations,
	int maxCrossoverIterations, int maxPrimitiveSetSize, double angleEpsilon) :
	ms(ms),
	intraCrossProb(intraCrossProb),
	intraMutationProb(intraMutationProb),
	createNewMutationProb(createNewMutationProb),
	maxMutationIterations(maxMutationIterations),
	maxCrossoverIterations(maxCrossoverIterations),
	maxPrimitiveSetSize(maxPrimitiveSetSize),
	angleEpsilon(angleEpsilon)
{
	rndEngine.seed(rndDevice());
}

int lmu::PrimitiveSetCreator::getRandomPrimitiveIdxNoSphere(const PrimitiveSet& ps) const
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	int sphereCounter = 0;
	for (const auto& p : ps)
		sphereCounter += p.type == lmu::PrimitiveType::Sphere ? 1 : 0;

	// All spheres!
	if (sphereCounter == ps.size()) return -1;
	
	while (true)
	{
		int primitiveIdx = du(rndEngine, parmu_t{ 0, (int)ps.size() - 1 });
		if (ps[primitiveIdx].type != PrimitiveType::Sphere) return primitiveIdx;
	}
}

int lmu::PrimitiveSetCreator::getRandomPrimitiveIdx(const PrimitiveSet& ps) const
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	return du(rndEngine, parmu_t{ 0, (int)ps.size() - 1 });
}

lmu::PrimitiveSet lmu::PrimitiveSetCreator::mutate(const PrimitiveSet& ps) const
{
	static std::bernoulli_distribution db{};
	using parmb_t = decltype(db)::param_type;

	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	if (db(rndEngine, parmb_t{ createNewMutationProb }) || ps.empty())
	{
		std::cout << "Mutation Create New" << std::endl;
		return create();
	}

	auto newPS = ps;
	
	for (int i = 0; i < du(rndEngine, parmu_t{ 1, (int)maxMutationIterations }); ++i)
	{
		bool intra = db(rndEngine, parmb_t{ intraMutationProb });

		if (intra)
		{			
			std::cout << "Mutation Intra" << std::endl;

			int idx = getRandomPrimitiveIdx(newPS);
			auto newP = mutatePrimitive(newPS[idx], angleEpsilon);
			newPS[idx] = newP.isNone() ? newPS[idx] : newP;
		}
		else
		{
			std::cout << "Mutation Extra" << std::endl;

			int idx = getRandomPrimitiveIdxNoSphere(newPS); // Spheres are never replaced.
			if (idx != -1)
			{
				auto newP = createPrimitive();
				newPS[idx] = newP.isNone() ? newPS[idx] : newP;
			}
		}

		//TODO: Add mutation operators that change the size of the primitive set.
	}

	return newPS;
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
		bool intra = db(rndEngine, parmb_t{ intraMutationProb });

		if (intra)
		{
			//TODO (if it makes sense).
		}
		else
		{
			if (!ps1.empty() && !ps2.empty())
			{
				int idx1 = getRandomPrimitiveIdxNoSphere(ps1); // Spheres are never replaced.
				int idx2 = getRandomPrimitiveIdxNoSphere(ps2); // Spheres are never replaced.

				if (idx1 != -1 && idx2 != -1)
				{
					newPS1[idx1] = ps2[idx2];
					newPS2[idx2] = ps1[idx1];
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
	
	// Fill primitive set with all spheres. 
	for (const auto& m : ms)
	{
		if (m->type == ManifoldType::Sphere)
		{
			auto sphere = createSpherePrimitive(m);
			sphere.cutout = db(rndEngine, parmb_t{ 0.5 });
			ps.push_back(sphere);
		}
	}

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

lmu::ManifoldPtr lmu::PrimitiveSetCreator::getManifold(ManifoldType type, const Eigen::Vector3d& direction, const ManifoldSet& alreadyUsed, double angleEpsilon, bool ignoreDirection) const
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	ManifoldSet candidates;
	const double cos_e = std::cos(angleEpsilon);

	// Filter manifold list.
	std::copy_if(ms.begin(), ms.end(), std::back_inserter(candidates),
		[type, &alreadyUsed, &direction, cos_e, ignoreDirection](const ManifoldPtr& m)
	{
		//std::cout << (direction.norm() || ignoreDirection) << " " << m->n.norm() << std::endl;

		return
			m->type == type &&																// same type.
			std::find(alreadyUsed.begin(), alreadyUsed.end(), m) == alreadyUsed.end() &&	// not already used.
			(ignoreDirection || std::abs(direction.dot(m->n)) > cos_e);						// same direction (or flipped).
	});

	if (candidates.empty())
		return nullptr;

	return candidates[du(rndEngine, parmu_t{ 0, (int)candidates.size() - 1 })];
}

lmu::ManifoldPtr lmu::PrimitiveSetCreator::getPerpendicularPlane(const std::vector<ManifoldPtr>& planes, const ManifoldSet& alreadyUsed, double angleEpsilon) const
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
		if (m->type != ManifoldType::Plane || std::find(alreadyUsed.begin(), alreadyUsed.end(), m) != alreadyUsed.end()) // only planes that weren't used before.
			return false;

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

lmu::ManifoldPtr lmu::PrimitiveSetCreator::getParallelPlane(const ManifoldPtr& plane, const ManifoldSet & alreadyUsed, double angleEpsilon) const
{
	auto foundPlane = getManifold(ManifoldType::Plane, plane->n, alreadyUsed, angleEpsilon);

	return foundPlane;
}

lmu::Primitive lmu::PrimitiveSetCreator::createPrimitive() const
{
	const auto anyDirection = Eigen::Vector3d(0, 0, 0);

	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	static std::bernoulli_distribution db{};
	using parmb_t = decltype(db)::param_type;

	ManifoldSet primManifoldSet;
	auto primitiveType = PrimitiveType::Box;  // (PrimitiveType)du(rndEngine, parmu_t{ 1, numPrimitiveTypes - 1 }); TODO

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

		plane = getParallelPlane(plane, planes, angleEpsilon);
		if (!plane)
			break;
		planes.push_back(plane);
		
		plane = getPerpendicularPlane(planes, planes, angleEpsilon);
		if (!plane)
			break;
		planes.push_back(plane);
		
		plane = getParallelPlane(plane, planes, angleEpsilon);
		if (!plane)
			break;
		planes.push_back(plane);
		
		plane = getPerpendicularPlane(planes, planes, angleEpsilon);
		if (!plane)
			break;
		planes.push_back(plane);
		
		plane = getParallelPlane(plane, planes, angleEpsilon);
		if (!plane)
			break;
		planes.push_back(plane);

		primitive = createBoxPrimitive(planes);
	}
	break;

	//case PrimitiveType::Cone:
		//TODO
		//break;
	case PrimitiveType::Cylinder:
	{
		auto cyl = getManifold(ManifoldType::Cylinder, anyDirection, {}, 0.0, true);
		if (cyl)
		{
			ManifoldSet planes;
			for (int i = 0; i < 2; ++i)
			{
				auto p = getManifold(ManifoldType::Plane, cyl->n, planes, angleEpsilon);
				if (p)
					planes.push_back(p);
			}
			primitive = createCylinderPrimitive(cyl, planes);
		}
	}
	break;
	//case PrimitiveType::Sphere:
	//{
	//	primitive = createSpherePrimitive(getManifold(ManifoldType::Sphere, anyDirection, {}, 0.0, true));
	//	break;
	//}
	}

	// Primitive is cut out of the model or added.
	primitive.cutout = db(rndEngine, parmb_t{ 0.5 });

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
			auto newPlane = getParallelPlane(p.ms[planePairIdx], p.ms, angleEpsilon);
			if (newPlane)
			{
				auto newPlanes = ManifoldSet(p.ms);
				newPlanes[planePairIdx+1] = newPlane;

				primitive = createBoxPrimitive(newPlanes);
			}

			break;
		}

		case PrimitiveType::Sphere: 
			//Do nothing. 
			break;
	}

	// Primitive is cut out of the model or added.
	primitive.cutout = db(rndEngine, parmb_t{ 0.5 });

	return primitive;
}

// ==================== RANKER ====================

lmu::PrimitiveSetRanker::PrimitiveSetRanker(const PointCloud& pc, const ManifoldSet& ms, double distanceEpsilon,int maxPrimitiveSetSize) :
	pc(pc),
	ms(ms),
	distanceEpsilon(distanceEpsilon),
	bestRank(-std::numeric_limits<double>::max()),
	maxPrimitiveSetSize(maxPrimitiveSetSize)
{
}

lmu::PrimitiveSetRank lmu::PrimitiveSetRanker::rank(const PrimitiveSet& ps) const
{
	if (ps.empty())
		return -std::numeric_limits<double>::max();

	CSGNode node = opUnion();
	for (const auto& p : ps)
	{
		node.addChild(p.cutout ? op<ComplementOperation>({ geometry(p.imFunc) }) : geometry(p.imFunc));
	}

	const double delta = 0.01;

	int validPoints = 0; 
	int checkedPoints = 0;

	for (const auto& manifold : ms)
	{
		for (int i = 0; i < manifold->pc.rows(); ++i)
		{
			Eigen::Vector3d p = manifold->pc.block<1, 3>(i, 0);
			Eigen::Vector3d n = manifold->pc.block<1, 3>(i, 3);

			auto dg = node.signedDistanceAndGradient(p);
			double d = dg[0];
			Eigen::Vector3d g = dg.bottomRows(3);

			//TODO: do something with the normal.

			validPoints += std::abs(d) < delta && n.dot(g) >= 0.0 ? 1 : 0;
			checkedPoints++;
		}
	}

	double s = 0.2;

	//std::cout << "Rank Ready." << std::endl;

	double r = (double)validPoints / (double)checkedPoints - s * (double)ps.size() / (double)maxPrimitiveSetSize;

	if(bestRank < r) 
	{
		bestRank = r;
		bestPrimitives = ps;
		//std::cout << "NEW BEST: " << r << std::endl;
	}

	return r; 
	/*double meanGeometryScore = 0.0;
	std::vector<int> totalValidPoints(pc.rows(), 0);

	for (const auto prim : ps)
	{
		double geometryScore = 0.0;
		double validPoints = 0.0;
		double wrongPoints = 0.0;

		for (int i = 0; i < pc.rows(); ++i)
		{
			Eigen::Vector3d p = pc.block<1, 3>(i, 0);
			Eigen::Vector3d n = pc.block<1, 3>(i, 3);

			//TODO: do something with the normal.

			double d = 0.0;

			if (prim.imFunc)
				d = prim.imFunc->signedDistance(p);

			if (d <= distanceEpsilon)
			{
				validPoints += 1.0;

				if (d < -distanceEpsilon)
				{
					wrongPoints += 1.0;
				}
				else
				{
					totalValidPoints[i] = 1;
				}
			}
		}

		geometryScore = (validPoints - wrongPoints) / validPoints;
		meanGeometryScore += geometryScore;
	}
	meanGeometryScore /= (double)ps.size();

	double totalGeometryScore = (double)std::accumulate(totalValidPoints.begin(), totalValidPoints.end(), 0) / (double)pc.rows();

	double s = 0.01;

	return totalGeometryScore + meanGeometryScore - s * ps.size();*/
}

std::string lmu::PrimitiveSetRanker::info() const
{
	return std::string();
}

lmu::PrimitiveSet lmu::PrimitiveSetRanker::bestPrimitiveSet() const
{
	return bestPrimitives;
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

		if(strictlyParallel)
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
	if (!m)
		return Primitive::None();

	Eigen::Affine3d t = Eigen::Affine3d::Identity();
	t.translate(m->p);

	auto sphereIF = std::make_shared<IFSphere>(t, m->r.x(), ""); //TODO: Add name.
	sphereIF->meshRef() = Mesh();


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
		cylinderIF->meshRef() = Mesh();

		return Primitive(cylinderIF, { m, planes[0], planes[1] }, PrimitiveType::Cylinder);
	}
	case 0:	//Estimate cylinder height and center position using the point cloud only since no planes exist.
	{
		auto heightPos = lmu::estimateCylinderHeightAndPosFromPointCloud(*m);
		auto height = std::get<0>(heightPos);
		auto pos = std::get<1>(heightPos);

		Eigen::Matrix3d rot = getRotationMatrix(m->n);
		Eigen::Affine3d t = (Eigen::Affine3d)(Eigen::Translation3d(pos) * rot);

		auto cylinderIF = std::make_shared<IFCylinder>(t, m->r.x(), height, "");
		cylinderIF->meshRef() = Mesh();

		return Primitive(cylinderIF, { m }, PrimitiveType::Cylinder);
	}
	default:
		return Primitive::None();
	}
}

lmu::PrimitiveSet lmu::extractPrimitivesFromBorderlessManifolds(const ManifoldSet& manifolds)
{
	PrimitiveSet primitives;

	for (const auto& m : manifolds)
	{
		if (m->type == ManifoldType::Sphere)
		{
			Eigen::Affine3d t = Eigen::Affine3d::Identity();
			t.translate(m->p);

			auto sphereIF = std::make_shared<IFSphere>(t, m->r.x(), "");
			sphereIF->meshRef() = Mesh();

			std::cout << "SPHERE: " << sphereIF->transform().matrix() << std::endl;

			Primitive p(sphereIF, { m }, PrimitiveType::Sphere);

			primitives.push_back(p);
		}
		//TODO
		//else if (m->type == ManifoldType::Torus)
		//{
		//
		//}
	}

	return primitives;
}

lmu::PrimitiveSet lmu::extractCylindersFromCurvedManifolds(const ManifoldSet& manifolds, bool estimateHeight)
{
	PrimitiveSet primitives;

	for (const auto& m : manifolds)
	{
		if (m->type == ManifoldType::Cylinder)
		{
			auto heightAndPos = estimateCylinderHeightAndPosFromPointCloud(*m);
			double height = std::get<0>(heightAndPos);
			Eigen::Vector3d estimatedPos = std::get<1>(heightAndPos);

			Eigen::Vector3d up(0, 0, 1);
			Eigen::Vector3d f = m->n;
			Eigen::Vector3d r = (f).cross(up).normalized();
			Eigen::Vector3d u = (r).cross(f).normalized();

			Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
			rot <<
				r.x(), f.x(), u.x(),
				r.y(), f.y(), u.y(),
				r.z(), f.z(), u.z();

			Eigen::Affine3d t = (Eigen::Affine3d)(Eigen::Translation3d(/*m->p*/estimatedPos) * rot);

			auto cylinderIF = std::make_shared<IFCylinder>(t, m->r.x(), height, "");
			cylinderIF->meshRef() = Mesh();

			std::cout << "Cylinder: " << std::endl;
			std::cout << "Estimated Height: " << height << std::endl;
			std::cout << "----------------------" << std::endl;

			Primitive p(cylinderIF, { m }, PrimitiveType::Cylinder);

			if (!std::isnan(height) && !std::isinf(height))
			{
				primitives.push_back(p);
			}
			else
			{
				std::cout << "Filtered cylinder with nan or inf height. " << std::endl;
			}
		}
	}
	return primitives;
}

std::tuple<double, Eigen::Vector3d> lmu::estimateCylinderHeightAndPosFromPointCloud(const Manifold& m)
{
	// Get matrix for transform to identity rotation.

	Eigen::Vector3d up(0, 0, 1);
	Eigen::Vector3d f = m.n;
	Eigen::Vector3d r = (f).cross(up).normalized();
	Eigen::Vector3d u = (r).cross(f).normalized();

	Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
	rot <<
		r.x(), f.x(), u.x(),
		r.y(), f.y(), u.y(),
		r.z(), f.z(), u.z();

	Eigen::Affine3d t = (Eigen::Affine3d)(rot);
	auto tinv = t.inverse();

	// Transform cylinder direction to identity rotation and find index of principal axis.

	Eigen::Vector3d f2 = (tinv * Eigen::Vector3d(0, 0, 0)) - (tinv * m.n);
	double fa[3] = { std::abs(f2.x()), std::abs(f2.y()), std::abs(f2.z()) };
	int coordinateIdx = std::distance(fa, std::max_element(fa, fa + 3));

	double minC = std::numeric_limits<double>::max();
	double maxC = -std::numeric_limits<double>::max();

	// Get largest extend along principal axis (= cylinder height)

	for (int i = 0; i < m.pc.rows(); ++i)
	{
		Eigen::Vector3d p = m.pc.row(i).leftCols(3);
		p = tinv * p;

		double c = p.coeff(coordinateIdx);

		if (c < minC)
		{
			minC = c;

		}
		if (c > maxC)
		{
			maxC = c;

		}
	}

	double height = std::abs(maxC - minC);

	// Get min / max extend of point cloud to calculate the center of the point cloud's AABB (= cylinder pos).

	Eigen::Vector3d minPos = (Eigen::Vector3d)(m.pc.leftCols(3).colwise().minCoeff());
	Eigen::Vector3d maxPos = (Eigen::Vector3d)(m.pc.leftCols(3).colwise().maxCoeff());

	//std::cout << minPos << std::endl;
	//std::cout << maxPos << std::endl;

	Eigen::Vector3d pos = (minPos + ((maxPos - minPos) * 0.5));

	return std::make_tuple(height, pos);
}


lmu::ManifoldPtr lmu::estimateSecondCylinderPlaneFromPointCloud(const Manifold& m, const Manifold& firstPlane)
{
	Eigen::Vector3d minPos = (Eigen::Vector3d)(m.pc.leftCols(3).colwise().minCoeff());
	Eigen::Vector3d maxPos = (Eigen::Vector3d)(m.pc.leftCols(3).colwise().maxCoeff());

	//Take the point of the point cloud's min-max points which is farer away from the first plane as the second plane's point.
	Eigen::Vector3d p = (firstPlane.p - minPos).norm() > (firstPlane.p - maxPos).norm() ? minPos : maxPos;

	auto secondPlane = std::make_shared<Manifold>(ManifoldType::Plane, p, -firstPlane.n, Eigen::Vector3d(0, 0, 0), PointCloud());

	return secondPlane;
}

