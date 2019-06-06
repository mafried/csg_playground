#include "primitive_extraction.h"
#include "primitive_helper.h"
#include "csgnode.h"
#include "csgnode_helper.h"

lmu::GAResult lmu::extractPrimitivesWithGA(const RansacResult& ransacRes)
{
	GAResult result;
		
	PrimitiveSetTournamentSelector selector(2);

	PrimitiveSetIterationStopCriterion criterion(10);
	
	PrimitiveSetCreator creator(ransacRes.manifolds, 0.0, 0.0, 10, 10, 10, M_PI / 18.0);

	PrimitiveSetRanker ranker(ransacRes.pc, 0.2);

	lmu::PrimitiveSetGA::Parameters params(150, 2, 0.3, 0.3, false, Schedule(), Schedule(), false);
	PrimitiveSetGA ga;

	auto res = ga.run(params, selector, creator, ranker, criterion);

	result.primitives = res.population[0].creature;
	result.manifolds = ransacRes.manifolds;

	return result;
}

// ==================== CREATOR ====================

lmu::PrimitiveSetCreator::PrimitiveSetCreator(const ManifoldSet& ms, double intraCrossProb, double intraMutationProb, int maxMutationIterations, 
	int maxCrossoverIterations, int maxPrimitiveSetSize, double angleEpsilon) :
	ms(ms),
	intraCrossProb(intraCrossProb),
	intraMutationProb(intraMutationProb),
	maxMutationIterations(maxMutationIterations),
	maxCrossoverIterations(maxCrossoverIterations),
	maxPrimitiveSetSize(maxPrimitiveSetSize),
	angleEpsilon(angleEpsilon)
{
	rndEngine.seed(rndDevice());
}

lmu::PrimitiveSet lmu::PrimitiveSetCreator::mutate(const PrimitiveSet& ps) const
{
	std::cout << "Mutation" << std::endl;

	static std::bernoulli_distribution db{};
	using parmb_t = decltype(db)::param_type;

	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	auto newPS = ps;

	bool intra = db(rndEngine, parmb_t{ intraMutationProb });

	for (int i = 0; i < du(rndEngine, parmu_t{ 0, (int)maxMutationIterations }); ++i)
	{
		if (newPS.empty())
			break;

		int primitiveIdx = du(rndEngine, parmu_t{ 0, (int)newPS.size() - 1 });

		if (intra)
		{		
			//TODO
			//newPS[primitiveIdx] = mutatePrimitive(newPS[primitiveIdx], angleEpsilon);
		}
		else
		{
			auto p = createPrimitive();
			if (p.type != PrimitiveType::None)
			{
				newPS[primitiveIdx] = p;
			}
		}
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

	bool intra = db(rndEngine, parmb_t{ intraMutationProb });

	for (int i = 0; i < du(rndEngine, parmu_t{ 0, (int)maxCrossoverIterations }); ++i)
	{
		if (intra)
		{
			//TODO
		}
		else
		{
			if (!ps1.empty() && !ps2.empty())
			{
				int idx1 = du(rndEngine, parmu_t{ 0, (int)ps1.size() - 1 });
				int idx2 = du(rndEngine, parmu_t{ 0, (int)ps2.size() - 1 });

				newPS1[idx1] = ps2[idx2];
				newPS2[idx2] = ps1[idx1];
			}
		}
	}

	return {newPS1, newPS2};
}

lmu::PrimitiveSet lmu::PrimitiveSetCreator::create() const
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	int setSize = du(rndEngine, parmu_t{ 0, (int)maxPrimitiveSetSize });

	PrimitiveSet ps; 
	//ps.reserve(setSize);
	
	while (ps.size() < setSize)
	{
		//std::cout << "try to create primitive" << std::endl;
		auto p = createPrimitive();
		if (p.type != PrimitiveType::None)
		{
			ps.push_back(std::move(p));			
		}
		else
		{
			std::cout << "None" << std::endl;
		}
	}

	std::cout << "PS SIZE: " << ps.size() << std::endl;

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
		return 
			m->type == type &&																// same type.
			std::find(alreadyUsed.begin(), alreadyUsed.end(), m) == alreadyUsed.end() &&	// not already used.
			(ignoreDirection || std::abs(direction.dot(m->n)) > cos_e);						// same direction (or flipped).
	});
	
	if (candidates.empty())
		return nullptr;

	return candidates[du(rndEngine, parmu_t{ 0, (int)candidates.size() - 1 })];
}

lmu::Primitive lmu::PrimitiveSetCreator::createPrimitive() const
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	ManifoldSet primManifoldSet; 
	auto primitiveType = (PrimitiveType)du(rndEngine, parmu_t{ 1, numPrimitiveTypes - 1 });
		
	switch (primitiveType)
	{
	case PrimitiveType::Box:
		
		/*auto startingPlane = getManifold(ManifoldType::Plane, Eigen::Vector3d(0, 0, 0), {}, 0.0, true);
		ManifoldSet planes;
		planes.push_back(startingPlane);
		for (int i = 0; i < 5; ++i)
		{
			auto p = getManifold(ManifoldType::Plane, cyl->n, planes, angleEpsilon);
			if (p)
				planes.push_back(p);
		}
		return createBoxPrimitive()
		//TODO: use getManifold() to implement a method that fills a 6 plane manifold set (primManifoldSet) for the box.
		//TODO*/
		break; 
	case PrimitiveType::Cone: 	
		//TODO
		//break;
	case PrimitiveType::Cylinder:
		{
			auto cyl = getManifold(ManifoldType::Cylinder, Eigen::Vector3d(0, 0, 0), {}, 0.0, true);
			if (cyl)
			{
				ManifoldSet planes; 
				for (int i = 0; i < 2; ++i)
				{
					auto p = getManifold(ManifoldType::Plane, cyl->n, planes, angleEpsilon);
					if (p)
						planes.push_back(p);
				}
				return createCylinderPrimitive(cyl, planes);
			}
		}
		break;
	case PrimitiveType::Sphere:
		return createSpherePrimitive(getManifold(ManifoldType::Sphere, Eigen::Vector3d(0, 0, 0), {}, 0.0, true));		
	}

	return lmu::Primitive();
}

lmu::Primitive lmu::PrimitiveSetCreator::mutatePrimitive(const Primitive& p, double angleEpsilon) const
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	if (p.ms.empty())
		return p;
		
	int manifoldIdx = du(rndEngine, parmu_t{ 0, (int)p.ms.size() - 1 });
	
	auto m = p.ms[manifoldIdx];

	Primitive newP = p;

	auto newManifold = getManifold(m->type, m->n, p.ms, angleEpsilon);
	newManifold = newManifold ? newManifold : m;

	newP.ms[manifoldIdx] = newManifold;

	return newP;
}

// ==================== RANKER ====================

lmu::PrimitiveSetRanker::PrimitiveSetRanker(const PointCloud& pc, double distanceEpsilon) :
	pc(pc),
	distanceEpsilon(distanceEpsilon)
{
}

lmu::PrimitiveSetRank lmu::PrimitiveSetRanker::rank(const PrimitiveSet& ps) const
{
	//CSGNode node = opUnion();
	//for (const auto& p : ps)	
	//	node.addChild(geometry(p.imFunc));
	
	double meanGeometryScore = 0.0;
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

			double d = prim.imFunc->signedDistance(p);

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

	double s = 0.1;

	return totalGeometryScore + meanGeometryScore - s * ps.size();
}

std::string lmu::PrimitiveSetRanker::info() const
{
	return std::string();
}

lmu::Primitive lmu::createBoxPrimitive(const ManifoldSet& planes)
{
	if (planes.size() != 6)
		return Primitive();

	for (const auto& p : planes)
		if (p->type != ManifoldType::Plane)
			return Primitive();

	//TODO: create IFBox out of 6 planes.
	Primitive prim;

	return prim;
}

lmu::Primitive lmu::createSpherePrimitive(const lmu::ManifoldPtr& m)
{
	if (!m)
		return Primitive();

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

		double height = (i0-i1).norm();
		Eigen::Vector3d pos = i0 + (0.5 * (i1 - i0));
		
		// Compute cylinder transform.
		Eigen::Matrix3d rot = getRotationMatrix(m->n);
		Eigen::Affine3d t = (Eigen::Affine3d)(Eigen::Translation3d(pos) * rot);

		// Create primitive. 
		auto cylinderIF = std::make_shared<IFCylinder>(t, m->r.x(), height, "");
		cylinderIF->meshRef() = Mesh();

		return Primitive (cylinderIF, { m, planes[0], planes[1] }, PrimitiveType::Cylinder);
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

		return Primitive(cylinderIF, { m}, PrimitiveType::Cylinder);
	}
	default:
		return Primitive();
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

			auto sphereIF = std::make_shared<IFSphere>(t,m->r.x(), "");
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
		
	auto secondPlane = std::make_shared<Manifold>(ManifoldType::Plane, p , -firstPlane.n, Eigen::Vector3d(0, 0, 0), PointCloud());

	return secondPlane;
}

