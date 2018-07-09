#ifndef CSGNODE_GEN_H
#define CSGNODE_GEN_H

#include <vector>
#include <memory>
#include <random>
#include <boost/graph/connected_components.hpp>

#include "csgnode.h"
#include "evolution.h"
#include "mesh.h"
#include "csgnode_helper.h"

#include <Eigen/Core>

namespace lmu
{
	struct ImplicitFunction;

	template<typename T = unsigned long>
	struct Geometry
	{
		Geometry(ImplicitFunctionType type, const std::string& name) :
			type(type),
			name(name)
		{
		}

		Geometry()
		{
		}

		T mask;

		Eigen::Affine3d transform;
		Eigen::Vector3d size;
		Eigen::Vector3d pos;
		ImplicitFunctionType type;
		std::string name;
	};

	struct GeometrySet
	{
		std::vector<Geometry<>> geometries;

		const std::vector<std::shared_ptr<lmu::ImplicitFunction>> createFuncs() const
		{
			std::vector<std::shared_ptr<lmu::ImplicitFunction>> funcs;

			for (const auto& geo : geometries)
			{
				switch (geo.type)
				{
				case ImplicitFunctionType::Sphere:
					funcs.push_back(std::make_shared<IFSphere>(geo.transform, geo.size.x(), geo.name));
					break;
				case ImplicitFunctionType::Box:
					funcs.push_back(std::make_shared<IFBox>(geo.transform, geo.size, 2, geo.name));
					break;
				case ImplicitFunctionType::Cylinder:
					funcs.push_back(std::make_shared<IFCylinder>(geo.transform, geo.size.x(), geo.size.y(), geo.name));
					break;

				}
			}

			return funcs;
		}
	};

	

	struct GeometrySetCreator
	{
		GeometrySetCreator(const GeometrySet& geometrySet, const Eigen::Vector3i& gridSize = Eigen::Vector3i(50,50,50), double gridStep = 1.0, double maxObjSize = 10.0, double minObjSize = 3.0) :
			_gridSize(gridSize),
			_gridStep(gridStep),			
			_geos(geometrySet),
			_maxObjSize(maxObjSize),
			_minObjSize(minObjSize)
		{
		}

		GeometrySet mutate(const GeometrySet& set) const
		{
			static std::bernoulli_distribution db{};
			using parmb_t = decltype(db)::param_type;

			GeometrySet newSet;

			for (const auto& geo : set.geometries)
			{
				Geometry<> newGeo(geo);
				
				if (db(_rndEngine, parmb_t{ 0.5 }))
				{
					newGeo.pos = randomTransform();
					newGeo.transform = (Eigen::Affine3d)(Eigen::Translation3d(newGeo.pos.x(), newGeo.pos.y(), newGeo.pos.z()));
					newGeo.size = randomSize();
				}

				newSet.geometries.push_back(newGeo);
			}

			return newSet;
		}

		std::vector<GeometrySet> crossover(const GeometrySet& set1, const GeometrySet& set2) const
		{
			static std::bernoulli_distribution db{};
			using parmb_t = decltype(db)::param_type;

			GeometrySet cSet1 = set1; 
			GeometrySet cSet2 = set2;

			for(int i = 0; i < cSet1.geometries.size(); i++)
			{
				if (db(_rndEngine, parmb_t{ 0.5 }))
				{
					std::swap(cSet1.geometries[i], cSet2.geometries[i]);
				}
			}

			return { cSet1, cSet2 };		
		}

		GeometrySet create() const
		{
			GeometrySet set;

			for(const auto& geo : _geos.geometries)
			{
				Geometry<> newGeo; 
				newGeo.name = geo.name; 
				newGeo.type = geo.type; 
				newGeo.pos = randomTransform();
				newGeo.transform = (Eigen::Affine3d)(Eigen::Translation3d(newGeo.pos.x(), newGeo.pos.y(), newGeo.pos.z()));
				newGeo.size = randomSize();

				set.geometries.push_back(newGeo);
			}

			return set;
		}
		
		std::string info() const
		{
			return std::string();
		}

	private:

Eigen::Vector3d randomSize() const
{
	static std::uniform_real_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	double x = du(_rndEngine, parmu_t{ _minObjSize, _maxObjSize });
	double y = du(_rndEngine, parmu_t{ _minObjSize, _maxObjSize });
	double z = du(_rndEngine, parmu_t{ _minObjSize, _maxObjSize });

	return Eigen::Vector3d(x, y, z);
}

Eigen::Vector3d randomTransform() const
{
	static std::uniform_int_distribution<> du{};
	using parmu_t = decltype(du)::param_type;

	double x = du(_rndEngine, parmu_t{ 0, _gridSize.x() });
	double y = du(_rndEngine, parmu_t{ 0, _gridSize.y() });
	double z = du(_rndEngine, parmu_t{ 0, _gridSize.z() });

	return Eigen::Vector3d(x, y, z);
}



GeometrySet _geos;
Eigen::Vector3i _gridSize;
double _gridStep;
double _maxObjSize;
double _minObjSize;
mutable std::default_random_engine _rndEngine;
mutable std::random_device _rndDevice;

//lmu::Graph _graph;
	};

	struct GeometrySetRanker
	{

		GeometrySetRanker()
		{

		}

		double rank(const GeometrySet& gs) const
		{
			auto graph = createConnectionGraph(gs.createFuncs());

			std::vector<int> component(boost::num_vertices(graph));
			int numCoCos = boost::connected_components(graph, &component[0]);

			//std::cout << "Rank: " << (1.0 / (double)numCoCos) << std::endl;

			return 1.0 / (double)numCoCos;
		}

		std::string info() const
		{
			return std::string();
		}

	private:


		//lmu::Graph _connectionGraph;
	};


	using GeometrySetTournamentSelector = TournamentSelector<RankedCreature<GeometrySet>>;

	using GeometrySetIterationStopCriterion = IterationStopCriterion<RankedCreature<GeometrySet>>;
	using GeometrySetNoFitnessIncreaseStopCriterion = NoFitnessIncreaseStopCriterion<RankedCreature<GeometrySet>>;
	using GeometrySetFitnessReachedStopCriterion = FitnessReachedStopCriterion<RankedCreature<GeometrySet>>;

	using GeometrySetGA = GeneticAlgorithm<GeometrySet, GeometrySetCreator, GeometrySetRanker, GeometrySetTournamentSelector, GeometrySetFitnessReachedStopCriterion>;

	GeometrySet generateConnectedGeometrySetWithGA(const GeometrySet& geometrySet, const Eigen::Vector3i& gridSize = Eigen::Vector3i(50, 50, 50), double gridStep = 1.0, 
		double maxObjSize = 10.0, double minObjSize = 3.0, int maxIter = 50, int populationSize = 150, int tournamentNum = 2)
	{
		GeometrySetGA ga;
		GeometrySetGA::Parameters p(populationSize, 2, 0.3, 0.3, true);

		GeometrySetTournamentSelector s(tournamentNum, true);

		//GeometrySetIterationStopCriterion isc(100);
		//GeometrySetNoFitnessIncreaseStopCriterion isc(10, 0.01, 50);
		GeometrySetFitnessReachedStopCriterion isc(1.0, 0.1, maxIter);
		GeometrySetCreator c(geometrySet, gridSize, gridStep, maxObjSize, minObjSize);


		lmu::GeometrySetRanker r;

		auto res = ga.run(p, s, c, r, isc);

		return res.population[0].creature;
	}

	void replaceLastUnion(CSGNode& node)
	{
		for (auto& child : node.childsRef())
		{
			if (child.operationType() == CSGNodeOperationType::Union)
			{
				if (child.childsRef()[1].function() != nullptr && child.childsRef()[1].function()->name() == "Null")
				{
					std::swap(child, child.childsRef()[0]);
				}
				else if (child.childsRef()[0].function() != nullptr && child.childsRef()[0].function()->name() == "Null")
				{
					std::swap(child, child.childsRef()[1]);
				}
			}
		}
	}

	CSGNode createCSGNodeFromGeometrySet(const GeometrySet& set, int setIdx = 0)
	{
		if (setIdx >= set.geometries.size())
			return geometry(std::make_shared<IFNull>("Null"));
		

		CSGNode node = createOperation(CSGNodeOperationType::Union, "Union", { geometry(set.createFuncs()[setIdx]), createCSGNodeFromGeometrySet(set, setIdx + 1) });	

		replaceLastUnion(node);

		return node;
	}
}

#endif