#ifndef EVOLUTION_H
#define EVOLUTION_H

#include <vector>
#include <algorithm>
#include <tuple>
#include <random>
#include <limits>
#include <memory>
#include <atomic>
#include <future>

#include "csgtree.h"

namespace lmu
{
	template<typename Creature>
	struct RankedCreature
	{
		RankedCreature(const Creature& c, double rank = unranked()) :
			creature(c),
			rank(rank)
		{
		}		

		static double unranked() 
		{
			return std::numeric_limits<double>::min();
		}

		Creature creature;
		double rank;
	};

	template<typename RankedCreature>
	struct TournamentSelector
	{
		TournamentSelector(int k) :
			_k(k)
			{
				_rndEngine.seed(_rndDevice());
			}

			RankedCreature selectFrom(const std::vector<RankedCreature>& population) const
			{
				static std::uniform_int_distribution<> d{};
				using parm_t = decltype(d)::param_type;

				bool firstRun = true;
				int best = 0;
				for (int i = 0; i < _k; ++i)
				{
					int idx = d(_rndEngine, parm_t{ 0, (int)population.size() - 1 });
					if (firstRun || population[idx].rank < population[best].rank)
					{
						firstRun = false;
						best = idx;
					}
				}

				return population[best];
			}
	private:
		int _k;
		mutable std::default_random_engine _rndEngine;
		mutable std::random_device _rndDevice;
	};

	template<typename RankedCreature>
	struct IterationStopCriterion
	{
		IterationStopCriterion(int maxIterations) :
			_maxIterations(maxIterations)
		{
		}

		bool shouldStop(const std::vector<RankedCreature>& population, int iterationCount) const
		{
			std::cout << "Iteration " << iterationCount << " of " << _maxIterations << std::endl;
			return iterationCount >= _maxIterations;
		}

	private:
		int _maxIterations;
	};

	template<
		typename Creature, typename CreatureCreator, typename CreatureRanker, 
		typename ParentSelector = TournamentSelector<RankedCreature<Creature>>,
		typename StopCriterion = IterationStopCriterion<RankedCreature<Creature>>
	>
	class GeneticAlgorithm
	{
	public: 

		using RankedCreature = RankedCreature<Creature>;

		struct Parameters
		{
			Parameters(int populationSize, int numBestParents, double mutationRate, double crossoverRate) : 
				populationSize(populationSize),
				numBestParents(numBestParents),
				mutationRate(mutationRate),
				crossoverRate(crossoverRate)
			{
			}

			int populationSize;
			int numBestParents;
			double mutationRate;
			double crossoverRate;
		};

		struct Statistics
		{
			Statistics() : 
				numMutations(0),
				numMutationTries(0),
				numCrossovers(0),
				numCrossoverTries(0)
			{
			}

			int numMutations; 
			int numMutationTries;
			int numCrossovers;
			int numCrossoverTries;
			double deltaBestScores;
			double deltaWorstScores;
			std::vector<double> bestCandidateScores;
			std::vector<double> worstCandidateScores;

			void update()
			{
				int size = bestCandidateScores.size();
				deltaBestScores = size < 2 ? 
					0.0 : bestCandidateScores[size-1] - bestCandidateScores[size-2];
				size = worstCandidateScores.size();
				deltaWorstScores = size < 2 ?
					0.0 : worstCandidateScores[size - 1] - worstCandidateScores[size - 2];

			}

			void print()
			{
				std::cout << "--- Iteration Statistics ---" << std::endl;
				std::cout << "Mutations: " << numMutations << " Tried: " << numMutationTries << " (" << (double)numMutations / (double)numMutationTries * 100.0 << "%)" << std::endl;
				std::cout << "Crossovers: " << numCrossovers << " Tried: " << numCrossoverTries << " (" << (double)numCrossovers / (double)numCrossoverTries * 100.0 << "%)" << std::endl;
				std::cout << "Score Delta Best: " << deltaBestScores << " Worst: " << deltaWorstScores << std::endl;
			}
		};

		struct Result
		{
			Result()
			{
			}

			Result(const std::vector<RankedCreature>& population, const Statistics& statistics) :
				population(population),
				statistics(statistics)
			{
			}

			std::vector<RankedCreature> population;
			Statistics statistics;
		};

		GeneticAlgorithm()
		{
			_rndEngine.seed(_rndDevice());
		}
	
		void stop()
		{
			std::cout << "Stop requested." << std::endl;
			_stopRequested.store(true);
		}

		std::future<Result> runAsync(const Parameters& params, const ParentSelector& parentSelector, const CreatureCreator& creator, const CreatureRanker& ranker, const StopCriterion& stopCriterion)
		{
			return std::async(std::launch::async, [&]() 
			{ 
				return run(params, parentSelector, creator, ranker, stopCriterion);
			});
		}

		Result run(const Parameters& params, const ParentSelector& parentSelector, const CreatureCreator& creator, const CreatureRanker& ranker, const StopCriterion& stopCriterion) const
		{
			Statistics stats;

			auto population = createRandomPopulation(params.populationSize, creator);
		
			std::cout << "Random population with " << population.size() << " creatures was created." << std::endl;

			int iterationCount = 0;
			_stopRequested.store(false);

			while (!stopCriterion.shouldStop(population, iterationCount) && !_stopRequested.load())
			{
				std::cout << "Start iteration " << std::endl;

				rankAndSortPopulation(population, ranker);

				std::cout << "Best: " << population.front().rank << " Worst: " << population.back().rank << std::endl;
				stats.bestCandidateScores.push_back(population.front().rank);
				stats.worstCandidateScores.push_back(population.back().rank);
				
				population.front().creature.write("tree_tmp.dot");

				auto newPopulation = getNBestParents(population, params.numBestParents);

				while (newPopulation.size() < params.populationSize)
				{
					auto parent1 = parentSelector.selectFrom(population);
					auto parent2 = parentSelector.selectFrom(population);

					auto offspring = crossover(parent1, parent2, params.crossoverRate, creator, stats);
					
					newPopulation.push_back(mutate(offspring[0], params.mutationRate, creator, stats));
					newPopulation.push_back(mutate(offspring[1], params.mutationRate, creator, stats));
				}
				
				population = newPopulation; 

				stats.update();
				stats.print();

				iterationCount++;
			}

			return Result(population, stats);
		}

	private:

		RankedCreature mutate(const RankedCreature& creature, double mutationRate, const CreatureCreator& creator, Statistics& stats) const
		{
			stats.numMutationTries++;

			static std::bernoulli_distribution d{};
			using parm_t = decltype(d)::param_type;

			if (d(_rndEngine, parm_t{ mutationRate }))
			{
				stats.numMutations++;

				return RankedCreature(creator.mutate(creature.creature), RankedCreature::unranked());
			}
			else
			{
				return creature;
			}
		}

		std::vector<RankedCreature> crossover(const RankedCreature& parent1, const RankedCreature& parent2, double crossoverRate, const CreatureCreator& creator, Statistics& stats) const
		{
			stats.numCrossoverTries++;

			static std::bernoulli_distribution d{};
			using parm_t = decltype(d)::param_type;

			if (d(_rndEngine, parm_t{ crossoverRate }))
			{
				stats.numCrossovers++;

				auto crs = creator.crossover(parent1.creature, parent2.creature);
				
				std::vector<RankedCreature> rankedCrs;
				rankedCrs.reserve(crs.size());
				for (const auto& cr : crs)
					rankedCrs.push_back(RankedCreature(cr, RankedCreature::unranked()));
				
				return rankedCrs;
			}
			else
			{
				return { parent1, parent2 };
			}
		}

		std::vector<RankedCreature> getNBestParents(const std::vector<RankedCreature>& population, int numBestParents) const
		{
			//Assumption: popuplation size > numBestCreatures
			return std::vector<RankedCreature>(population.begin(), population.begin() + numBestParents);
		}

		std::vector<RankedCreature> createRandomPopulation(int populationSize, const CreatureCreator& creator) const
		{
			std::vector<RankedCreature> population;
			population.reserve(populationSize);

			for (int i = 0; i < populationSize; ++i)
			{
				auto creature = creator.create();
				population.push_back(RankedCreature(creature, RankedCreature::unranked()));
			}
			return population;
		}

		std::vector<RankedCreature> rankAndSortPopulation(std::vector<RankedCreature>& population, const CreatureRanker& ranker) const 
		{
			std::cout << "Rank population." << std::endl;

			double rankSum = 0.0;
			for (auto& c : population)
			{	
				c.rank = ranker.rank(c.creature);
				rankSum += c.rank;
			}

			//normalize rank
			for (auto& c : population)			
				c.rank /= rankSum;
			
			std::cout << "Sort population." << std::endl;

			//sort by rank
			std::sort(population.begin(), population.end(),
				[](const RankedCreature& a, const RankedCreature& b) -> bool
			{
				return a.rank < b.rank; 
			});

			return population;
		}

		mutable std::default_random_engine _rndEngine;
		mutable std::random_device _rndDevice;
		mutable std::atomic<bool> _stopRequested;
	};

	struct ImplicitFunction;

	struct CSGTreeCreator
	{
		CSGTreeCreator(const std::vector<std::shared_ptr<ImplicitFunction>>& functions, double createNewRandomProb = 0.5, double subtreeProb = 0.7, int maxTreeDepth = 10);

		CSGTree mutate(const CSGTree& tree) const;
		std::vector<CSGTree> crossover(const CSGTree& tree1, const CSGTree& tree2) const;
		CSGTree create() const;
		CSGTree create(int maxDepth) const;

	private: 

		void create(CSGTree& tree, int maxDepth, int curDepth) const;

		int getRndFuncIndex(const std::vector<int>& usedFuncIndices) const;

		double _createNewRandomProb;
		double _subtreeProb; 
		int _maxTreeDepth;
		std::vector<std::shared_ptr<ImplicitFunction>> _functions;
		mutable std::default_random_engine _rndEngine;
		mutable std::random_device _rndDevice;
	};

	struct CSGTreeRanker
	{
		CSGTreeRanker(double lambda, const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& functions);

		double rank(const CSGTree& tree) const;

	private:
		double _lambda;
		std::vector<std::shared_ptr<lmu::ImplicitFunction>> _functions;
	};

	using CSGTreeTournamentSelector = TournamentSelector<RankedCreature<CSGTree>>;

	using CSGTreeIterationStopCriterion = IterationStopCriterion<RankedCreature<CSGTree>>;

	using CSGTreeGA = GeneticAlgorithm<CSGTree, CSGTreeCreator, CSGTreeRanker>;
}

#endif 