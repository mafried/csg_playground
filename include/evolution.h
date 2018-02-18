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
#include <chrono>
#include <iostream>
#include <fstream>

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

	const double worstRank = -std::numeric_limits<double>::max();

	template<typename RankedCreature>
	struct TournamentSelector
	{
		TournamentSelector(int k, bool dropWorstPossible = false) :
			_k(k), 
			_dropWorstPossible(dropWorstPossible)
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
					while (firstRun && _dropWorstPossible && population[idx].rank == lmu::worstRank)
					{
						std::cout << "Dropped from tournament." << std::endl;
						idx = d(_rndEngine, parm_t{ 0, (int)population.size() - 1 });
					}
					if (firstRun || population[idx].rank > population[best].rank)
					{					
						firstRun = false;
						best = idx;
					}
				}

				return population[best];
			}

			std::string info() const
			{
				std::stringstream ss;
				ss << "Tournament Selector (k=" << _k << ")";
				return ss.str(); 
			}

	private:
		int _k;
		bool _dropWorstPossible;
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

		std::string info() const
		{
			std::stringstream ss;
			ss << "Iteration Stop Criterion Selector (n=" << _maxIterations << ")";
			return ss.str(); 
		}

	private:
		int _maxIterations;
	};

	template<typename RankedCreature>
	struct NoFitnessIncreaseStopCriterion
	{
		NoFitnessIncreaseStopCriterion(int maxCount, double delta, int maxIterations) :
			_maxCount(maxCount),
			_delta(delta),
			_maxIterations(maxIterations),
			_currentCount(0),
			_lastBestRank(0.0)
		{
		}

		bool shouldStop(const std::vector<RankedCreature>& population, int iterationCount)
		{
			std::cout << "Iteration " << iterationCount << std::endl;

			if (iterationCount >= _maxIterations)
				return true;

			if (population.empty())
				return true;

			double currentBestRank = population[0].rank;

			if(currentBestRank - _lastBestRank <= _delta)
			{
				//No change
				_currentCount++;
			}
			else
			{
				_currentCount = 0;
			}

			_lastBestRank = currentBestRank;

			return _currentCount >= _maxCount;
		}

		std::string info() const
		{
			std::stringstream ss;
			ss << "No Change Stop Criterion Selector (maxCount=" << _maxCount << ", delta="<< _delta << ", "<< _maxIterations << ")";
			return ss.str();
		}

	private:
		int _maxCount;
		int _currentCount;
		int _maxIterations;
		double _delta;
		double _lastBestRank;
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
			Parameters(int populationSize, int numBestParents, double mutationRate, double crossoverRate, bool rankingInParallel) :
				populationSize(populationSize),
				numBestParents(numBestParents),
				mutationRate(mutationRate),
				crossoverRate(crossoverRate),
				rankingInParallel(rankingInParallel)
			{
			}

			std::string info() const
			{
				std::stringstream ss;
				ss << "Population Size: " << populationSize <<
					" Num Best Parents: " << numBestParents <<
					" Mutation Rate: " << mutationRate <<
					" Crossover Rate: " << crossoverRate <<
					" Ranking in parallel: " << rankingInParallel;
				return ss.str();
			}

			int populationSize;
			int numBestParents;
			double mutationRate;
			double crossoverRate;
			bool rankingInParallel;
		};

		struct Statistics
		{
			Statistics(const std::string& info = std::string()) :
				info(info),
				numMutations(0),
				numMutationTries(0),
				numCrossovers(0),
				numCrossoverTries(0),
				duration(0)
			{
			}

			std::string info;
			int numMutations;
			int numMutationTries;
			int numCrossovers;
			int numCrossoverTries;
			double deltaBestScores;
			double deltaWorstScores;
			std::vector<double> bestCandidateScores;
			std::vector<double> worstCandidateScores;

			long duration;
			std::chrono::high_resolution_clock::time_point time;
			void durationTick()
			{
				if (time != std::chrono::high_resolution_clock::time_point())
				{
					duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - time).count();
					time = std::chrono::high_resolution_clock::time_point();
				}
				else
				{
					time = std::chrono::high_resolution_clock::now();
				}
			}

			void update()
			{
				int size = bestCandidateScores.size();
				deltaBestScores = size < 2 ?
					0.0 : bestCandidateScores[size - 1] - bestCandidateScores[size - 2];
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
				std::cout << "Duration: " << duration << std::endl;
			}

			void save(const std::string& file)
			{
				std::cout << "Save statistics to file " << file << "." << std::endl;

				std::ofstream fs(file);

				std::istringstream iss(info);
				std::string line;
				while (std::getline(iss, line))
				{
					fs << "# " << line << std::endl;
				}

				fs << "# Duration: " << duration << std::endl;

				fs << "# iteration    best candidate score    worst candidate score" << std::endl;

				for (int i = 0; i < bestCandidateScores.size(); ++i)
				{
					fs << i << " " << bestCandidateScores[i] << " " << worstCandidateScores[0] << std::endl;
				
				}

				fs.close();
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

		std::future<Result> runAsync(const Parameters& params, const ParentSelector& parentSelector, const CreatureCreator& creator, const CreatureRanker& ranker, StopCriterion& stopCriterion)
		{
			return std::async(std::launch::async, [&]() 
			{ 
				return run(params, parentSelector, creator, ranker, stopCriterion);
			});
		}

		std::string assembleInfoString(const Parameters& params, const ParentSelector& parentSelector, const CreatureCreator& creator, const CreatureRanker& ranker, const StopCriterion& stopCriterion) const
		{
			std::stringstream ss; 

			ss << "Parameters: " << params.info() << std::endl;
			ss << "Parent Selector: " << parentSelector.info() << std::endl;
			ss << "Creator: " << creator.info() << std::endl;
			ss << "Ranker: " << ranker.info() << std::endl;
			ss << "Stop Criterion: " << stopCriterion.info() << std::endl;

			return ss.str();
		}

		Result run(const Parameters& params, const ParentSelector& parentSelector, const CreatureCreator& creator, const CreatureRanker& ranker, StopCriterion& stopCriterion) const
		{
			Statistics stats(assembleInfoString(params, parentSelector, creator, ranker, stopCriterion));

			stats.durationTick();

			auto population = createRandomPopulation(params.populationSize, creator);
		
			std::cout << "Random population with " << population.size() << " creatures was created." << std::endl;

			int iterationCount = 0;
			_stopRequested.store(false);

			while (!stopCriterion.shouldStop(population, iterationCount) && !_stopRequested.load())
			{
				std::cout << "Start iteration " << std::endl;

				rankAndSortPopulation(population, ranker, params.rankingInParallel);

				std::cout << "Best: " << population.front().rank << " Worst: " << population.back().rank << std::endl;
				stats.bestCandidateScores.push_back(population.front().rank);
				stats.worstCandidateScores.push_back(population.back().rank);
				
				//population.front().creature.write("tree_tmp.dot");

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

			stats.durationTick();
			
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

		std::vector<RankedCreature> rankAndSortPopulation(std::vector<RankedCreature>& population, const CreatureRanker& ranker, bool inParallel = false) const 
		{
			std::cout << "Rank population." << std::endl;
						
			//double minRank = std::numeric_limits<double>::max();
			//double maxRank = -std::numeric_limits<double>::max();
			
			if (inParallel)
			{
#ifndef _OPENMP 
				throw std::runtime_error("Cliques should run in parallel but OpenMP is not available.");
#endif

#pragma omp parallel for
				for (int i = 0; i < population.size(); ++i)
				{
					population[i].rank = ranker.rank(population[i].creature);
				}
			}
			else
			{
				for (auto& c : population)
				{
					c.rank = ranker.rank(c.creature);
					//minRank = minRank < c.rank ? minRank : c.rank;
					//maxRank = maxRank > c.rank ? maxRank : c.rank;
				}
			}

			//normalize rank
			//for (auto& c : population)			
			//	c.rank = (c.rank - minRank) / (maxRank - minRank);
			
			std::cout << "Sort population." << std::endl;

			//sort by rank
			std::sort(population.begin(), population.end(),
				[](const RankedCreature& a, const RankedCreature& b) -> bool
			{
				return a.rank > b.rank; 
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
		CSGTreeCreator(const std::vector<std::shared_ptr<ImplicitFunction>>& functions, double createNewRandomProb = 0.5, double subtreeProb = 0.7, int maxTreeDepth = 10, const lmu::Graph& connectionGraph = lmu::Graph());

		CSGTree mutate(const CSGTree& tree) const;
		std::vector<CSGTree> crossover(const CSGTree& tree1, const CSGTree& tree2) const;
		CSGTree create() const;
		CSGTree create(int maxDepth) const;

		std::string info() const; 

	private: 

		void create(CSGTree& tree, int maxDepth, int curDepth) const;

		int getRndFuncIndex(const std::vector<int>& usedFuncIndices, const lmu::CSGTree& tree) const;

		double _createNewRandomProb;
		double _subtreeProb; 
		int _maxTreeDepth;
		std::vector<std::shared_ptr<ImplicitFunction>> _functions;
		mutable std::default_random_engine _rndEngine;
		mutable std::random_device _rndDevice;

		lmu::Graph _connectionGraph;
	};

	struct CSGTreeRanker
	{
		CSGTreeRanker(double lambda, const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& functions, const lmu::Graph& connectionGraph = lmu::Graph());

		double rank(const CSGTree& tree) const;
		std::string info() const;

		bool treeIsInvalid(const lmu::CSGTree& tree) const;

	private:
		double _lambda;
		std::vector<std::shared_ptr<lmu::ImplicitFunction>> _functions;
		bool _earlyOutTest;
		lmu::Graph _connectionGraph;
	};

	using CSGTreeTournamentSelector = TournamentSelector<RankedCreature<CSGTree>>;

	using CSGTreeIterationStopCriterion = IterationStopCriterion<RankedCreature<CSGTree>>;
	using CSGTreeNoFitnessIncreaseStopCriterion = NoFitnessIncreaseStopCriterion<RankedCreature<CSGTree>>;

	using CSGTreeGA = GeneticAlgorithm<CSGTree, CSGTreeCreator, CSGTreeRanker, CSGTreeTournamentSelector, CSGTreeNoFitnessIncreaseStopCriterion>;
}

#endif 