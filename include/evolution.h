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
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include <omp.h>

#include "helper.h"

namespace lmu
{
	enum class ScheduleType
	{
		LOG,
		EXP,
		IDENTITY
	};

	ScheduleType scheduleTypeFromString(std::string scheduleType);
	
	struct Schedule
	{
		Schedule() : type(ScheduleType::IDENTITY)
		{
		}

		Schedule(ScheduleType type) : 
			type(type)
		{
		}

		ScheduleType type; 
		
		//ignore for now.
		//double lam; 
		//int limit;

		double getFactor(int iteration) const
		{
			switch (type)
			{
			case ScheduleType::LOG:
				return logSchedule(iteration);
			case ScheduleType::EXP:
				return expSchedule(iteration);
			default:
			case ScheduleType::IDENTITY:
				return identitySchedule(iteration);
			}
		}

		// schedules
		static double expSchedule(int t, double lam = 0.005, int limit = 100000)
		{
			if (t >= limit) return 0.0;

			return std::exp(-lam*t);
		}

		static double logSchedule(int t, double lam = 1.0, int limit = 100000)
		{
			if (t >= limit) return 0.0;

			return std::log(2.0) / std::log(lam*t + 1);
		}

		static double identitySchedule(int t, double lam = 1.0, int limit = 100000)
		{
			return 1.0;
		}
	};

	template<typename Creature, typename Rank = double>
	struct RankedCreature
	{
		RankedCreature(const Creature& c, Rank rank = unranked()) :
			creature(c),
			rank(rank)
		{
		}		

		static Rank unranked() 
		{
			return Rank(-std::numeric_limits<double>::max());
		}

		Creature creature;
		Rank rank;		
	};
		
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

					//while (_dropWorstPossible && population[idx].rank == lmu::worstRank)
					//{
					//	std::cout << "Dropped from tournament." << std::endl;
					//	idx = d(_rndEngine, parm_t{ 0, (int)population.size() - 1 });
					//
					//	if (dropWorstCounter++ > population.size())
					//	{
					//		dropWorstCounter = 0;
					//		break;
					//	}
					//}
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
			//std::cout << "Iteration " << iterationCount << " of " << _maxIterations << std::endl;
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

	template<typename RankedCreature, typename Rank = double>
	struct NoFitnessIncreaseStopCriterion
	{
		NoFitnessIncreaseStopCriterion(int maxCount, Rank delta, int maxIterations) :
			_maxCount(maxCount),
			_delta(delta),
			_maxIterations(maxIterations),
			_currentCount(0),
			_lastBestRank(0.0)
		{
		}

		bool shouldStop(const std::vector<RankedCreature>& population, int iterationCount)
		{
			//std::cout << "Iteration " << iterationCount << std::endl;

			if (iterationCount >= _maxIterations)
				return true;

			if (population.empty())
				return true;

			Rank currentBestRank = population[0].rank;

			if(currentBestRank - _lastBestRank < _delta)
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
			ss << "No Change Stop Criterion Selector (maxCount=" << _maxCount << ", delta="<< _delta << ", maxIterations="<< _maxIterations << ")";
			return ss.str();
		}

	private:
		int _maxCount;
		int _currentCount;
		int _maxIterations;
		Rank _delta;
		Rank _lastBestRank;
	};

	template<typename RankedCreature>
	struct EmptyPopulationManipulator
	{
		void manipulateBeforeRanking(std::vector<RankedCreature>& population) const
		{
		}

		void manipulateAfterRanking(std::vector<RankedCreature>& population) const
		{
		}

		std::string info() const
		{
			return "Empty Population Manipulator";
		}
	};

	template<
		typename Creature, typename CreatureCreator, typename CreatureRanker, typename Rank = double,
		typename ParentSelector = TournamentSelector<RankedCreature<Creature, Rank>>,
		typename StopCriterion = IterationStopCriterion<RankedCreature<Creature, Rank>>,
		typename PopulationManipulator = EmptyPopulationManipulator<RankedCreature<Creature, Rank>>
	>		
	class GeneticAlgorithm
	{
	public:

		using RankedCreature = RankedCreature<Creature, Rank>;

		struct Parameters
		{
			Parameters(int populationSize, int numBestParents, double mutationRate, double crossoverRate, bool rankingInParallel, const Schedule& crossoverSchedule, const Schedule& mutationSchedule, bool useCaching) :
				populationSize(populationSize),
				numBestParents(numBestParents),
				mutationRate(mutationRate),
				crossoverRate(crossoverRate),
				rankingInParallel(rankingInParallel),
				crossoverSchedule(crossoverSchedule),
				mutationSchedule(mutationSchedule),
				useCaching(useCaching)
			{
			}

			std::string info() const
			{
				std::stringstream ss;
				ss << "Population Size: " << populationSize <<
					" Num Best Parents: " << numBestParents <<
					" Mutation Rate: " << mutationRate <<
					" Crossover Rate: " << crossoverRate <<
					" Ranking in parallel: " << rankingInParallel <<
					" Use Caching: " << useCaching;
				return ss.str();
			}

			int populationSize;
			int numBestParents;
			double mutationRate;
			double crossoverRate;
			bool rankingInParallel;
			Schedule crossoverSchedule;
			Schedule mutationSchedule;
			bool useCaching;
		};

		struct Statistics
		{
			Statistics(const std::string& info = std::string()) :
				info(info),
				numMutations(0),
				numMutationTries(0),
				numCrossovers(0),
				numCrossoverTries(0),
				numCacheHits(0),
				numCacheTries(0),
				bestScore(0.0),
				worstScore(std::numeric_limits<double>::max())
			{
			}

			std::string info;
			int numMutations;
			int numMutationTries;
			int numCrossovers;
			int numCrossoverTries;

			int numCacheHits; 
			int numCacheTries;

			Rank bestScore;
			Rank worstScore;
			std::vector<Rank> bestCandidateScores;

			std::vector<Rank> worstCandidateScores;
			std::vector<long long> rankingDurations;
			std::vector<long long> sortingDurations;
			std::vector<long long> optDurations;

			std::vector<long long> scmDurations;
		
			TimeTicker totalDuration, iterationDuration;

			void update()
			{
				bestScore = bestCandidateScores.empty() ? Rank(0.0) : bestCandidateScores.back();
				worstScore = worstCandidateScores.empty() ? Rank(0.0) : worstCandidateScores.back();
			}

			void print()
			{
				//std::cout << "--- Iteration Statistics ---" << std::endl;
				//std::cout << "Mutations: " << numMutations << " Tried: " << numMutationTries << " (" << (double)numMutations / (double)numMutationTries * 100.0 << "%)" << std::endl;
				//std::cout << "Crossovers: " << numCrossovers << " Tried: " << numCrossoverTries << " (" << (double)numCrossovers / (double)numCrossoverTries * 100.0 << "%)" << std::endl;
				//std::cout << "Cache Hits: " << numCacheHits << " Tried: " << numCacheTries << " (" << (double)numCacheHits / (double)numCacheTries * 100.0 << "%)" << std::endl;

				std::cout << "Score Best: " << bestScore << std::endl;//" Worst: " << worstScore << std::endl;				
			}

			void save(std::ostream& stream, const Creature* bestCreature = nullptr)
			{			
				auto end = std::chrono::system_clock::now();
				std::time_t end_time = std::chrono::system_clock::to_time_t(end);
				stream << "# Time: " << std::ctime(&end_time);

				std::istringstream iss(info);
				std::string line;
				while (std::getline(iss, line))
				{
					stream << "# " << line << std::endl;
				}
				
				stream << "# Duration: " << totalDuration.current << std::endl;

				if (bestCreature)
				{
					stream << "# Best Candidate: " << std::endl;

					stream << bestCreature->info() << std::endl;
				}

				stream << "# iteration    best candidate score    worst candidate score    optimization durations    ranking durations    sorting durations    scm durations" << std::endl;
				
				for (int i = 0; i < bestCandidateScores.size(); ++i)
				{
					stream << i << " | " << bestCandidateScores[i] << " | " << worstCandidateScores[i] << " | " << optDurations[i] << " | " << rankingDurations[i] << " | "  << sortingDurations[i] << " | " << scmDurations[i] << std::endl;
				}
			}

			void save(const std::string& file, const Creature* bestCreature = nullptr)
			{
				std::cout << "Save statistics to file " << file << "." << std::endl;

				std::ofstream fs(file);

				save(fs, bestCreature);

				fs.close();

				std::cout << "Done saving file " << std::endl;
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

		std::future<Result> runAsync(const Parameters& params, const ParentSelector& parentSelector, const CreatureCreator& creator, 
			const CreatureRanker& ranker, StopCriterion& stopCriterion, const PopulationManipulator& popMan)
		{
			return std::async(std::launch::async, [&]() 
			{ 
				return run(params, parentSelector, creator, ranker, stopCriterion, popMan);
			});
		}

		std::string assembleInfoString(const Parameters& params, const ParentSelector& parentSelector, const CreatureCreator& creator, 
			const CreatureRanker& ranker, const StopCriterion& stopCriterion, const PopulationManipulator& popMan) const
		{
			std::stringstream ss; 

			ss << "Parameters: " << params.info() << std::endl;
			ss << "Parent Selector: " << parentSelector.info() << std::endl;
			ss << "Creator: " << creator.info() << std::endl;
			ss << "Ranker: " << ranker.info() << std::endl;
			ss << "Stop Criterion: " << stopCriterion.info() << std::endl;
			ss << "Population Manipulator: " << popMan.info() << std::endl;
			
			return ss.str();
		}

		Result run(const Parameters& params, const ParentSelector& parentSelector, const CreatureCreator& creator, 
			const CreatureRanker& ranker, StopCriterion& stopCriterion, const PopulationManipulator& popMan = PopulationManipulator()) const
		{
			Statistics stats(assembleInfoString(params, parentSelector, creator, ranker, stopCriterion, popMan));
	
			//std::cout << "Create random population..." << std::endl;

			auto population = createRandomPopulation(params.populationSize, creator);
		
			//std::cout << "Random population with " << population.size() << " creatures was created." << std::endl;

			int iterationCount = 0;
			_stopRequested.store(false);

			double crossoverRate = params.crossoverRate;
			double mutationRate = params.mutationRate;

			while (!stopCriterion.shouldStop(population, iterationCount) && !_stopRequested.load())			
			{
				//std::cout << "Start iteration " << std::endl;
				stats.iterationDuration.reset();
				
				//std::cout << "Optimize population." << std::endl;
				popMan.manipulateBeforeRanking(population);
				stats.optDurations.push_back(stats.iterationDuration.tick());

				//std::cout << "Rank population." << std::endl;
				rankPopulation(population, ranker, params.rankingInParallel, params.useCaching, stats);
				stats.rankingDurations.push_back(stats.iterationDuration.tick());
				
				//std::cout << "After Rank manipulation." << std::endl;
				popMan.manipulateAfterRanking(population);
	
				sortPopulation(population);
				stats.sortingDurations.push_back(stats.iterationDuration.tick());
				
				stats.bestCandidateScores.push_back(population.front().rank);
				stats.worstCandidateScores.push_back(population.back().rank);
				
				auto newPopulation = getNBestParents(population, params.numBestParents);

				//std::cout << "Optimize population." << std::endl;
				//popMan.manipulateBeforeRanking(newPopulation);
				//stats.optDurations.push_back(stats.iterationDuration.tick());
				
				while (newPopulation.size() < params.populationSize)
				{
					auto parent1 = parentSelector.selectFrom(population);
					auto parent2 = parentSelector.selectFrom(population);
									
					auto offspring = crossover(parent1, parent2, crossoverRate, creator, stats);					
				
					newPopulation.push_back(mutate(offspring[0], mutationRate, creator, stats));
					newPopulation.push_back(mutate(offspring[1], mutationRate, creator, stats));
				}
				stats.scmDurations.push_back(stats.iterationDuration.tick());
				
				population = newPopulation; 
				stats.update();
				stats.print();
				iterationCount++;

				// Update the cross-over rate and mutation rate based on 
				// some annealing schedule
				crossoverRate = params.crossoverRate * params.crossoverSchedule.getFactor(iterationCount);
				mutationRate = params.mutationRate * params.mutationSchedule.getFactor(iterationCount);
			}

			//std::cout << "End." << std::endl;

			stats.totalDuration.tick();
						
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

		void rankPopulation(std::vector<RankedCreature>& population, const CreatureRanker& ranker, bool inParallel, bool useCaching, Statistics& stats) const 
		{	
			if (inParallel)
			{
				size_t numThreads = 0;
				omp_set_dynamic(0);
#ifndef _OPENMP 
				throw std::runtime_error("Ranking should run in parallel but OpenMP is not available.");
#endif
				if (useCaching)
				{
					//Collect all creatures that need to be ranked ( == not in cache).
					std::vector<std::tuple<size_t,size_t>> creaturesToRank;
					creaturesToRank.reserve(population.size());
					for (int i = 0; i < population.size(); ++i)
					{
						stats.numCacheTries++;
						size_t hash = population[i].creature.hash(0);

						auto it = _rankLookup.find(hash);
						if (it != _rankLookup.end())
						{
							stats.numCacheHits++;
							population[i].rank = it->second;
						}
						else
						{
							creaturesToRank.push_back(std::make_tuple(i, hash));
						}
					}

					//Rank creatures not found in cache.					
#pragma omp parallel for									
					for (int i = 0; i < creaturesToRank.size(); ++i)
					{
						numThreads = omp_get_num_threads();
						size_t index = std::get<0>(creaturesToRank[i]);
						Rank rank = ranker.rank(population[index].creature);
						population[index].rank = rank;

						size_t hash = std::get<1>(creaturesToRank[i]);

						std::lock_guard<std::mutex> l(_mutex);
						_rankLookup[hash] = rank;
					}

				}
				else //no caching.
				{

#pragma omp parallel for	
					for (int i = 0; i < population.size(); ++i)
					{
						numThreads = omp_get_num_threads();
						population[i].rank = ranker.rank(population[i].creature);
					}
				}

				//std::cout << "Num threads: " << numThreads << std::endl;
			}
			else // single threaded
			{
				int i = 0;
				for (auto& c : population)
				{
					c.rank = rankCreatureSingleThreaded(c.creature, ranker, useCaching, stats);													
				}
			}
		}

		inline Rank rankCreatureSingleThreaded(const Creature& c, const CreatureRanker& ranker, bool useCaching, Statistics& stats) const
		{
			if (!useCaching)
				return ranker.rank(c);

			stats.numCacheTries++;

			size_t hash = c.hash(0);
			
			auto it = _rankLookup.find(hash);
			if (it != _rankLookup.end())
			{
				stats.numCacheHits++;
				return it->second;
			}
			
			Rank rank = ranker.rank(c);			
			_rankLookup[hash] = rank;
			
			return rank;
		}

		void sortPopulation(std::vector<RankedCreature>& population) const
		{
			//std::cout << "Sort population." << std::endl;

			//sort by rank
			std::sort(population.begin(), population.end(),
				[](const RankedCreature& a, const RankedCreature& b)
			{
				return a.rank > b.rank;
			});
		}

		mutable std::unordered_map<size_t, Rank> _rankLookup;
		mutable std::default_random_engine _rndEngine;
		mutable std::random_device _rndDevice;
		mutable std::atomic<bool> _stopRequested;
		mutable std::mutex _mutex;
	};

	struct ImplicitFunction;

	
}

#endif 
