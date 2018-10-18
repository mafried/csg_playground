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

#include <omp.h>

#include "helper.h"

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
			Parameters(int populationSize, int numBestParents, double mutationRate, double crossoverRate, bool rankingInParallel, 
				const std::function<void(const std::vector<RankedCreature>&)>& popInsp = [](const std::vector<RankedCreature>&) {return; }) :
				populationSize(populationSize),
				numBestParents(numBestParents),
				mutationRate(mutationRate),
				crossoverRate(crossoverRate),
				rankingInParallel(rankingInParallel),
				populationInspector(popInsp)
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

			std::function<void(const std::vector<RankedCreature>&)> populationInspector;

		};

		struct Statistics
		{
			Statistics(const std::string& info = std::string()) :
				info(info),
				numMutations(0),
				numMutationTries(0),
				numCrossovers(0),
				numCrossoverTries(0)	
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
			std::vector<long long> rankingDurations;
			std::vector<long long> sortingDurations;

			std::vector<long long> scmDurations;
		
			TimeTicker totalDuration, iterationDuration;

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
			}

			void save(const std::string& file, const Creature* bestCreature = nullptr)
			{
				std::cout << "Save statistics to file " << file << "." << std::endl;

				std::ofstream fs(file);

				auto end = std::chrono::system_clock::now();
				std::time_t end_time = std::chrono::system_clock::to_time_t(end);
				fs << "# Time: " << std::ctime(&end_time);

				std::istringstream iss(info);
				std::string line;
				while (std::getline(iss, line))
				{
					fs << "# " << line << std::endl;
				}
				
				fs << "# Duration: " << totalDuration.current << std::endl;

				if (bestCreature)
				{
					fs << "# Best Candidate: " << std::endl;

					fs << bestCreature->info() << std::endl;

					fs << "# iteration    best candidate score    worst candidate score    ranking durations    sorting durations    scm durations" << std::endl;
				}

				for (int i = 0; i < bestCandidateScores.size(); ++i)
				{
					fs << i << " " << bestCandidateScores[i] << " " << worstCandidateScores[i] << " " << rankingDurations[i] << " "  << sortingDurations[i] << " " << scmDurations[i] << std::endl;
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
	
			auto population = createRandomPopulation(params.populationSize, creator);
		
			std::cout << "Random population with " << population.size() << " creatures was created." << std::endl;

			int iterationCount = 0;
			_stopRequested.store(false);

			double crossoverRate = params.crossoverRate;
			double mutationRate = params.mutationRate;

			while (!stopCriterion.shouldStop(population, iterationCount) && !_stopRequested.load())			
			{
				std::cout << "Start iteration " << std::endl;
				stats.iterationDuration.reset();
				
				rankPopulation(population, ranker, params.rankingInParallel);
				stats.rankingDurations.push_back(stats.iterationDuration.tick());

				params.populationInspector(population);

				sortPopulation(population);
				stats.sortingDurations.push_back(stats.iterationDuration.tick());
				
				stats.bestCandidateScores.push_back(population.front().rank);
				stats.worstCandidateScores.push_back(population.back().rank);
				
				auto newPopulation = getNBestParents(population, params.numBestParents);

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
				crossoverRate = params.crossoverRate * identitySchedule(iterationCount);
				mutationRate = params.mutationRate * identitySchedule(iterationCount);
			}

			stats.totalDuration.tick();
						
			return Result(population, stats);
		}

	private:
	  // schedules
	  static double expSchedule(int t, double lam=0.005, int limit=100000)
	  {
	    if (t >= limit) return 0.0;

	    return std::exp(-lam*t);
	  }

	  static double logSchedule(int t, double lam=1.0, int limit=100000)
	  {
	    if (t >= limit) return 0.0;
	    
	    return std::log(2.0)/std::log(lam*t + 1);
	  }

	  static double identitySchedule(int t, double lam = 1.0, int limit = 100000)
	  {
		  return 1.0;
	  }


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

		void rankPopulation(std::vector<RankedCreature>& population, const CreatureRanker& ranker, bool inParallel = false) const 
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
		}

		void sortPopulation(std::vector<RankedCreature>& population) const
		{
			std::cout << "Sort population." << std::endl;

			//sort by rank
			std::sort(population.begin(), population.end(),
				[](const RankedCreature& a, const RankedCreature& b) -> bool
			{
				return a.rank > b.rank;
			});
		}

		mutable std::default_random_engine _rndEngine;
		mutable std::random_device _rndDevice;
		mutable std::atomic<bool> _stopRequested;
	};

	struct ImplicitFunction;

	
}

#endif 
