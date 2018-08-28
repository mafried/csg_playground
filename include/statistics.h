#ifndef STATISTICS_H
#define STATISTICS_H

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

namespace lmu
{
	template<typename T>
	using DataFrame = std::vector<T>;

	template<typename T>
	struct KMeansResult
	{
		std::vector<size_t> assignments;
		DataFrame<T> means;
	};

	template<typename T>
	KMeansResult<T> k_means(const DataFrame<T>& data,
		size_t k,
		size_t number_of_iterations) {
		static std::random_device seed;
		static std::mt19937 random_number_generator(seed());
		std::uniform_int_distribution<size_t> indices(0, data.size() - 1);

		// Pick centroids as random points from the dataset.
		DataFrame<T> means(k);
		for (auto& cluster : means) {
			cluster = data[indices(random_number_generator)];
		}

		std::vector<size_t> assignments(data.size());
		for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) 
		{
			// Find assignments.
			for (size_t point = 0; point < data.size(); ++point) 
			{
				double best_distance = std::numeric_limits<double>::max();
				size_t best_cluster = 0;
				for (size_t cluster = 0; cluster < k; ++cluster) 
				{
					//  const double distance = squared_l2_distance(data[point], means[cluster]);
					const double distance = std::abs(data[point] - means[cluster]); //(data[point] - means[cluster]).squaredNorm();
					if (distance < best_distance) {
						best_distance = distance;
						best_cluster = cluster;
					}
				}
				assignments[point] = best_cluster;
			}

			// Sum up and count points for each cluster.
			DataFrame<T> new_means(k);
			std::vector<size_t> counts(k, 0);
			for (size_t point = 0; point < data.size(); ++point) 
			{
				const auto cluster = assignments[point];

				new_means[cluster] += data[point];
								
				//new_means[cluster].x += data[point].x;
				//new_means[cluster].y += data[point].y;
				counts[cluster] += 1;
			}

			// Divide sums by counts to get new centroids.
			for (size_t cluster = 0; cluster < k; ++cluster) 
			{
				// Turn 0/0 into 0/1 to avoid zero division.
				const auto count = std::max<size_t>(1, counts[cluster]);
				//means[cluster].x = new_means[cluster].x / count;
				//means[cluster].y = new_means[cluster].y / count;
			
				means[cluster] = new_means[cluster] / count;

			}
		}

		return { assignments, means };
	}
}

typedef std::size_t                  SizeT;
typedef SizeT                        CountType;
typedef std::pair<double, CountType> ValueCountPair;
typedef std::vector<double>          LimitsContainer;
typedef std::vector<ValueCountPair>  ValueCountPairContainer;

void GetValueCountPairs(ValueCountPairContainer& vcpc, const double* values, SizeT n);
void ClassifyJenksFisherFromValueCountPairs(LimitsContainer& breaksArray, SizeT k, const ValueCountPairContainer& vcpc);

#endif