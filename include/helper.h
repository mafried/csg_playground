#ifndef HELPER_H
#define HELPER_H

#include <cassert>
#include <chrono>

#include <algorithm>
#include <random>

namespace lmu
{
	std::default_random_engine rndEngine();

	//From http://en.cppreference.com/w/cpp/numeric/math/acos
	//template<class T>
	//constexpr const T& clamp(const T& v, const T& lo, const T& hi)
	//{
	//	return clamp(v, lo, hi, std::less<>());
	//}

	template<class T, class Compare>
	constexpr const T& clamp(const T& v, const T& lo, const T& hi, Compare comp)
	{
		return assert(!comp(hi, lo)),
			comp(v, lo) ? lo : comp(hi, v) ? hi : v;
	}

    //From http://en.cppreference.com/w/cpp/numeric/math/acos
    template<class T>
    constexpr const T& clamp(const T& v, const T& lo, const T& hi)
    {
      return clamp(v, lo, hi, std::less<T>());
    }

	struct TimeTicker
	{
		TimeTicker() : 
			_time(std::chrono::high_resolution_clock::now())
		{

		}
		long long tick()
		{
			current = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - _time).count();
			reset();

			return current;
		}

		void reset()
		{
			_time = std::chrono::high_resolution_clock::now();
		}

		long long current;

	private:
		std::chrono::high_resolution_clock::time_point _time;

	};

	//https://stackoverflow.com/questions/5095407/all-combinations-of-k-elements-out-of-n
	template <typename Iterator>
	inline bool next_combination(const Iterator first, Iterator k, const Iterator last)
	{
		/* Credits: Thomas Draper */
		if ((first == last) || (first == k) || (last == k))
			return false;
		Iterator itr1 = first;
		Iterator itr2 = last;
		++itr1;
		if (last == itr1)
			return false;
		itr1 = last;
		--itr1;
		itr1 = k;
		--itr2;
		while (first != itr1)
		{
			if (*--itr1 < *itr2)
			{
				Iterator j = k;
				while (!(*itr1 < *j)) ++j;
				std::iter_swap(itr1, j);
				++itr1;
				++j;
				itr2 = k;
				std::rotate(itr1, j, last);
				while (last != j)
				{
					++j;
					++itr2;
				}
				std::rotate(k, itr2, last);
				return true;
			}
		}
		std::rotate(first, k, last);
		return false;
	}
}

#endif