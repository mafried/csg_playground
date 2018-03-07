#ifndef HELPER_H
#define HELPER_H

#include <cassert>
#include <chrono>

namespace lmu
{
	//From http://en.cppreference.com/w/cpp/numeric/math/acos
	template<class T>
	constexpr const T& clamp(const T& v, const T& lo, const T& hi)
	{
		return clamp(v, lo, hi, std::less<>());
	}

	template<class T, class Compare>
	constexpr const T& clamp(const T& v, const T& lo, const T& hi, Compare comp)
	{
		return assert(!comp(hi, lo)),
			comp(v, lo) ? lo : comp(hi, v) ? hi : v;
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
}

#endif