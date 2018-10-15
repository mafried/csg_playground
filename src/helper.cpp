#include "helper.h"

std::default_random_engine lmu::rndEngine()
{
	static std::default_random_engine eng;
	static std::random_device rd;
	eng.seed(rd());

	return eng;
}
