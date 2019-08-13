#include "cluster.h"
#include "dbscan.h"
#include <fstream>
#include <iostream>

#include <boost/algorithm/string.hpp>

using namespace lmu;

std::vector<Cluster> lmu::readClusterFromFile(const std::string& file, double scaleFactor)
{
	std::vector<Cluster> res;

	std::ifstream s(file);
	
	size_t numClusters;
	s >> numClusters;

	//std::cout << "Num Clusters: " << numClusters << std::endl;

	for (int i = 0; i < numClusters; ++i)
	{
		std::string manifoldTypesStr;
		s >> manifoldTypesStr;
		//std::cout << "types: " << manifoldTypesStr << std::endl;

		std::set<ManifoldType> manifoldTypes;
		std::vector<std::string> strTypes;
		boost::split(strTypes, manifoldTypesStr, [](char c) {return c == ';'; });
		std::transform(strTypes.begin(), strTypes.end(), std::inserter(manifoldTypes, manifoldTypes.end()),
			[](const std::string& s) -> ManifoldType { return  lmu::fromPredictedTypeType(std::stoi(s)); });

		for (const auto& mt : manifoldTypes)
		{
			std::cout << manifoldTypeToString(mt) << std::endl;
		}

		res.push_back(Cluster(readPointCloud(s, scaleFactor), i, manifoldTypes));
	}

	return res;
}
