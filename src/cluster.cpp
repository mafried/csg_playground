#include "cluster.h"
#include "dbscan.h"
#include <fstream>
#include <iostream>

#include <boost/algorithm/string.hpp>

#include "point_vis.h"
#include "pc_structure.h"

using namespace lmu;

std::vector<Cluster> lmu::readClusterFromFile(const std::string& file, double scaleFactor, bool with_header)
{
	std::vector<Cluster> res;

	std::ifstream s(file);

	if (with_header)
	{	
		s.ignore(10000, '\n');
		s.ignore(10000, '\n');
	}

	size_t numClusters;
	s >> numClusters;

	std::cout << "Num Clusters: " << numClusters << std::endl;

	for (int i = 0; i < numClusters; ++i)
	{
		std::string manifoldTypesStr;
		s >> manifoldTypesStr;
		//std::cout << "types: " << manifoldTypesStr << std::endl;

		std::set<ManifoldType> manifoldTypes;
		std::vector<std::string> strTypes;
		boost::split(strTypes, manifoldTypesStr, [](char c) {return c == ';'; });
		std::transform(strTypes.begin(), strTypes.end(), std::inserter(manifoldTypes, manifoldTypes.end()),
			[](const std::string& s) -> ManifoldType { return  lmu::fromPrimitiveType((PrimitiveType)std::stoi(s)); });

		for (const auto& mt : manifoldTypes)
		{
			std::cout << manifoldTypeToString(mt) << std::endl;
		}

		res.push_back(Cluster(readPointCloud(s, scaleFactor), i, manifoldTypes));
	}

	std::cout << "Num Clusters: " << res.size() << std::endl;

	return res;
}

std::vector<lmu::PointCloud> lmu::readRawPointCloudClusterFromFile(const std::string& file, double scale_factor)
{
	std::vector<lmu::PointCloud> res;

	std::ifstream s(file);

	size_t numClusters;
	s >> numClusters;

	for (int i = 0; i < numClusters; ++i)
	{
		res.push_back(readPointCloud(s, scale_factor));
	}

	std::cout << "Num Clusters: " << res.size() << std::endl;
		return res;
}
