#include "cluster.h"
#include "dbscan.h"
#include <fstream>
#include <iostream>

using namespace lmu;

std::vector<Cluster> lmu::readClusterFromFile(const std::string& file, double scaleFactor)
{
	std::vector<Cluster> res;

	std::ifstream s(file);
	
	size_t numClusters;
	s >> numClusters;

	std::cout << "Num Clusters: " << numClusters << std::endl;

	for (int i = 0; i < numClusters; ++i)
	{
		int manifoldType;
		s >> manifoldType;

		res.push_back(Cluster(readPointCloud(s, scaleFactor), i, (ManifoldType)manifoldType));
	}

	return res;
}
