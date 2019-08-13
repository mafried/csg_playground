#include <Eigen/Dense>

#include "pointcloud.h"
#include "primitives.h"

#ifndef CLUSTER_H
#define CLUSTER_H

namespace lmu
{	
	struct Cluster
	{
		Cluster(const PointCloud& pc, int label, const std::set<ManifoldType>& types) :
			pc(pc),
			label(label),
			manifoldTypes(types)
		{
		}

		PointCloud pc; 
		int label;
		std::set<ManifoldType> manifoldTypes;
	};

	std::vector<Cluster> readClusterFromFile(const std::string& file, double scaleFactor);
}

#endif