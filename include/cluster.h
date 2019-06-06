#include <Eigen/Dense>

#include "pointcloud.h"
#include "primitives.h"

#ifndef CLUSTER_H
#define CLUSTER_H

namespace lmu
{	
	struct Cluster
	{
		Cluster(const PointCloud& pc, int label, ManifoldType type) :
			pc(pc),
			label(label),
			manifoldType(type)
		{
		}

		PointCloud pc; 
		int label;
		ManifoldType manifoldType;
	};

	std::vector<Cluster> readClusterFromFile(const std::string& file, double scaleFactor);
}

#endif