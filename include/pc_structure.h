#ifndef PC_STRUCTURE_H
#define PC_STRUCTURE_H

#include "pointcloud.h"
#include "primitives.h"

namespace lmu
{
	void structure_pointcloud(const lmu::PointCloud& pc, const lmu::ManifoldSet& ms, double epsilon);
}

#endif