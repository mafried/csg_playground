#ifndef PC_STRUCTURE_H
#define PC_STRUCTURE_H

#include "pointcloud.h"
#include "primitives.h"

namespace lmu
{
	lmu::PointCloud structure_pointcloud(const lmu::ManifoldSet& ms, double epsilon);
}

#endif