#ifndef RANSAC_H
#define RANSAC_H

#include <vector>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "mesh.h"

namespace lmu
{
	void ransacWithSim(const Eigen::MatrixXd& points, const Eigen::MatrixXd& normals, double maxDelta, const std::vector<std::shared_ptr<ImplicitFunction>>& knownFunctions);
}

#endif