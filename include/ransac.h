#ifndef RANSAC_H
#define RANSAC_H

#include <vector>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "mesh.h"

namespace lmu
{
	std::vector<std::shared_ptr<ImplicitFunction>> ransacWithCGAL(const Eigen::MatrixXd& points, const Eigen::MatrixXd& normals);

	std::vector<std::shared_ptr<ImplicitFunction>> ransacWithPCL(const Eigen::MatrixXd& points, const Eigen::MatrixXd& normals);

	void ransacWithSim(const Eigen::MatrixXd& points, const Eigen::MatrixXd& normals, double maxDelta, const std::vector<std::shared_ptr<ImplicitFunction>>& knownFunctions);
	void ransacWithSimMultiplePointOwners(const Eigen::MatrixXd& points, const Eigen::MatrixXd& normals, double maxDelta, const std::vector<std::shared_ptr<ImplicitFunction>>& knownFunctions);

}

#endif