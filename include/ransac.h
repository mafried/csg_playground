#ifndef RANSAC_H
#define RANSAC_H

#include <vector>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "mesh.h"


namespace lmu
{
	struct CSGNodeSamplingParams;

	std::vector<std::shared_ptr<ImplicitFunction>> ransacWithCGAL(const Eigen::MatrixXd& points, const Eigen::MatrixXd& normals);

	//std::vector<std::shared_ptr<ImplicitFunction>> ransacWithPCL(const Eigen::MatrixXd& points, const Eigen::MatrixXd& normals);

	double ransacWithSim(const PointCloud& points, const CSGNodeSamplingParams& params, const std::vector<std::shared_ptr<ImplicitFunction>>& knownFunctions);
	void ransacWithSimMultiplePointOwners(const Eigen::MatrixXd& points, const Eigen::MatrixXd& normals, double maxDelta, const std::vector<std::shared_ptr<ImplicitFunction>>& knownFunctions);

}

#endif