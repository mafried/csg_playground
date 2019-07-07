#ifndef PRIMITIVE_HELPER_H
#define PRIMITIVE_HELPER_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace lmu
{
	Eigen::Matrix3d getRotationMatrix(const Eigen::Vector3d& dir, const Eigen::Vector3d& up = Eigen::Vector3d(0, 0, 1));

	Eigen::Vector3d getAnyPerpendicularVector(const Eigen::Vector3d& v);

}

#endif 