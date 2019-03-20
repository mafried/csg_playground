#include "primitive_helper.h"

Eigen::Matrix3d lmu::getRotationMatrix(const Eigen::Vector3d & dir, const Eigen::Vector3d& up)
{	
	Eigen::Vector3d f = dir;
	Eigen::Vector3d r = f.cross(up).normalized();
	Eigen::Vector3d u = r.cross(f).normalized();

	Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
	rot <<
		r.x(), f.x(), u.x(),
		r.y(), f.y(), u.y(),
		r.z(), f.z(), u.z();

	return rot;
}
