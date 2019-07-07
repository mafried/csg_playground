#include "primitive_helper.h"

#include <vector>

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

Eigen::Vector3d lmu::getAnyPerpendicularVector(const Eigen::Vector3d& v)
{
	//Taken from https://stackoverflow.com/questions/41275311/a-good-way-to-find-a-vector-perpendicular-to-another-vector

	std::vector<Eigen::Vector3d> cvs =
	{
		Eigen::Vector3d(1, 0, 0),
		Eigen::Vector3d(0, 1, 0),
		Eigen::Vector3d(0, 0, 1)
	};

	int idxSmallestDot;
	double smallestDot = std::numeric_limits<double>::max();
	for (int i = 0; i < cvs.size(); ++i)
	{
		double dot = std::abs(cvs[i].dot(v));
		if (dot < smallestDot)
		{
			smallestDot = dot;
			idxSmallestDot = i;
		}
	}

	return cvs[idxSmallestDot] - (cvs[idxSmallestDot].dot(v) * v);
}
