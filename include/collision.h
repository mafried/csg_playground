#include <Eigen/Dense>

#ifndef COLLISION_H
#define COLLISION_H

namespace lmu
{
	struct ImplicitFunction;
	struct IFSphere;
	struct Mesh;

	void getAABB(const lmu::ImplicitFunction& f, Eigen::Vector3d& pmin, Eigen::Vector3d& pmax);
	bool collidesAABB(const lmu::ImplicitFunction& f1, const lmu::ImplicitFunction& f2);

	bool collides(const lmu::ImplicitFunction& f1, const lmu::ImplicitFunction& f2);

	bool collides(const lmu::IFSphere& f1, const lmu::IFSphere& f2);

	bool collides(const lmu::Mesh& m1, const lmu::Mesh& m2);
}

#endif