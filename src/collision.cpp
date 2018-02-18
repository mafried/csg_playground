#include "..\include\collision.h"
#include "..\include\mesh.h"
#include "igl/copyleft/cgal/intersect_other.h"

bool lmu::collides(const lmu::ImplicitFunction & f1, const lmu::ImplicitFunction & f2)
{
	if (f1.type() == ImplicitFunctionType::Sphere && f2.type() == ImplicitFunctionType::Sphere)
		return collides(static_cast<const lmu::IFSphere&>(f1), static_cast<const lmu::IFSphere&>(f2));
	else //mesh-mesh collision check fallback.
		throw collides(f1.meshCRef(), f2.meshCRef());

	return false;
}

bool lmu::collides(const lmu::IFSphere & f1, const lmu::IFSphere & f2)
{
	return (f1.pos() - f2.pos()).squaredNorm() < (f1.radius() + f2.radius())*(f1.radius() + f2.radius());
}

bool lmu::collides(const lmu::Mesh& m1, const lmu::Mesh& m2)
{
	Eigen::MatrixXi iF;

	return igl::copyleft::cgal::intersect_other(m1.vertices, m1.indices, m2.vertices, m2.indices, true, iF);
}
