#include "..\include\collision.h"
#include "..\include\mesh.h"
#include "igl/copyleft/cgal/intersect_other.h"
#include "igl/bounding_box.h"

bool lmu::collides(const lmu::ImplicitFunction & f1, const lmu::ImplicitFunction & f2)
{
	if (f1.type() == ImplicitFunctionType::Sphere && f2.type() == ImplicitFunctionType::Sphere)
		return collides(static_cast<const lmu::IFSphere&>(f1), static_cast<const lmu::IFSphere&>(f2));
	else //mesh-mesh collision check fallback.
		return collides(f1.meshCRef(), f2.meshCRef(), true);

	return false;
}

bool lmu::collides(const lmu::IFSphere & f1, const lmu::IFSphere & f2)
{
	return (f1.pos() - f2.pos()).squaredNorm() < (f1.radius() + f2.radius())*(f1.radius() + f2.radius());
}

bool lmu::collides(const lmu::Mesh& m1, const lmu::Mesh& m2, bool fastWithBBox)
{
	if (fastWithBBox)
	{
		Eigen::Vector3d max1 = m1.vertices.colwise().maxCoeff();
		Eigen::Vector3d min1 = m1.vertices.colwise().minCoeff();

		Eigen::Vector3d max2 = m2.vertices.colwise().maxCoeff();
		Eigen::Vector3d min2 = m2.vertices.colwise().minCoeff();

		double radius1 = (max1 - min1).norm() / 4.0; //This is bullshit but works for the generator use case.
		double radius2 = (max2 - min2).norm() / 4.0; //This is bullshit but works for the generator use case.

		Eigen::Vector3d pos1 = min1 + ((max1 - min1) / 2.0);
		Eigen::Vector3d pos2 = min2 + ((max2 - min2) / 2.0);
		
		return (pos1 - pos2).squaredNorm() < (radius1 + radius2)*(radius1 + radius2);
	}
	else
	{
		Eigen::MatrixXi iF;
		return igl::copyleft::cgal::intersect_other(m1.vertices, m1.indices, m2.vertices, m2.indices, true, iF);
	}
}
