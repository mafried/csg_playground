#include "..\include\collision.h"
#include "..\include\mesh.h"
#include "igl/copyleft/cgal/intersect_other.h"


void lmu::getAABB(const lmu::ImplicitFunction& f, Eigen::Vector3d& pmin, Eigen::Vector3d& pmax)
{	
	/*lmu::PointCloud pc = f.pointsCRef();

	double xmin = pc(0, 0), ymin = pc(0, 1), zmin = pc(0, 2);
	double xmax = xmin, ymax = ymin, zmax = zmin;
	for (int i = 1; i < pc.rows(); ++i) {
		xmin = (pc(i, 0) < xmin) ? pc(i, 0) : xmin;
		ymin = (pc(i, 1) < ymin) ? pc(i, 1) : ymin;
		zmin = (pc(i, 2) < zmin) ? pc(i, 2) : zmin;
		xmax = (pc(i, 0) > xmax) ? pc(i, 0) : xmax;
		ymax = (pc(i, 1) > ymax) ? pc(i, 1) : ymax;
		zmax = (pc(i, 2) > zmax) ? pc(i, 2) : zmax;
	}*/

	pmin = f.aabb().c - f.aabb().s;
	pmax = f.aabb().c + f.aabb().s;

	/*pmin(0) = xmin;
	pmin(1) = ymin;
	pmin(2) = zmin;
	pmax(0) = xmax;
	pmax(1) = ymax;
	pmax(2) = zmax;*/
}

bool lmu::collidesAABB(const lmu::ImplicitFunction& f1, const lmu::ImplicitFunction& f2)
{	
	Eigen::Vector3d pm1, pM1, pm2, pM2;

	lmu::getAABB(f1, pm1, pM1);
	lmu::getAABB(f2, pm2, pM2);

	return (pm1(0) <= pM2(0) && pM1(0) >= pm2(0)) &&
		(pm1(1) <= pM2(1) && pM1(1) >= pm2(1)) &&
		(pm1(2) <= pM2(2) && pM1(2) >= pm2(2));
}

bool lmu::collides(const lmu::ImplicitFunction & f1, const lmu::ImplicitFunction & f2)
{
	if (!f1.aabb().overlapsWith(f2.aabb(), 0.001)) return false;
		
	if (f1.type() == ImplicitFunctionType::Sphere && f2.type() == ImplicitFunctionType::Sphere)
		return collides(static_cast<const lmu::IFSphere&>(f1), static_cast<const lmu::IFSphere&>(f2));
	else //mesh-mesh collision check fallback.
		return collides(f1.meshCRef(), f2.meshCRef());

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
