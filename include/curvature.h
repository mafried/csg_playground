#ifndef CURVATURE_H
#define CURVATURE_H

#include <Eigen/Core>

#include "csgnode.h"

namespace lmu
{

	//double dx = (node.signedDistance(Eigen::Vector3d(ps.x() + h, ps.y(), ps.z())) - node.signedDistance(Eigen::Vector3d(ps.x() - h, ps.y(), ps.z()))) / (2.0 * h);
	//double dy = (node.signedDistance(Eigen::Vector3d(ps.x(), ps.y() + h, ps.z())) - node.signedDistance(Eigen::Vector3d(ps.x(), ps.y() - h, ps.z()))) / (2.0 * h);
	//double dz = (node.signedDistance(Eigen::Vector3d(ps.x(), ps.y(), ps.z() + h)) - node.signedDistance(Eigen::Vector3d(ps.x(), ps.y(), ps.z() - h))) / (2.0 * h);
	
	inline double dx(const Eigen::Vector3d& ps, const CSGNode& node, double h)
	{
		double dx = (node.signedDistance(Eigen::Vector3d(ps.x() + h, ps.y(), ps.z())) - node.signedDistance(Eigen::Vector3d(ps.x() - h, ps.y(), ps.z()))) / (2.0 * h);

		return dx;
	}

	inline double dy(const Eigen::Vector3d& ps, const CSGNode& node, double h)
	{
		double dy = (node.signedDistance(Eigen::Vector3d(ps.x(), ps.y() + h, ps.z())) - node.signedDistance(Eigen::Vector3d(ps.x(), ps.y() - h, ps.z()))) / (2.0 * h);
		return dy;
	}

	inline double dz(const Eigen::Vector3d& ps, const CSGNode& node, double h)
	{
		double dz = (node.signedDistance(Eigen::Vector3d(ps.x(), ps.y(), ps.z() + h)) - node.signedDistance(Eigen::Vector3d(ps.x(), ps.y(), ps.z() - h))) / (2.0 * h);
		return dz;
	}


	inline double dxx(const Eigen::Vector3d& ps, const CSGNode& node, double h)
	{
		double dxx = (dx(Eigen::Vector3d(ps.x() + h, ps.y(), ps.z()), node, h) - dx(Eigen::Vector3d(ps.x() - h, ps.y(), ps.z()), node, h)) / (2.0 * h);
		return dxx;
	}

	inline double dxy(const Eigen::Vector3d& ps, const CSGNode& node, double h)
	{
		double dxy = (dx(Eigen::Vector3d(ps.x(), ps.y() + h, ps.z()), node, h) - dx(Eigen::Vector3d(ps.x(), ps.y() - h, ps.z()), node, h)) / (2.0 * h);
		return dxy;
	}

	inline double dxz(const Eigen::Vector3d& ps, const CSGNode& node, double h)
	{
		double dxz = (dx(Eigen::Vector3d(ps.x(), ps.y(), ps.z() + h), node, h) - dx(Eigen::Vector3d(ps.x(), ps.y(), ps.z() - h), node, h)) / (2.0 * h);
		return dxz;
	}


	inline double dyx(const Eigen::Vector3d& ps, const CSGNode& node, double h)
	{
		double dyx = (dy(Eigen::Vector3d(ps.x() + h, ps.y(), ps.z()), node, h) - dy(Eigen::Vector3d(ps.x() - h, ps.y(), ps.z()), node, h)) / (2.0 * h);
		return dyx;
	}

	inline double dyy(const Eigen::Vector3d& ps, const CSGNode& node, double h)
	{
		double dyy = (dy(Eigen::Vector3d(ps.x(), ps.y() + h, ps.z()), node, h) - dy(Eigen::Vector3d(ps.x(), ps.y() - h, ps.z()), node, h)) / (2.0 * h);
		return dyy;
	}

	inline double dyz(const Eigen::Vector3d& ps, const CSGNode& node, double h)
	{
		double dyz = (dy(Eigen::Vector3d(ps.x(), ps.y(), ps.z() + h), node, h) - dy(Eigen::Vector3d(ps.x(), ps.y(), ps.z() - h), node, h)) / (2.0 * h);
		return dyz;
	}


	inline double dzx(const Eigen::Vector3d& ps, const CSGNode& node, double h)
	{
		double dzx = (dz(Eigen::Vector3d(ps.x() + h, ps.y(), ps.z()), node, h) - dz(Eigen::Vector3d(ps.x() - h, ps.y(), ps.z()), node, h)) / (2.0 * h);
		return dzx;
	}

	inline double dzy(const Eigen::Vector3d& ps, const CSGNode& node, double h)
	{
		double dzy = (dz(Eigen::Vector3d(ps.x(), ps.y() + h, ps.z()), node, h) - dz(Eigen::Vector3d(ps.x(), ps.y() - h, ps.z()), node, h)) / (2.0 * h);
		return dzy;
	}

	inline double dzz(const Eigen::Vector3d& ps, const CSGNode& node, double h)
	{
		double dzz = (dz(Eigen::Vector3d(ps.x(), ps.y(), ps.z() + h), node, h) - dz(Eigen::Vector3d(ps.x(), ps.y(), ps.z() - h), node, h)) / (2.0 * h);
		return dzz;
	}

	inline Eigen::Matrix<double, 3, 3> hessian(const Eigen::Vector3d& ps, const CSGNode& node, double h)
	{
		Eigen::Matrix<double, 3, 3> m; 

		m <<
			dxx(ps, node, h), dxy(ps, node, h), dxz(ps, node, h),
			dyx(ps, node, h), dyy(ps, node, h), dyz(ps, node, h),
			dzx(ps, node, h), dzy(ps, node, h), dzz(ps, node, h);

		return m;
	}

	inline Eigen::Matrix<double, 3, 3> adjunct(const Eigen::Matrix<double, 3, 3>& m)
	{
		Eigen::Matrix<double, 3, 3> r;

		r <<
			m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1), m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2), m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1),
			m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2), m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0), m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2),
			m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0), m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1), m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);

		return r; 
	}

	struct Curvature
	{
		double gaussCurv; 
		double meanCurv; 
		double k1; 
		double k2; 
	};

	Curvature curvature(const Eigen::Vector3d& ps, const CSGNode& node, double h);

	Eigen::MatrixXd filterPrimitivePointsByCurvature(const std::vector<ImplicitFunctionPtr>& funcs, double h, double t);

	Eigen::VectorXd computeCurvature(const Eigen::MatrixXd & samplePoints, const CSGNode & node, double h, bool normalize);

}

#endif 