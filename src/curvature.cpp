
#include <Eigen/Core>

#include "curvature.h"
#include "csgnode_helper.h"

//from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.413.3008&rep=rep1&type=pdf
lmu::Curvature lmu::curvature(const Eigen::Vector3d & ps, const CSGNode & node, double h)
{
	Eigen::Matrix<double, 3, 3> m;
	Eigen::Matrix<double, 3, 3> hess = hessian(ps, node, h);
	Eigen::Matrix<double, 3, 3> hessAd = adjunct(hess);

	//std::cout << "Hesse: " << hess << std::endl;
	//std::cout << "Ad: " << hessAd << std::endl;

	Eigen::Vector3d g(dx(ps, node, h), dy(ps, node, h), dz(ps, node, h));


	//std::cout << "Gradient: " << g.normalized() << std::endl;
	//std::cout << "Gradient: " << Eigen::Vector3d(dx(ps, node, h), dy(ps, node, h), dz(ps, node, h)).normalized() << std::endl;
	//std::cout << "Trace: " << hess.trace() << std::endl;

	double gaussCurv = (g.transpose() * hessAd * g)(0, 0) / std::pow(g.norm(), 4);

	double meanCurv = ((g.transpose() * hess * g)(0, 0) - std::pow(g.norm(), 2) * hess.trace()) / (2.0 * std::pow(g.norm(), 3));

	double s = std::sqrt(meanCurv * meanCurv - gaussCurv);
	double k1 = meanCurv + s;
	double k2 = meanCurv - s;

	Curvature c;
	c.gaussCurv = gaussCurv;
	c.meanCurv = meanCurv;
	c.k1 = k1;
	c.k2 = k2;

	return c;
}

Eigen::MatrixXd lmu::filterPrimitivePointsByCurvature(const std::vector<ImplicitFunctionPtr>& funcs, double h, double t)
{
	std::vector<Eigen::Matrix<double,1,6>> points; 
		
	for (const auto& func : funcs)
	{
		for (int i = 0; i < func->pointsCRef().rows(); ++i)
		{
			Eigen::Matrix<double, 1, 6> point = func->pointsCRef().row(i); 
						
			Curvature c = curvature(point.leftCols(3), geometry(func), h);

			double deviationFromFlatness = std::sqrt(c.k1 * c.k1 + c.k2 * c.k2);

			if (deviationFromFlatness >= t)
			{
				points.push_back(point);
			}
		}
	}

	Eigen::MatrixXd m(points.size(), 6);
	int i = 0;
	for (const auto& point : points)
		m.row(i++) = points[i];

	return m;
}

Eigen::VectorXd lmu::computeCurvature(const Eigen::MatrixXd& samplePoints, const CSGNode& node, double h, bool normalize)
{
	Eigen::VectorXd res(samplePoints.rows());

	double maxDev = 0;

	for (int i = 0; i < samplePoints.rows(); ++i)
	{
		Curvature c = curvature(samplePoints.row(i), node, h);

		double deviationFromFlatness = std::sqrt(c.k1 * c.k1 + c.k2 * c.k2);

		res.row(i) << deviationFromFlatness;

		std::cout << deviationFromFlatness << " ";

		maxDev = maxDev < deviationFromFlatness ? deviationFromFlatness : maxDev;

	}

	if (normalize && maxDev > 0.0)
	{
		for (int i = 0; i < samplePoints.rows(); ++i)
		{
			res.row(i) << (res.row(i) / maxDev);
		}
	}

	return res;
}