
#include <Eigen/Core>

#include "curvature.h"
#include "csgnode_helper.h"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>

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

Eigen::MatrixXd lmu::filterPrimitivePointsByCurvature(const std::vector<ImplicitFunctionPtr>& funcs, double h, const std::unordered_map<lmu::ImplicitFunctionPtr, double>& outlierTestValues, FilterBehavior behavior, bool normalized)
{
	std::vector<Eigen::Matrix<double,1,6>> points; 
	
	double min = std::numeric_limits<double>::max();
	double max = -std::numeric_limits<double>::max();

	if (normalized)
	{
		for (const auto& func : funcs)
		{
			for (int i = 0; i < func->pointsCRef().rows(); ++i)
			{
				Eigen::Matrix<double, 1, 6> point = func->pointsCRef().row(i);

				Curvature c = curvature(point.leftCols(3), geometry(func), h);

				double deviationFromFlatness = std::sqrt(c.k1 * c.k1 + c.k2 * c.k2);

				min = deviationFromFlatness < min ? deviationFromFlatness : min; 
				max = deviationFromFlatness > max ? deviationFromFlatness : max;
			}
		}
	}

	for (const auto& func : funcs)
	{
		double t = outlierTestValues.at(func);
		std::cout << func->name() << ": " << t << std::endl;

		for (int i = 0; i < func->pointsCRef().rows(); ++i)
		{
			Eigen::Matrix<double, 1, 6> point = func->pointsCRef().row(i); 
						
			Curvature c = curvature(point.leftCols(3), geometry(func), h);

			double deviationFromFlatness = std::sqrt(c.k1 * c.k1 + c.k2 * c.k2);

			deviationFromFlatness = normalized ? (deviationFromFlatness - min) / (max - min) : deviationFromFlatness; 

			//std::cout << min << " " << max << " " <<  deviationFromFlatness << std::endl;

			switch (behavior)
			{
			case FilterBehavior::FILTER_FLAT_SURFACES:
				if (deviationFromFlatness > t)
					points.push_back(point);				
				break;
			case FilterBehavior::FILTER_CURVY_SURFACES:
				if (deviationFromFlatness < t)
					points.push_back(point);
				break;
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

Eigen::Matrix<double, -1, 5> lmu::estimateCurvature(const PointCloud & pc, double searchRadius)
{
	pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> pcEstimator;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

	pcl::PointCloud<pcl::PointXYZ>::Ptr points(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

	for (int i = 0; i < pc.rows(); i++)
	{
		auto row = pc.row(i);

		points->push_back(pcl::PointXYZ(row.col(0).value(), row.col(1).value(), row.col(2).value()));
		normals->push_back(pcl::Normal(row.col(3).value(), row.col(4).value(), row.col(5).value()));
	}

	/*pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
	normal_estimation.setInputCloud(points);
	normal_estimation.setSearchMethod(tree);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::Normal>);
	normal_estimation.setRadiusSearch(0.03);
	normal_estimation.compute(*cloud_with_normals);*/

	pcEstimator.setInputCloud(points);
	pcEstimator.setInputNormals(normals);
	pcEstimator.setSearchMethod(tree);
	pcEstimator.setRadiusSearch(searchRadius);

	pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principalCurvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>());
	pcEstimator.compute(*principalCurvatures);

	Eigen::Matrix<double, -1, 5> res(principalCurvatures->size(), 5);

	for (int i = 0; i < res.rows(); i++)
	{
		auto prcu = principalCurvatures->points[i];
		res.row(i) << prcu.pc1, prcu.pc2, prcu.principal_curvature_x, prcu.principal_curvature_y, prcu.principal_curvature_z;
		//std::cout << prcu.pc1 << " " << prcu.pc2 << std::endl;
	}

	return res;
}
