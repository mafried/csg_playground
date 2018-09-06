#ifndef RANSAC_H
#define RANSAC_H

#include <vector>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "mesh.h"

namespace lmu
{

	enum class Primitives
	{
		PLANE = 1,
		SPHERE = 2,
		CYLINDER = 4,
		CONE = 8,
		TORUS = 16
	};

	Primitives operator &(Primitives lhs, Primitives rhs);

	struct RansacCGALParams
	{		
		double probability;         ///< Probability to control search endurance. %Default value: 5%.
		std::size_t min_points; ///< Minimum number of points of a shape. %Default value: 1% of total number of input points.
		double epsilon;             ///< Maximum tolerance Euclidian distance from a point and a shape. %Default value: 1% of bounding box diagonal.
		double normal_threshold;	  ///< Maximum tolerance normal deviation from a point's normal to the normal on shape at projected point. %Default value: 0.9 (around 25 degrees).
		double cluster_epsilon;	    ///< Maximum distance between points to be considered connected. %Default value: 1% of bounding box diagonal.
		Primitives primitives;
	};

	std::vector<std::shared_ptr<ImplicitFunction>> ransacWithCGAL(const Eigen::MatrixXd& points, const Eigen::MatrixXd& normals, std::vector<int>& indices, const RansacCGALParams& params);

	std::vector<std::shared_ptr<ImplicitFunction>> ransacWithPCL(const Eigen::MatrixXd& points, const Eigen::MatrixXd& normals);

	void ransacWithSim(const Eigen::MatrixXd& points, const Eigen::MatrixXd& normals, double maxDelta, const std::vector<std::shared_ptr<ImplicitFunction>>& knownFunctions);
}

#endif