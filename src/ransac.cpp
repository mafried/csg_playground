#include "..\include\ransac.h"

void lmu::ransacWithSim(const Eigen::MatrixXd & points, const Eigen::MatrixXd & normals, double maxDelta, const std::vector<std::shared_ptr<ImplicitFunction>>& knownFunctions)
{
	for (auto const& func : knownFunctions)
	{
		std::vector<Eigen::Matrix<double, 1, 6>> pointsAndNormals;

		for (int i = 0; i < points.rows(); ++i)
		{	
			if (std::abs(func->signedDistanceAndGradient(points.row(i))[0]) <= maxDelta)
			{
				Eigen::Matrix<double, 1, 6> row;
				row << points.row(i), normals.row(i);
				pointsAndNormals.push_back(row);
			}
		}
		
		Eigen::MatrixXd points(pointsAndNormals.size(), 6);
		int i = 0;
		for (const auto& row : pointsAndNormals)
			points.row(i++) = row;
		
	
		func->setPoints(points);

	}
}
