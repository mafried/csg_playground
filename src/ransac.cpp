#include "..\include\ransac.h"

#include <unordered_map>

void lmu::ransacWithSim(const Eigen::MatrixXd & points, const Eigen::MatrixXd & normals, double maxDelta, const std::vector<std::shared_ptr<ImplicitFunction>>& knownFunctions)
{
	for (auto const& func : knownFunctions)
	{
		std::vector<Eigen::Matrix<double, 1, 6>> pointsAndNormals;

		for (int i = 0; i < points.rows(); ++i)
		{	
			double sd = std::abs(func->signedDistanceAndGradient(points.row(i))[0]);

			//std::cout << sd;

			if (sd <= maxDelta)
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

void lmu::ransacWithSim(const Eigen::MatrixXd & points, const Eigen::MatrixXd & normals, const std::vector<std::shared_ptr<ImplicitFunction>>& knownFunctions)
{
	std::unordered_map<std::shared_ptr<ImplicitFunction>, std::vector<Eigen::Matrix<double, 1, 6>>> pointsAndNormalMap;
	
	for (int i = 0; i < points.rows(); ++i)
	{
		std::shared_ptr<ImplicitFunction> closestFunc = nullptr;
		double closestDistance = std::numeric_limits<double>::max();

		for (auto const& func : knownFunctions)
		{
			double sd = std::abs(func->signedDistanceAndGradient(points.row(i))[0]);

			if (sd < closestDistance)
			{
				closestFunc = func;
				closestDistance = sd;
			}
		}

		Eigen::Matrix<double, 1, 6> row;
		row << points.row(i), normals.row(i);
		pointsAndNormalMap[closestFunc].push_back(row);
	}

	for (auto const& func : knownFunctions)
	{
		Eigen::MatrixXd points(pointsAndNormalMap[func].size(), 6);
		
		int i = 0;
		for (const auto& row : pointsAndNormalMap[func])
			points.row(i++) = row;

		func->setPoints(points);
	}
}

