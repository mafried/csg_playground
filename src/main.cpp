#define BOOST_PARAMETER_MAX_ARITY 12

#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>

#include <Eigen/Core>
#include <iostream>
#include <tuple>
#include <chrono>

#include "mesh.h"
#include "ransac.h"
#include "pointcloud.h"
#include "collision.h"
#include "congraph.h"
#include "csgtree.h"
#include "tests.h"

#include "csgnode_evo.h"
#include "csgnode_helper.h"

#include "evolution.h"

#include "curvature.h"

#include <boost/algorithm/string.hpp>

void ransac(const Eigen::MatrixXd& inputPointCloud, const RansacCGALParams& params, bool interactive);

int main(int argc, char *argv[])
{
	using namespace Eigen;
	using namespace std;
	
	std::string file = std::string(argv[1]);
	std::string op = std::string(argv[2]); 
	double scale = std::stod(std::string(argv[3]));
	bool interactive = std::stoi(std::string(argv[4])) == 1;

	auto inputPointCloud = lmu::readPointCloud(file, scale);

	if (boost::iequals(op, "ransac"))
	{		
		RansacCGALParams params;
		params.cluster_epsilon = std::stod(std::string(argv[5]));
		params.epsilon = std::stod(std::string(argv[6]));
		params.min_points = std::stoul(std::string(argv[7]));
		params.normal_threshold = std::stod(std::string(argv[8]));
		params.probability = std::stod(std::string(argv[9]));
		params.primitives = static_cast<Primitives>(std::stoul(std::string(argv[10])));
		
		ransac(inputPointCloud, params, interactive);
	}
	else 
	{
		std::cerr << "Operation " << op << " not available." << std::endl;
	}
}

void ransac(const Eigen::MatrixXd& inputPointCloud, const RansacCGALParams& params, bool interactive)
{
	std::vector<int> types; 

	auto shapes = lmu::ransacWithCGAL(inputPointCloud.leftCols(3), inputPointCloud.rightCols(3), types, params);

	for (const auto& shape : shapes)
	{
		lmu::writePointCloud("pc_" + shape->name() + ".dat", shape->points());
	}

	std::ofstream s("res.dat");


	for (int type : types)
	{
		s << type << std::endl;
	}

	s.close();

	if (!interactive)
		return;

	igl::opengl::glfw::Viewer viewer;
	viewer.mouse_mode = igl::opengl::glfw::Viewer::MouseMode::Rotation;
	viewer.data().point_size = 5.0;
	
	int rows = 0;
	for (const auto& shape : shapes)
	{
		rows += shape->points().rows();
	}

	Eigen::MatrixXd points(rows, 6);
	int j = 0;
	int k = 0;

	Eigen::MatrixXd colors(16, 3);
	colors.row(0) = Eigen::Vector3d(1, 0, 0);
	colors.row(1) = Eigen::Vector3d(0, 1, 0);
	colors.row(2) = Eigen::Vector3d(0, 0, 1);
	colors.row(3) = Eigen::Vector3d(1, 0, 1);
	colors.row(4) = Eigen::Vector3d(1, 1, 0);
	colors.row(5) = Eigen::Vector3d(0, 1, 1);
	colors.row(6) = Eigen::Vector3d(1, 1, 1);
	colors.row(7) = Eigen::Vector3d(0, 0, 0);

	colors.row(8) = Eigen::Vector3d(.5, 0, 0);
	colors.row(9) = Eigen::Vector3d(0, .5, 0);
	colors.row(10) = Eigen::Vector3d(0, 0, .5);
	colors.row(11) = Eigen::Vector3d(.5, 0, .5);
	colors.row(12) = Eigen::Vector3d(.5, .5, 0);
	colors.row(13) = Eigen::Vector3d(0, .5, .5);
	colors.row(14) = Eigen::Vector3d(.5, .5, .5);
	colors.row(15) = Eigen::Vector3d(0, 0, 0);

	for (const auto& shape : shapes)
	{
		for (int i = 0; i < shape->points().rows() - 1; ++i)
		{
			auto row = shape->points().row(i);
			points.row(j) = row;
			points.row(j)[3] = colors.row(k % colors.size())[0];
			points.row(j)[4] = colors.row(k % colors.size())[1];
			points.row(j)[5] = colors.row(k % colors.size())[2];

			j++;
		}

		std::cout << shape->name() << " Color RGB: " << colors.row(k % colors.size()) << std::endl;

		k++;
	}

	viewer.data().add_points(points.leftCols(3), points.rightCols(3));
	//viewer.data().add_points(inputPointCloud.leftCols(3), Eigen::Vector3d(0.1,0.1,0.1));

	viewer.core.background_color = Eigen::Vector4f(1, 1, 1, 1);

	viewer.launch();
}