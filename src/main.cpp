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

#include "dnf.h"

enum class ApproachType
{
	None = 0,
	BaselineGA, 
	Partition
};

ApproachType approachType = ApproachType::BaselineGA;
ParallelismOptions paraOptions = ParallelismOptions::GAParallelism;
int sampling = 30;//35;
int nodeIdx = 0;

void update(igl::opengl::glfw::Viewer& viewer)
{
}

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int mods)
{
	switch (key)
	{
	default:
		return false;
	case '-':
		viewer.core.camera_dnear -= 0.1;
		return true;
	case '+':
		viewer.core.camera_dnear += 0.1;
		return true;
	}
	update(viewer);
	return true;
}

int main(int argc, char *argv[])
{
	using namespace Eigen;
	using namespace std;
	
	std::string file; 
	double scaleFactor; 
	bool readHeader;

	if (argc != 4)
	{
		std::cerr << "Viewer needs 3 arguments (file, scale factor, read header yes/no)." << std::endl;
		return -1;
	}
	else
	{
		file = std::string(argv[1]); 
		scaleFactor = std::stod(std::string(argv[2]));
		readHeader = std::stol(std::string(argv[3])) != 0;
	}
	
	igl::opengl::glfw::Viewer viewer;
	viewer.mouse_mode = igl::opengl::glfw::Viewer::MouseMode::Rotation;

	// Initialize
	update(viewer);

	
	auto pointCloud = lmu::readPointCloud(file, scaleFactor, readHeader);

	viewer.data().point_size = 4.0;
	
	viewer.data().set_points(pointCloud.leftCols(3), pointCloud.rightCols(3));

	viewer.core.background_color = Eigen::Vector4f(1, 1, 1, 1);

	viewer.launch();
	
}