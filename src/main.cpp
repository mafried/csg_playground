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

enum class ApproachType
{
	None = 0,
	BaselineGA, 
	Partition
};

ApproachType approachType = ApproachType::BaselineGA;
ParallelismOptions paraOptions = ParallelismOptions::GAParallelism;
int sampling = 30;//35;
int nodeIdx = 1;

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
	
	//RUN_TEST(CSGNodeTest);


	igl::opengl::glfw::Viewer viewer;
	viewer.mouse_mode = igl::opengl::glfw::Viewer::MouseMode::Rotation;

	// Initialize
	update(viewer);

	Eigen::AngleAxisd rot90x(M_PI / 2.0, Vector3d(0.0, 0.0, 1.0));


	CSGNode node(nullptr);

	if (nodeIdx == 0)
	{
		node =
			op<Union>(
		{
			op<Union>(
			{
				op<Union>(
				{
					op<Difference>({
						geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(-0.2, 0, -1)*rot90x), 0.2, 0.8, "Cylinder_2"),
						geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(-0.2, 0, -1)*rot90x), 0.1, 0.8, "Cylinder_3")
					}),

					op<Union>(
					{
						geo<IFBox>((Eigen::Affine3d)Eigen::Translation3d(0, 0, -0.5), Eigen::Vector3d(0.5,1.0,1.0),2, "Box_2"),
						geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(0, 0, -1)*rot90x), 0.5, 0.5, "Cylinder_0")
					})
				})
				,
				op<Union>(
				{
					op<Union>(
					{
						geo<IFBox>((Eigen::Affine3d)Eigen::Translation3d(-0.3, 0, -0.5), Eigen::Vector3d(0.2,0.8,0.9),2, "Box_3"),
						geo<IFBox>((Eigen::Affine3d)Eigen::Translation3d(0.3, 0, -0.5), Eigen::Vector3d(0.2,0.8,1.0),2, "Box_4")
					}),

					op<Union>(
					{
						geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(0.3, 0, -1)*rot90x), 0.4, 0.2, "Cylinder_1"),
						geo<IFBox>(Eigen::Affine3d::Identity(), Eigen::Vector3d(1.0,2.0,0.1),2, "Box_1")
					})
				})

			}),

			op<Union>(
			{
				geo<IFBox>((Eigen::Affine3d)Eigen::Translation3d(0, 0, -0.2), Eigen::Vector3d(0.8,1.8,0.2),2, "Box_0"),
				op<Union>(
				{
					op<Union>(
					{
						op<Difference>(
						{
							geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, 1.0, 0.2), 0.2, "Sphere_0"),
							geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, 1.0, 0.6), 0.4, "Sphere_1")
						}),
						op<Difference>(
						{
							geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, 1.0, 0.2), 0.2, "Sphere_2"),
							geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, 1.0, 0.6), 0.4, "Sphere_3")
						})
					}),
					op<Union>(
					{
						op<Difference>(
						{
							geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, -1.0, 0.2), 0.2, "Sphere_4"),
							geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, -1.0, 0.6), 0.4, "Sphere_5")
						}),
						op<Difference>(
						{
							geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, -1.0, 0.2), 0.2, "Sphere_6"),
							geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, -1.0, 0.6), 0.4, "Sphere_7")
						})
					})
				})
			})
		});
	}
	else if (nodeIdx == 1)
	{
		node =

			op<Difference>(
		{
		op<Union>(
		{
			op<Union>(
			{
				geo<IFBox>(Eigen::Affine3d::Identity(), Eigen::Vector3d(0.6,0.6,0.6),2, "Box_0", 2.0),
				geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, -0.3, 0), 0.3, "Sphere_0")
			}),
			geo<IFCylinder>(Eigen::Affine3d::Identity(), 0.2, 1.0, "Cylinder_0"),
		}),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, 0.7, 0), 0.4, "Sphere_1")
		});
	}
	else if (nodeIdx == 2)
	{
		node =
			op<Union>(
		{
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, 0.0, 0), 0.25, "Sphere_1"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, 0.3, 0), 0.25, "Sphere_2"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, 0.6, 0), 0.25, "Sphere_3"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, 0.9, 0), 0.25, "Sphere_4"),

			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.3, 0.0, 0), 0.25, "Sphere_7"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.3, 0.3, 0), 0.25, "Sphere_8"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.3, 0.6, 0), 0.25, "Sphere_9"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.3, 0.9, 0), 0.25, "Sphere_10"),

			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.6, 0.0, 0), 0.25, "Sphere_13"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.6, 0.3, 0), 0.25, "Sphere_14"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.6, 0.6, 0), 0.25, "Sphere_15"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.6, 0.9, 0), 0.25, "Sphere_16"),

			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.9, 0.0, 0), 0.25, "Sphere_19"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.9, 0.3, 0), 0.25, "Sphere_20"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.9, 0.6, 0), 0.25, "Sphere_21"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.9, 0.9, 0), 0.25, "Sphere_22"),

			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(1.2, 0.0, 0), 0.25, "Sphere_25"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(1.2, 0.3, 0), 0.25, "Sphere_26"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(1.2, 0.6, 0), 0.25, "Sphere_27"),
			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(1.2, 0.9, 0), 0.25, "Sphere_28"),
		});
	}
	else if (nodeIdx == 3)
	{
		node = geo<IFCone>((Eigen::Affine3d)(Eigen::Translation3d(0.3, 0, -1)*rot90x), Eigen::Vector3d(0.6, 0.6, 0.6), "cone");
	}
	else
	{
		std::cerr << "Could not get node. Idx: " << nodeIdx << std::endl;
		return -1;
	}


	//auto n1 = geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0,0,0),1.0, "Sphere_0");

	//std::cout << n1.signedDistance(Eigen::Vector3d(3, 0, 0)) << " " << n1.signedDistanceAndGradient(Eigen::Vector3d(3, 0, 0)) << std::endl;

	//auto curvature = lmu::curvature(Eigen::Vector3d(3,2,1), n1, 0.001);

	//std::cout << "gauss: " << curvature.gaussCurv << " mean: " << curvature.meanCurv << std::endl;
	//int i; 
	//std::cin >> i;

	//lmu::Mesh csgMesh = computeMesh(node, Eigen::Vector3i(50, 50, 50));
	//viewer.data().set_mesh(csgMesh.vertices, csgMesh.indices);

	//auto error = computeDistanceError(csgMesh.vertices, node, node2, true);
	//viewer.data().set_colors(error);

	//high: lmu::computePointCloud(node, Eigen::Vector3i(120, 120, 120), 0.05, 0.01);
	//medium: lmu::computePointCloud(node, Eigen::Vector3i(75, 75, 75), 0.05, 0.01);


	auto pointCloud = lmu::computePointCloud(node, Eigen::Vector3i(100, 100, 100), 0.05, 0.03);

	std::vector<ImplicitFunctionPtr> shapes; 
	for (const auto& geoNode : allGeometryNodePtrs(node))
		shapes.push_back(geoNode->function());

	auto dims = lmu::computeDimensions(node);

	auto graph = lmu::createConnectionGraph(shapes, std::get<0>(dims), std::get<1>(dims), 0.01);
	auto graph2 = lmu::createConnectionGraph(shapes);

	lmu::writeConnectionGraph("connectionGraph.dot", graph);
	lmu::writeConnectionGraph("connectionGraph2.dot", graph2);

	lmu::ransacWithSim(pointCloud.leftCols(3), pointCloud.rightCols(3), 0.05, shapes);

	std::cout << "Before: " << pointCloud.rows() << std::endl;


	pointCloud = lmu::filterPrimitivePointsByCurvature(shapes, 0.01, 0.1);
	
	std::cout << "After: " << pointCloud.rows() << std::endl;

	//pointCloud = getSIFTKeypoints(pointCloud, 0.01, 0.001, 3, 4, false);

	//auto colors = lmu::computeCurvature(pointCloud.leftCols(3), node, 0.01, true);

	viewer.data().point_size = 1.0;
	viewer.data().set_points(pointCloud.leftCols(3), Eigen::MatrixXd()/*, colors*/);
	viewer.core.background_color = Eigen::Vector4f(1, 1, 1, 1);

	viewer.launch();
	
}