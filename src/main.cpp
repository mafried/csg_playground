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
#include "tests.h"

#include "csgnode_evo.h"
#include "csgnode_evo_v2.h"
#include "csgnode_helper.h"

#include "evolution.h"

#include "curvature.h"

#include "dnf.h"

#include "statistics.h"

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
						geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(-0.3, 0, -1)*rot90x), 0.1, 1, "Cylinder_3")
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
						geo<IFBox>((Eigen::Affine3d)(Eigen::Translation3d(-0.3, 0, -0.5)), Eigen::Vector3d(0.2,0.8,0.9),2, "Box_3"),
						geo<IFBox>((Eigen::Affine3d)Eigen::Translation3d(0.3, 0, -0.5), Eigen::Vector3d(0.2,0.8,1.0),2, "Box_4")
					}),

					op<Union>(
					{
						geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(0.3, 0, -1)*rot90x), 0.4, 0.2, "Cylinder_1"),
						geo<IFBox>(Eigen::Affine3d::Identity(), Eigen::Vector3d(1.0,2.0,0.1),2, "Box_1") //Box close to spheres
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

	/*node =
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
	})
	})
	})
	});*/

	/*node =
		op<Difference>({
			op<Union>(
			{
				geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, 0.0, 0), 1, "Sphere_1"),
				geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, 1.2, 0), 1, "Sphere_2"),
				geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, 2.4, 0), 1, "Sphere_3")
			}),

			geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, 3.0, 0), 1, "Sphere_4")

			//op<Difference>(
			//{
			//}
			//)
		}
	);*/

	/*node = op<Union>(
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
	});*/

	/*node = op<Union>({
		geo<IFBox>((Eigen::Affine3d)(Eigen::Translation3d(-0.3, 0, -0.5) *rot90x), Eigen::Vector3d(0.2, 0.8, 0.9), 2, "Box_3"),
		geo<IFBox>((Eigen::Affine3d)(Eigen::Translation3d(-0.3, 0, -0.5)), Eigen::Vector3d(0.2, 0.8, 0.9), 2, "Box_2"),
	});*/


	/*node = op<Difference>({
		geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(-0.2, 0, -1)), 0.2, 0.8, "Cylinder_2"),
		geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(-0.2, 0, -1)), 0.1, 2, "Cylinder_3")
	});
	*/

	/*node = op<Difference>(
	{
		geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, -1.0, 0.2), 0.2, "Sphere_4"),
		geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, -1.0, 0.6), 0.4, "Sphere_5")
	});*/


	/*node = op<Union>(
	{
		geo<IFBox>(Eigen::Affine3d::Identity(), Eigen::Vector3d(1.0,2.0,0.1),2, "Box_1"),



		op<Union>(
		{
			op<Difference>(
			{
				geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, -1.0, 0.2), 0.2, "Sphere_6"),
				geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, -1.0, 0.6), 0.4, "Sphere_7")
			})

		})
	});*/

/*node = op<Union>(
{
	geo<IFBox>(Eigen::Affine3d::Identity(), Eigen::Vector3d(1.0,2.0,0.2),2, "Box_1"), //Box close to spheres
	
	
	op<Difference>(
	{
		geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, 1.0, 0.2), 0.2, "Sphere_0"),
		geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, 1.0, 0.6), 0.4, "Sphere_1")
	}),
	
	op<Difference>(
	{
		geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, 1.0, 0.2), 0.2, "Sphere_2"),
		geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, 1.0, 0.6), 0.4, "Sphere_3")
	}),

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
	
});*/


//node = geo<IFBox>(Eigen::Affine3d::Identity(), Eigen::Vector3d(1.0, 2.0, 0.2), 2, "Box_1");

//node = op<Difference>(
//{
//	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, -1.0, 0.2), 0.2, "Sphere_6"),
//	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, -1.0, 0.6), 0.4, "Sphere_7")
//});


node = op<Union>(
{
	op<Difference>({
		geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(-0.2, 0, -1)*rot90x), 0.2, 0.8, "Cylinder_2"),
		geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(-0.3, 0, -1)*rot90x), 0.1, 1, "Cylinder_3")
	}),

	op<Union>(
	{
		geo<IFBox>((Eigen::Affine3d)Eigen::Translation3d(0, 0, -0.5), Eigen::Vector3d(0.5,1.0,1.0),2, "Box_2"),
		geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(0, 0, -1)*rot90x), 0.5, 0.5, "Cylinder_0")
	}),

	geo<IFBox>(Eigen::Affine3d::Identity(), Eigen::Vector3d(1.0,2.0,0.2),2, "Box_1"), //Box close to spheres


	op<Difference>(
	{
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, 1.0, 0.2), 0.2, "Sphere_0"),
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, 1.0, 0.6), 0.4, "Sphere_1")
	}),

	op<Difference>(
	{
		geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, 1.0, 0.2), 0.2, "Sphere_2"),
		geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, 1.0, 0.6), 0.4, "Sphere_3")
	}),

	op<Difference>(
	{
		geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, -1.0, 0.2), 0.2, "Sphere_4"),
		geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, -1.0, 0.6), 0.4, "Sphere_5")
	}),
	op<Difference>(
	{
		geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, -1.0, 0.2), 0.2, "Sphere_6"),
		geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, -1.0, 0.6), 0.4, "Sphere_7")
	}),

	geo<IFBox>((Eigen::Affine3d)(Eigen::Translation3d(-0.3, 0, -0.5)), Eigen::Vector3d(0.2,0.8,0.9),2, "Box_3"),

	geo<IFBox>((Eigen::Affine3d)Eigen::Translation3d(0.3, 0, -0.5), Eigen::Vector3d(0.2,0.8,1.0),2, "Box_4"),

	geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(0.3, 0, -1)*rot90x), 0.4, 0.2, "Cylinder_1"),
	
});

auto nodeUnion = op<Union>(
{
	op<Union>({
	geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(-0.2, 0, -1)*rot90x), 0.2, 0.8, "Cylinder_2"),
	geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(-0.3, 0, -1)*rot90x), 0.1, 1, "Cylinder_3")
}),

op<Union>(
{
	geo<IFBox>((Eigen::Affine3d)Eigen::Translation3d(0, 0, -0.5), Eigen::Vector3d(0.5,1.0,1.0),2, "Box_2"),
	geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(0, 0, -1)*rot90x), 0.5, 0.5, "Cylinder_0")
}),

geo<IFBox>(Eigen::Affine3d::Identity(), Eigen::Vector3d(1.0,2.0,0.2),2, "Box_1"), //Box close to spheres


op<Union>(
{
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, 1.0, 0.2), 0.2, "Sphere_0"),
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, 1.0, 0.6), 0.4, "Sphere_1")
}),

op<Union>(
{
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, 1.0, 0.2), 0.2, "Sphere_2"),
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, 1.0, 0.6), 0.4, "Sphere_3")
}),

op<Union>(
{
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, -1.0, 0.2), 0.2, "Sphere_4"),
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, -1.0, 0.6), 0.4, "Sphere_5")
}),
op<Union>(
{
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, -1.0, 0.2), 0.2, "Sphere_6"),
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, -1.0, 0.6), 0.4, "Sphere_7")
}),

geo<IFBox>((Eigen::Affine3d)(Eigen::Translation3d(-0.3, 0, -0.5)), Eigen::Vector3d(0.2,0.8,0.9),2, "Box_3"),

geo<IFBox>((Eigen::Affine3d)Eigen::Translation3d(0.3, 0, -0.5), Eigen::Vector3d(0.2,0.8,1.0),2, "Box_4"),

geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(0.3, 0, -1)*rot90x), 0.4, 0.2, "Cylinder_1"),

});

	double samplingStepSize = 0.03; 
	double maxDistance = 0.01;
	double noiseSigma = 0.03;

    auto pointCloud = lmu::computePointCloud(node, samplingStepSize, maxDistance, noiseSigma);

	
	std::vector<ImplicitFunctionPtr> shapes; 
	for (const auto& geoNode : allGeometryNodePtrs(node))
	{
		shapes.push_back(geoNode->function());
	}
	lmu::ransacWithSimMultiplePointOwners(pointCloud.leftCols(3), pointCloud.rightCols(3), maxDistance * 5 , shapes);

	int totalNumPoints = 0;
	for (const auto& geoNode : allGeometryNodePtrs(node))
	{
		int curNumPts = geoNode->function()->pointsCRef().rows();
		totalNumPoints += curNumPts;

		std::cout << "Shape: " << geoNode->function()->name() << " Points: " << curNumPts << std::endl;
	}
	std::cout << "NUM POINTS: " << totalNumPoints << std::endl;
	std::cout << "Point-cloud size: " << pointCloud.rows() << std::endl;

	auto dims = lmu::computeDimensions(node);

	auto graph = lmu::createConnectionGraph(shapes, std::get<0>(dims), std::get<1>(dims), samplingStepSize);
		
	//auto graph = lmu::createConnectionGraph(shapes);

	lmu::writeConnectionGraph("connectionGraph.dot", graph);
	//lmu::writeConnectionGraph("connectionGraph2.dot", graph2);

	
	//pointCloud = lmu::filterPrimitivePointsByCurvature(shapes, 0.01, lmu::computeOutlierTestValues(shapes), FilterBehavior::FILTER_FLAT_SURFACES, false);

	//shapes.clear();
	//lmu::ransacWithSim(pointCloud.leftCols(3), pointCloud.rightCols(3), 0.05, shapes);



	lmu::movePointsToSurface(shapes, true, 0.00001);

	//auto dnf = lmu::computeShapiro(shapes, true, lmu::Graph(), { 0.001 });

	//auto res = lmu::computeCSGNode(shapes, graph, { 0.001 });//lmu::DNFtoCSGNode(dnf);

	//CSGNode res = op<Union>();

	//SampleParams p{ 0.001 };

	//auto partitions = lmu::partitionByPrimeImplicants(graph, p, true);

	//res = lmu::computeShapiroWithPartitions(partitions, p);
		
	auto partitions = lmu::getUnionPartitionsByPrimeImplicantsWithPruning(graph, { 0.001 });
	int i = 0;
	for (const auto& p : partitions)
	{
		std::cout << "partition" << std::endl;
		lmu::writeConnectionGraph("p_" + std::to_string(i++), p);
	}

	/*double lambda = lmu::lambdaBasedOnPoints(shapes);
	std::cout << "lambda: " << lambda << std::endl;
	lmu::CSGNodeRanker r(lambda, shapes, graph);
	std::cout << "QUALITY NODE: " << r.rank(node) << std::endl;
	std::cout << "QUALITY NODE UNION: " << r.rank(nodeUnion) << std::endl;

	lmu::CSGNodeRankerV2 r2(graph, 0.0, 0.01);
	std::cout << "V2" << std::endl;

	std::cout << "QUALITY NODE: " << r2.rank(node) << std::endl;
	std::cout << "QUALITY NODE UNION: " << r2.rank(nodeUnion) << std::endl;

	return 0;*/

	//auto res = lmu::computeShapiroWithPartitions(partitions, { 0.001 }); //lmu::createCSGNodeWithGAV2(graph);
	//auto res = computeGAWithPartitionsV2(/*partitions*/{ graph }, false, "stats.dat");
	//auto res = lmu::computeShapiroWithPartitions(partitions, { 0.001 });
	//auto res = computeGAWithPartitions(/*partitions*/{graph}, false, "stats.dat");

	//auto res = DNFtoCSGNode(lmu::computeShapiro(lmu::getImplicitFunctions(graph), true, graph, { 0.001 }));
	//std::cout << "NODESIZE: " << numNodes(res) << std::endl;

	/*auto cliques = lmu::getCliques(graph);
	auto cliquesAndNodes = computeNodesForCliques(cliques, paraOptions);

	optimizeCSGNodeClique(cliquesAndNodes, 100.0);

	auto res = mergeCSGNodeCliqueSimple(cliquesAndNodes);
	*/

	/*lmu::writeNode(res, "tree.dot");

	auto mesh = lmu::computeMesh(res, Eigen::Vector3i(100, 100, 100));

	pointCloud = lmu::computePointCloud(res, samplingStepSize, maxDistance, 0);
	viewer.data().set_points(pointCloud.leftCols(3), pointCloud.rightCols(3));

	igl::writeOBJ("mesh.obj", mesh.vertices, mesh.indices);
	
	//std::cout << lmu::espressoExpression(dnf) << std::endl;

	
	//for (const auto& func : shapes)
	//{
	//	viewer.data().add_points(func->pointsCRef().leftCols(3), func->pointsCRef().rightCols(3));
	//}

	
	//std::cout << "Before: " << pointCloud.rows() << std::endl;
	
	//pointCloud = lmu::filterPrimitivePointsByCurvature(shapes, 0.01, 7 , FilterBehavior::FILTER_CURVY_SURFACES), true;
	
	//std::cout << "After: " << pointCloud.rows() << std::endl;

	//pointCloud = getSIFTKeypoints(pointCloud, 0.01, 0.001, 3, 4, false);

	//auto colors = lmu::computeCurvature(pointCloud.leftCols(3), node, 0.01, true);

	viewer.data().point_size = 4.0;
	
	Eigen::MatrixX3d colors = pointCloud.rightCols(3);
	//colors.transpose() =  ((colors.transpose() + Eigen::Vector3d(1.0,1.0,1.0)).cwiseProduct(Eigen::Vector3d(0.5, 0.5, 0.5)));

	for (int r = 0; r < colors.rows(); r++) {
		Vector3d v = colors.row(r);
		v = (v + Eigen::Vector3d(1.0, 1.0, 1.0)).cwiseProduct(Eigen::Vector3d(0.5, 0.5, 0.5));
		colors.row(r) = v;
	}

	//std::cout << "Considered Clause " << g_clause << std::endl;
	//viewer.data().set_points(g_testPoints.leftCols(3), g_testPoints.rightCols(3));
	
	//viewer.data().set_mesh(node.function

	//for (const auto& shape : allGeometryNodePtrs(node))
	//{
		//if (shape->name() == "Sphere_1")
		//	viewer.data().add_points(shape->function()->pointsCRef().leftCols(3), shape->function()->pointsCRef().rightCols(3));
	//}
	
	//std::cout << "TEST: " << g_testPoints;

	std::ofstream fs("output.dat");
	fs << g_testPoints;
	fs.close();
	

	*/

	/*double lower_bound = 0;
	double upper_bound = 1;
	std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
	std::default_random_engine re;
	double a_random_double = unif(re);

	lmu::DataFrame<Eigen::Vector3d> data(100000);
	for (int i = 0; i < data.size(); ++i)
	{
		
		if (i > data.size() / 2)
			data[i] = Eigen::Vector3d(2.0 + unif(re), 2.0 + unif(re), 2.0 + unif(re));
		else 
			data[i] = Eigen::Vector3d(unif(re), unif(re), unif(re));

	}

	auto d = lmu::k_means(data, 2, 30);

	for (const auto& i : d.means)
		std::cout << i << std::endl;


	Eigen::MatrixX3d points; 
	points.resize(data.size(), Eigen::NoChange_t::NoChange);
	Eigen::MatrixX3d colors;
	colors.resize(data.size(), Eigen::NoChange_t::NoChange);

	for (int i = 0; i < data.size(); ++i)
	{
		points.row(i) = data[i];

		if (d.assignments[i] == 1)
			colors.row(i) << 1.0, 0.0, 0.0;
		else
			colors.row(i) << 0.0, 1.0, 0.0;		
	}


	viewer.data().set_points(points, colors);
	*/

	viewer.core.background_color = Eigen::Vector4f(1, 1, 1, 1);

	viewer.launch();
	
}