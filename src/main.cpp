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


double x_off = 2.0;

// previous default model
node = op<Union>(
{

	geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(-0.2 + x_off, 0, -1)*rot90x), 0.2, 2.8, "Cylinder_2"),


op<Union>(
{
	geo<IFBox>((Eigen::Affine3d)Eigen::Translation3d(x_off, 0, -0.5), Eigen::Vector3d(0.5,1.0,1.0),2, "Box_2"),
	geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(x_off, 0, -1)*rot90x), 0.5, 0.5, "Cylinder_0")
}),

geo<IFBox>((Eigen::Affine3d)Eigen::Translation3d(x_off, 0, 0), Eigen::Vector3d(1.0,2.0,0.2),2, "Box_1"), //Box close to spheres


op<Difference>(
{
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5 + x_off, 1.0, 0.2), 0.2, "Sphere_0"),
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5 + x_off, 1.0, 0.6), 0.4, "Sphere_1")
}),

op<Difference>(
{
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5 + x_off, 1.0, 0.2), 0.2, "Sphere_2"),
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5 + x_off, 1.0, 0.6), 0.4, "Sphere_3")
}),

op<Difference>(
{
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5 + x_off, -1.0, 0.2), 0.2, "Sphere_4"),
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5 + x_off, -1.0, 0.6), 0.4, "Sphere_5")
}),
op<Difference>(
{
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5 + x_off, -1.0, 0.2), 0.2, "Sphere_6"),
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5 + x_off, -1.0, 0.6), 0.4, "Sphere_7")
}),

geo<IFBox>((Eigen::Affine3d)(Eigen::Translation3d(-0.3 + x_off, 0, -0.5)), Eigen::Vector3d(0.2,0.8,0.9),2, "Box_3"),

geo<IFBox>((Eigen::Affine3d)Eigen::Translation3d(0.3 + x_off, 0, -0.5), Eigen::Vector3d(0.2,0.8,1.0),2, "Box_4"),

geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(0.3 + x_off, 0, -1)*rot90x), 0.4, 0.2, "Cylinder_1"),


op<Difference>({
	geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(-0.2, 0, -1)*rot90x), 0.2, 0.8, "Cylinder_20"),
	geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(-0.3, 0, -1)*rot90x), 0.1, 1, "Cylinder_30")
}),

op<Union>(
{
	geo<IFBox>((Eigen::Affine3d)Eigen::Translation3d(0, 0, -0.5), Eigen::Vector3d(0.5,1.0,1.0),2, "Box_20"),
	geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(0, 0, -1)*rot90x), 0.5, 0.5, "Cylinder_00")
}),

geo<IFBox>(Eigen::Affine3d::Identity(), Eigen::Vector3d(1.0,2.0,0.2),2, "Box_10"), //Box close to spheres


op<Difference>(
{
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, 1.0, 0.2), 0.2, "Sphere_00"),
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, 1.0, 0.6), 0.4, "Sphere_10")
}),

op<Difference>(
{
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, 1.0, 0.2), 0.2, "Sphere_20"),
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, 1.0, 0.6), 0.4, "Sphere_30")
}),

op<Difference>(
{
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, -1.0, 0.2), 0.2, "Sphere_40"),
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, -1.0, 0.6), 0.4, "Sphere_50")
}),
op<Difference>(
{
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, -1.0, 0.2), 0.2, "Sphere_60"),
	geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.5, -1.0, 0.6), 0.4, "Sphere_70")
}),

geo<IFBox>((Eigen::Affine3d)(Eigen::Translation3d(-0.3, 0, -0.5)), Eigen::Vector3d(0.2,0.8,0.9),2, "Box_30"),

geo<IFBox>((Eigen::Affine3d)Eigen::Translation3d(0.3, 0, -0.5), Eigen::Vector3d(0.2,0.8,1.0),2, "Box_40"),

geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(0.3, 0, -1)*rot90x), 0.4, 0.2, "Cylinder_10"),

});

	
try
{
	node = fromJSONFile("C:/Projekte/csg_playground_build/Release/tree.json");
}
catch (const std::exception& ex)
{
	std::cout << "ERROR: " << ex.what() << std::endl;
}

double samplingStepSize = 0.2; 
double maxDistance = 0.1;
double maxAngleDistance = M_PI / 18.0;
double noiseSigma = 0.0;
CSGNodeSamplingParams samplingParams(maxDistance, maxAngleDistance, noiseSigma, samplingStepSize);
double connectionGraphSamplingStepSize = 0.2;
double gradientStepSize = 0.001;

auto pointCloud = lmu::computePointCloud(node, samplingParams);
auto funcs = lmu::allDistinctFunctions(node);
//auto pointCloud = lmu::readPointCloudXYZ("C:/Projekte/csg_playground_build/Release/model.xyz" , 1.0);
//auto funcs = lmu::fromFilePRIM("C:/Projekte/csg_playground_build/Release/model.prim");


std::cout << "Points: " << pointCloud.rows() << std::endl;

//auto funcs = allDistinctFunctions(node);
auto dims = lmu::computeDimensions(funcs);
auto graph = lmu::createConnectionGraph(funcs, std::get<0>(dims), std::get<1>(dims), connectionGraphSamplingStepSize);

writeConnectionGraph("cg.dot", graph);

//double res = lmu::ransacWithSim(pointCloud, samplingParams, node);
//std::cout << "used points: " << res << "%" << std::endl;

auto pointClouds = lmu::readPointCloudXYZPerFunc("C:/Projekte/csg_playground_build/Release/model.xyz", 1.0);
for (auto& f : funcs)
{
	f->setPoints(pointClouds[f->name()]);
}

for (const auto& f1 : funcs)
{
	std::cout << f1->name() << " connected with: ";

	for (const auto& f2 : funcs)
	{		
		if (lmu::areConnected(graph, f1, f2))
		{
			std::cout << " " << f2->name();
		}
	}

	std::cout << std::endl;
}

/*for (const auto& func : funcs)
{	

	viewer.data().add_points(func->pointsCRef().leftCols(3), Eigen::Matrix<double, 1, 3>(0, 0, 0));//;func->pointsCRef().rightCols(3));
	
}*/

lmu::filterPoints(funcs, graph, gradientStepSize);

double score = lmu::computeNormalizedGeometryScore(node, funcs, gradientStepSize);

std::cout << "SCORE: " << score << std::endl;

double distThreshold = 0.9;
double angleThreshold = 0.9;

SampleParams p{ gradientStepSize, distThreshold, angleThreshold };
//lmu::getUnionPartitionsByPrimeImplicants(graph, p);

int i = 0;
for (const auto& func : funcs)
{
	//if (/*func->name() != "cylinder_0" && func->name() != "cube_0" && func->name() != "cube_2" && */func->name() != "cube_4" && func->name() != "cylinder_0")
	//	continue;
	//if (func->name() != "cube_0")
	//	continue;

	Eigen::Matrix<double, 1, 3> c; 
	
	switch (i % 6)
	{
	case 0: 
		c = Eigen::Matrix<double, 1, 3>(1, 0, 0);
		break;
	case 1:
		c = Eigen::Matrix<double, 1, 3>(1, 0, 1);
		break; 
	case 2:
		c = Eigen::Matrix<double, 1, 3>(1, 1, 0);
		break;
	case 3:
		c = Eigen::Matrix<double, 1, 3>(1, 1, 1);
		break;
	case 4:
		c = Eigen::Matrix<double, 1, 3>(0, 0, 1);
		break;
	case 5:
		c = Eigen::Matrix<double, 1, 3>(0, 1, 1);
		break;
	}

	viewer.data().add_points(func->pointsCRef().leftCols(3), c);//;func->pointsCRef().rightCols(3));

	i++;
}


writeNode(node, "tree.dot");

	auto mesh = lmu::computeMesh(node, Eigen::Vector3i(100, 100, 100));
	viewer.data().set_mesh(mesh.vertices, mesh.indices);

	viewer.data().point_size = 5.0;
	viewer.core.background_color = Eigen::Vector4f(1, 1, 1, 1);

	viewer.launch();
	
}