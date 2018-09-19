#define BOOST_PARAMETER_MAX_ARITY 12

#include <Eigen/Core>
#include <iostream>
#include <tuple>
#include <chrono>
#include <string>

#include "mesh.h"
#include "pointcloud.h"
#include "csgnode_helper.h"
#include "constants.h"


using namespace lmu;


static void usage(const char* pname) {
  std::cout << "Usage:" << std::endl;
  std::cout << pname << " modelID samplingStepSize maxDistance noiseSigma outBasename" << std::endl;
  std::cout << std::endl;
  std::cout << "Example: " << pname << " 11 0.03 0.01 0.03 model" << std::endl;
}


int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;


  if (argc != 6) {
    usage(argv[0]);
    return -1;
  }

  int nodeIdx = std::stoi(argv[1]);

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
  else if (nodeIdx == 4) 
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
			    })
			})
		  })
	      });
    }
  else if (nodeIdx == 5) 
    {
      node = op<Union>(
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
	    });
    }
  else if (nodeIdx == 6)
    {
      node = op<Union>({
	  geo<IFBox>((Eigen::Affine3d)(Eigen::Translation3d(-0.3, 0, -0.5) *rot90x), Eigen::Vector3d(0.2, 0.8, 0.9), 2, "Box_3"),
	    geo<IFBox>((Eigen::Affine3d)(Eigen::Translation3d(-0.3, 0, -0.5)), Eigen::Vector3d(0.2, 0.8, 0.9), 2, "Box_2"),
	    });
    }
  else if (nodeIdx == 7)
    {
      node = op<Difference>({
	  geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(-0.2, 0, -1)), 0.2, 0.8, "Cylinder_2"),
	    geo<IFCylinder>((Eigen::Affine3d)(Eigen::Translation3d(-0.2, 0, -1)), 0.1, 2, "Cylinder_3")
	    });
    }
  else if (nodeIdx == 8) 
    {
      node = op<Difference>(
	{
	  geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, -1.0, 0.2), 0.2, "Sphere_4"),
	    geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(-0.5, -1.0, 0.6), 0.4, "Sphere_5")
	    });
    }
  else if (nodeIdx == 9) 
    {
      node = op<Union>(
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
	    });
    }
  else if (nodeIdx == 10)
    {
      node = op<Union>(
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
	
	    });
    }
  else if (nodeIdx == 11)
    {
      // previous default model
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
    }
  else if (nodeIdx == 12) 
    {
      node =
	op<Difference>(
	  {
	    op<Union>(
	      {
		geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, 0.0, 0), 0.25, "Sphere_1"),
		  geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, 0.3, 0), 0.25, "Sphere_2"),
		  geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0, 0.6, 0), 0.25, "Sphere_3"),
		  }),
	      geo<IFSphere>((Eigen::Affine3d)Eigen::Translation3d(0.3, 0.5, 0), 0.25, "Sphere_4")
	      });
    }
  else
    {
      std::cerr << "Could not get node. Idx: " << nodeIdx << std::endl;
      return -1;
    }


  double samplingStepSize = std::stod(argv[2]); //0.03; 
  double maxDistance = std::stod(argv[3]); //0.01;
  double noiseSigma = std::stod(argv[4]); //0.03;
  std::string modelBasename = argv[5];

  auto pointCloud = lmu::computePointCloud(node, samplingStepSize, maxDistance, noiseSigma);

  std::string pcName = modelBasename + ".xyz"; //"model.xyz";
  lmu::writePointCloudXYZ(pcName, pointCloud);


  std::vector<ImplicitFunctionPtr> shapes; 
  for (const auto& geoNode : allGeometryNodePtrs(node)) {
    shapes.push_back(geoNode->function());
    std::cout << "Shape: " << geoNode->function()->name() << std::endl;
  }
  std::string primName = modelBasename + ".prim"; //"model.prim";
  lmu::writePrimitives(primName, shapes);

  return 0;
}
