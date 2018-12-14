#define BOOST_PARAMETER_MAX_ARITY 12

#include <igl/writeOBJ.h>

#include <Eigen/Core>
#include <iostream>
#include <tuple>
#include <chrono>
#include <string>

#include "mesh.h"
#include "ransac.h"
#include "pointcloud.h"
#include "collision.h"
#include "congraph.h"

//#include "tests.h"

#include "csgnode_evo.h"
#include "csgnode_evo_v2.h"
#include "csgnode_helper.h"
#include "evolution.h"
#include "curvature.h"
#include "dnf.h"
#include "statistics.h"
#include "constants.h"
#include "params.h"


using namespace lmu;


/*
enum class ApproachType
{
  None = 0,
  BaselineGA, 
  Partition
};
*/


enum class PartitionType {
  none=0,
  pi, //prime implicant
  piWithPruning,
  ap // articulation point
};


enum class CSGRecoveryType {
  ga=0, 
  shapiro
};


static void usage(const char* pname) {
  std::cout << "Usage:" << std::endl;
  std::cout << pname << " points.xyz shapes.prim params.ini"
	    << " partitionType recoveryType outBasename" 
	    << std::endl;
  std::cout << std::endl;
  std::cout << "Example: " << pname 
	    << " model.xyz model.prim params.ini none|pi|piWithPruning|ap ga|ga2|shapiro model" << std::endl;
}


int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  if (argc != 7) {
    usage(argv[0]);
    return -1;
  }

  CSGNode node(nullptr);

  ParameterSet params(argv[3]);
  params.print();
  
  double samplingStepSize = params.getDouble("Sampling", "StepSize", 0.0);
  double maxDistance = params.getDouble("Sampling", "MaxDistance", 0.03);
  double maxAngleDistance = params.getDouble("Sampling", "MaxAngleDistance", M_PI / 18.0);
  double errorSigma = params.getDouble("Sampling", "ErrorSigma", 0.01);
  double connectionGraphSamplingStepSize = params.getDouble("Sampling", "ConnectionGraphSamplingStepSize", 0.01);
  
  std::string pcName = argv[1]; // "model.xyz";

  auto pointCloud = lmu::readPointCloudXYZ(pcName, 1.0);

  std::string primName = argv[2]; // "model.prim";

  std::vector<ImplicitFunctionPtr> shapes; 
  shapes = lmu::fromFilePRIM(primName);
  
  std::cout << "Compute Connection Graph" << std::endl;
  
  auto dims = lmu::computeDimensions(shapes);
  auto graph = lmu::createConnectionGraph(shapes, std::get<0>(dims), std::get<1>(dims), connectionGraphSamplingStepSize);
	
  for (const auto& f1 : shapes)
  {
	  std::cout << f1->name() << " connected with: ";

	  for (const auto& f2 : shapes)
	  {
		  if (lmu::areConnected(graph, f1, f2))
		  {
			  std::cout << " " << f2->name();
		  }
	  }

	  std::cout << std::endl;
  }

  lmu::writeConnectionGraph("connectionGraph.dot", graph);

  std::cout << "Simulate RANSAC" << std::endl;

  double pointsInPrimitiveRate = lmu::ransacWithSim(pointCloud, CSGNodeSamplingParams(maxDistance, maxAngleDistance, errorSigma, samplingStepSize), shapes);

  std::cout << "Complete point cloud size: " << pointCloud.rows() << std::endl;
  std::cout << "Points in primitives: " << pointsInPrimitiveRate << "%" << std::endl;

  CSGNode res = op<Union>();

  double gradientStepSize = params.getDouble("Sampling", "GradientStepSize", 0.001);
  double distThreshold = params.getDouble("Sampling", "DistanceThreshold", 0.9);
  double angleThreshold = params.getDouble("Sampling", "AngleThreshold", 0.9);


  SampleParams p{ gradientStepSize, distThreshold, angleThreshold };

  
  //lmu::arrangeGradients(shapes, graph, gradientStepSize);

  //lmu::reducePoints(shapes, graph, gradientStepSize);
  //lmu::reducePointsBasedOnVariance(shapes, graph, gradientStepSize);

  lmu::filterPoints(shapes, graph, gradientStepSize);
  //lmu::movePointsToSurface(shapes, true, 0.0001);


  std::string partitionType = argv[4];
  std::string recoveryType = argv[5];

  // Some comments:
  // * currently I am not using any parallelism options 
  // * for the GA: connectionGraph is set to Graph() (same as for the GA used for 
  // WSCG18 as far as I understand), but then it means it can not be used 
  // for the max-size computation for the trees.
  // * should we call optimizeCSGNodeStructure for each recoveryType? 
  
  if (partitionType == "none") 
  {
    if (recoveryType == "shapiro") {
      auto dnf = lmu::computeShapiro(shapes,true, graph, p);
      res = lmu::DNFtoCSGNode(dnf);
    } else if (recoveryType == "ga") {
      res = createCSGNodeWithGA(shapes, params, graph);
      optimizeCSGNodeStructure(res);
	//} else if (recoveryType == "ga2") {
	  //res = createCSGNodeWithGAV2(graph, params);
	  //optimizeCSGNodeStructure(res); Some issues here
    } else {
      std::cerr << "Invalid recovery type" << std::endl;
      usage(argv[0]);
      return -1;
    }
  } else {
    // First compute the partition based on the partition type
    std::vector<lmu::Graph> partitions;
    if (partitionType == "pi") {
      partitions = lmu::getUnionPartitionsByPrimeImplicants(graph, p);
    } else if (partitionType == "piWithPruning") {
      partitions = lmu::getUnionPartitionsByPrimeImplicantsWithPruning(graph, p);
    } else if (partitionType == "ap") {
      partitions = lmu::getUnionPartitionsByArticulationPoints(graph);
    } else {
      std::cerr << "Invalid partition type" << std::endl;
      usage(argv[0]);
      return -1;
    }

    // Then call Shapiro or GA with partitions
    if (recoveryType == "shapiro") {
      res = lmu::computeShapiroWithPartitions(partitions, p);
    } else if (recoveryType == "ga") {
      res = lmu::computeGAWithPartitions(partitions, params);
	//} else if (recoveryType == "ga2") {
	//	res = lmu::computeGAWithPartitionsV2(partitions, params);
	}
	else {
      std::cerr << "Invalid recovery type" << std::endl;
      usage(argv[0]);
      return -1;
    }
  }


  std::string outBasename = argv[6];
  lmu::writeNode(res, outBasename + "_tree.dot");

  auto mesh = lmu::computeMesh(res, Eigen::Vector3i(100, 100, 100));

  igl::writeOBJ(outBasename + "_mesh.obj", mesh.vertices, mesh.indices);

  
  //std::cout << lmu::espressoExpression(dnf) << std::endl;
	
  //std::cout << "Before: " << pointCloud.rows() << std::endl;
  //pointCloud = lmu::filterPrimitivePointsByCurvature(shapes, 0.01, 7 , FilterBehavior::FILTER_CURVY_SURFACES), true;
  //std::cout << "After: " << pointCloud.rows() << std::endl;

  //pointCloud = getSIFTKeypoints(pointCloud, 0.01, 0.001, 3, 4, false);
  //auto colors = lmu::computeCurvature(pointCloud.leftCols(3), node, 0.01, true);
  //std::cout << "Considered Clause " << g_clause << std::endl;


  return 0;
}
