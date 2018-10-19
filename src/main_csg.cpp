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
  std::cout << pname << " points.xyz shapes.prim samplingStepSize maxDistance"
	    << " partitionType recoveryType outBasename" 
	    << std::endl;
  std::cout << std::endl;
  std::cout << "Example: " << pname 
	    << " model.xyz model.prim 0.03 0.01 none|pi|piWithPruning|ap ga|ga2|shapiro model" << std::endl;
}


int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  if (argc != 8) {
    usage(argv[0]);
    return -1;
  }


  Eigen::AngleAxisd rot90x(M_PI / 2.0, Vector3d(0.0, 0.0, 1.0));

  CSGNode node(nullptr);

  double samplingStepSize = std::stod(argv[3]); // 0.03; 
  double maxDistance = std::stod(argv[4]); // 0.01;

  std::string pcName = argv[1]; // "model.xyz";
  auto pointCloud = lmu::readPointCloudXYZ(pcName, 1.0);

  std::string primName = argv[2]; // "model.prim";
  std::vector<ImplicitFunctionPtr> shapes; 
  shapes = lmu::fromFilePRIM(primName);
  auto dims = lmu::computeDimensions(shapes);

  auto graph = lmu::createConnectionGraph(shapes, std::get<0>(dims), std::get<1>(dims), samplingStepSize);
		
  lmu::writeConnectionGraph("connectionGraph.dot", graph);

  lmu::ransacWithSim(pointCloud.leftCols(3), pointCloud.rightCols(3), maxDistance, shapes);
  //lmu::ransacWithSimMultiplePointOwners(pointCloud.leftCols(3), pointCloud.rightCols(3), maxDistance * 5, shapes);

  int totalNumPoints = 0;
  for (const auto& shape : shapes)
  {
	  int curNumPts = shape->pointsCRef().rows();
	  totalNumPoints += curNumPts;

	  std::cout << "Shape: " << shape->name() << " Points: " << curNumPts << std::endl;
  }
  std::cout << "Points in primitives: " << totalNumPoints << std::endl;
  std::cout << "Complete point cloud size: " << pointCloud.rows() << std::endl;
	
  //pointCloud = lmu::filterPrimitivePointsByCurvature(shapes, 0.01, lmu::computeOutlierTestValues(shapes), FilterBehavior::FILTER_FLAT_SURFACES, false);
  //shapes.clear();
  //lmu::ransacWithSim(pointCloud.leftCols(3), pointCloud.rightCols(3), 0.05, shapes);

  lmu::movePointsToSurface(shapes, false, 0.0001);

  //auto dnf = lmu::computeShapiro(shapes, true, lmu::Graph(), { 0.001 });
  //auto res = lmu::computeCSGNode(shapes, graph, { 0.001 });//lmu::DNFtoCSGNode(dnf);

  CSGNode res = op<Union>();

  SampleParams p{ 0.001 };


  std::string partitionType = argv[5];
  std::string recoveryType = argv[6];

  // Some comments:
  // * currently I am not using any parallelism options 
  // * for the GA: connectionGraph is set to Graph() (same as for the GA used for 
  // WSCG18 as far as I understand), but then it means it can not be used 
  // for the max-size computation for the trees.
  // * should we call optimizeCSGNodeStructure for each recoveryType? 
  
  if (partitionType == "none") 
  {
    if (recoveryType == "shapiro") {
      auto dnf = lmu::computeShapiro(shapes, true, graph, p);
      res = lmu::DNFtoCSGNode(dnf);
    } else if (recoveryType == "ga") {
      res = createCSGNodeWithGA(shapes, true);
      optimizeCSGNodeStructure(res);
	} else if (recoveryType == "ga2") {
	  res = createCSGNodeWithGAV2(graph, true);
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
      res = lmu::computeGAWithPartitions(partitions);
    } else {
      std::cerr << "Invalid recovery type" << std::endl;
      usage(argv[0]);
      return -1;
    }
  }


  std::string outBasename = argv[7];
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
