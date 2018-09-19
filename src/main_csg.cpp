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
#include "csgnode_helper.h"
#include "evolution.h"
#include "curvature.h"
#include "dnf.h"
#include "statistics.h"
#include "constants.h"


using namespace lmu;


enum class ApproachType
{
  None = 0,
    BaselineGA, 
    Partition
    };


ApproachType approachType = ApproachType::BaselineGA;
ParallelismOptions paraOptions = ParallelismOptions::GAParallelism;


static void usage(const char* pname) {
  std::cout << "Usage:" << std::endl;
  std::cout << pname << " points.xyz shapes.prim samplingStepSize maxDistance method outBasename" 
	    << std::endl;
  std::cout << std::endl;
  std::cout << "Example: " << pname 
	    << " model.xyz model.prim 0.03 0.01 GA|SHAPIRO|SHAPIRO_PARTITION model" << std::endl;
}


int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  if (argc != 7) {
    usage(argv[0]);
    return -1;
  }


  std::string mode = argv[5];


  Eigen::AngleAxisd rot90x(M_PI / 2.0, Vector3d(0.0, 0.0, 1.0));

  CSGNode node(nullptr);

  double samplingStepSize = std::stod(argv[3]); //0.03; 
  double maxDistance = std::stod(argv[4]); //0.01;

  std::string pcName = argv[1]; //"model.xyz";
  auto pointCloud = lmu::readPointCloudXYZ(pcName, 1.0);

  std::string primName = argv[2]; //"model.prim";
  std::vector<ImplicitFunctionPtr> shapes; 
  shapes = lmu::fromFilePRIM(primName);
  auto dims = lmu::computeDimensions(shapes);

  auto graph = lmu::createConnectionGraph(shapes, std::get<0>(dims), std::get<1>(dims), samplingStepSize);
		
  lmu::writeConnectionGraph("connectionGraph.dot", graph);

  lmu::ransacWithSim(pointCloud.leftCols(3), pointCloud.rightCols(3), maxDistance, shapes);
	
  //pointCloud = lmu::filterPrimitivePointsByCurvature(shapes, 0.01, lmu::computeOutlierTestValues(shapes), FilterBehavior::FILTER_FLAT_SURFACES, false);
  //shapes.clear();
  //lmu::ransacWithSim(pointCloud.leftCols(3), pointCloud.rightCols(3), 0.05, shapes);

  lmu::movePointsToSurface(shapes, false, 0.0001);

  //auto dnf = lmu::computeShapiro(shapes, true, lmu::Graph(), { 0.001 });
  //auto res = lmu::computeCSGNode(shapes, graph, { 0.001 });//lmu::DNFtoCSGNode(dnf);

  CSGNode res = op<Union>();

  SampleParams p{ 0.001 };

  auto partitions = lmu::getUnionPartitionsByPrimeImplicants(graph, p);

  if (mode == "SHAPIRO") {
    auto dnf = lmu::computeShapiro(shapes, true, lmu::Graph(), p);
    res = lmu::DNFtoCSGNode(dnf);
    //res = lmu::computeCSGNode(shapes, graph, p);
  } else if (mode == "SHAPIRO_PARTITION")
    res = lmu::computeShapiroWithPartitions(partitions, p);
  else if (mode == "GA") {
    // Some comments:
    // * currently I am not using any parallelism options 
    // * it is not tested
    // * connectionGraph is set to Graph() (same as for the GA used for 
    // WSCG18 as far as I understand), but then it means it can not be used 
    // for the max-size computation for the trees.
    res = lmu::computeGAWithPartitions(partitions);
  } else {
    std::cerr << "Invalid mode: " << mode << std::endl;
    std::cerr << "Select one of: SHAPIRO, SHAPIRO_PARTITION, GA" << std::endl;
    return -1;
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
