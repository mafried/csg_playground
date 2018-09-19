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
#include "statistics.h"
#include "constants.h"


using namespace lmu;



enum class ApproachType
{
  None = 0,
    BaselineGA, 
    Partition
    };


static void usage(const char* pname) {
  std::cout << "Usage:" << std::endl;
  std::cout << pname << " points.xyz shapes.prim samplingStepSize"
	    << " maxDistance approachType parallelismOption outBasename" << std::endl;
  std::cout << std::endl;
  std::cout << "Example: " << pname << " model.xyz model.prim 0.03 0.01 2 0 model" << std::endl;
}


int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  
  if (argc != 8) {
    usage(argv[0]);
    return -1;
  }

  
  double samplingStepSize = std::stod(argv[3]);
  double maxDistance = std::stod(argv[4]);
  ApproachType approachType = static_cast<ApproachType>(std::stoi(argv[5]));
  ParallelismOptions paraOptions = static_cast<ParallelismOptions>(std::stoi(argv[6]));

  std::cout << "Start in batch mode. Approach Type: " << static_cast<int>(approachType) 
	    << " paraOptions: " << static_cast<int>(paraOptions) << " samplingStepSize: " 
	    << samplingStepSize << " maxDistance: " << maxDistance << std::endl;
  std::cout << "Per GA Parallelism: " 
	    << static_cast<int>((paraOptions & ParallelismOptions::GAParallelism)) << std::endl;
  std::cout << "Per Clique Parallelism: " 
	    << static_cast<int>((paraOptions & ParallelismOptions::PerCliqueParallelism)) << std::endl;
  
  
  //RUN_TEST(CSGNodeTest);

  Eigen::AngleAxisd rot90x(M_PI / 2.0, Vector3d(0.0, 0.0, 1.0));

  CSGNode node(nullptr);


  std::string pcName = argv[1]; //"model.xyz";
  auto pointCloud = lmu::readPointCloudXYZ(pcName, 1.0);

  std::string primName = argv[2]; //"model.prim";
  std::vector<ImplicitFunctionPtr> shapes; 
  shapes = lmu::fromFilePRIM(primName);
  auto dims = lmu::computeDimensions(shapes);



  std::ofstream f("pipeline_info.dat");
  f << "Approach Type: " << static_cast<int>(approachType) << std::endl;
  f << "Point cloud size: " << pointCloud.rows() << std::endl;

  const double alpha = M_PI / 18.0;
  const double epsilon = 0.01;


  TimeTicker ticker;

  auto graph = lmu::createConnectionGraph(shapes, std::get<0>(dims), std::get<1>(dims), samplingStepSize);
	
  auto conGraphDur = ticker.tick();
  f << "Connection graph creation: duration: " << conGraphDur << std::endl;

  lmu::writeConnectionGraph("connectionGraph.dot", graph);



  ticker.tick();
  lmu::ransacWithSim(pointCloud.leftCols(3), pointCloud.rightCols(3), maxDistance, shapes);
  auto ransacSimDur = ticker.tick();
  f << "Assign points to primitives: duration: " << ransacSimDur << std::endl;


  ticker.tick();
  auto cliques = lmu::getCliques(graph);
  auto cliqueDur = ticker.tick();
  f << "Clique enumeration: #cliques: " << cliques.size() << " duration: " << cliqueDur << std::endl;

  CSGNode recNode(nullptr);
  try
    {
      ticker.tick();
		
      switch (approachType)
	{
	case ApproachType::BaselineGA:
	  recNode = createCSGNodeWithGA(shapes, (paraOptions & ParallelismOptions::GAParallelism) == ParallelismOptions::GAParallelism, graph);
	  f << "Full GA: duration: " << ticker.tick() << std::endl;

	  break;

	case ApproachType::Partition:
	  {
	    auto cliquesAndNodes = computeNodesForCliques(cliques, paraOptions);
	    

	    int i = 0;
	    for (auto clno : cliquesAndNodes) {
	      CSGNode no = std::get<1>(clno);
	      std::string temptree = "temp" + std::to_string(i) + ".dot";
	      writeNode(no, temptree);
	      ++i;
	    }
	    

	    optimizeCSGNodeClique(cliquesAndNodes, 100.0);

	    auto cliqueCompDur = ticker.tick();
	    f << "Per clique node computation: duration: " << cliqueCompDur;
	    recNode = mergeCSGNodeCliqueSimple(cliquesAndNodes);

	    auto mergeDur = ticker.tick();
	    f <<  "Clique Merge: duration: " << mergeDur << std::endl;

	    f << "Full Partition: duration: " << (conGraphDur + cliqueDur + cliqueCompDur + mergeDur) << std::endl;
	  }
	  break;
	default: 
	  recNode = node;
	  break;
	}
    }
  catch (const std::exception& ex)
    {
      std::cout << "Could not merge. Reason: " << ex.what() << std::endl;
      int i;
      std::cin >> i;
      return -1;
    }

  f << "Output CSG tree: size: " << numNodes(recNode) << " depth: " << depth(recNode) << " geometry score: " << computeGeometryScore(recNode, epsilon, alpha, shapes) << std::endl;


  f.close();

  std::string outBasename = argv[7];

  writeNode(recNode, outBasename + "_tree.dot");
	
  auto treeMesh = computeMesh(recNode, Eigen::Vector3i(100, 100, 100));
  igl::writeOBJ(outBasename + "_mesh.obj", treeMesh.vertices, treeMesh.indices);


  return 0;
}

