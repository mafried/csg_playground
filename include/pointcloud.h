#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <unordered_map>

namespace lmu
{
  struct Mesh;
  class CSGNode;
  
  using PointCloud = Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>;
  using PointCloudWithLabels = Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor>;

  void writePointCloud(const std::string& file, PointCloud& points);
  void writePointCloudXYZ(const std::string& file, PointCloud& points);
  void writePointCloudXYZ(const std::string& file, const std::unordered_map<std::string, PointCloud>& points);

  PointCloud readPointCloud(std::istream& s, double scaleFactor = 1.0);
  PointCloud readPointCloud(const std::string& file, double scaleFactor=1.0);
  PointCloud readPointCloudXYZ(const std::string& file, double scaleFactor=1.0);
  std::unordered_map<std::string, PointCloud> readPointCloudXYZPerFunc(const std::string& file, double scaleFactor = 1.0);
  
  PointCloud pointCloudFromMesh(const lmu::Mesh & mesh, const lmu::CSGNode& node, double delta, double samplingRate, double errorSigma);
  
  Eigen::MatrixXd getSIFTKeypoints(Eigen::MatrixXd& points, double minScale, double minContrast, int numOctaves, int numScalesPerOctave, bool normalsAvailable);

  double computeAABBLength(const PointCloud& pc);
  Eigen::Vector3d computeAABBDims(const PointCloud& pc);

  struct PointCloudCharacteristics
  {
	  double meanDistance;
	  double maxDistance; 
	  double minDistance;
	  double medianDistance;
  };

  PointCloudCharacteristics getPointCloudCharacteristics(const PointCloud& pc, int k, double octreeResolution);

}

#endif