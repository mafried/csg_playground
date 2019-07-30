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

  PointCloud pointCloudFromVector(const std::vector<Eigen::Matrix<double, 1, 6>>& points);

  void writePointCloud(const std::string& file, PointCloud& points);
  void writePointCloudXYZ(const std::string& file, PointCloud& points);
  void writePointCloudXYZ(const std::string& file, const std::unordered_map<std::string, PointCloud>& points);

  PointCloud readPointCloud(std::istream& s, double scaleFactor = 1.0);
  PointCloud readPointCloud(const std::string& file, double scaleFactor=1.0);
  PointCloud readPointCloudXYZ(const std::string& file, double scaleFactor=1.0);
  std::unordered_map<std::string, PointCloud> readPointCloudXYZPerFunc(const std::string& file, double scaleFactor = 1.0);
  
  PointCloud pointCloudFromMesh(const lmu::Mesh & mesh, double delta, double samplingRate, double errorSigma);
  
  Eigen::MatrixXd getSIFTKeypoints(Eigen::MatrixXd& points, double minScale, double minContrast, int numOctaves, int numScalesPerOctave, bool normalsAvailable);

  double computeAABBLength(const PointCloud& pc);
  Eigen::Vector3d computeAABBDims(const PointCloud& pc);

  PointCloud farthestPointSampling(const PointCloud& p, int k);

  std::vector<std::tuple<Eigen::Vector3d, lmu::PointCloud>> kMeansClustering(const PointCloud& p, int k);
  Eigen::Affine3d getOrientation(const PointCloud& p);

  void transform(PointCloud &p, const Eigen::Affine3d& t);
  
  Eigen::Vector3d computeOBBDims(const PointCloud &p);

  struct PointCloudCharacteristics
  {
	  double meanDistance;
	  double maxDistance; 
	  double minDistance;
	  double medianDistance;
  };

  PointCloudCharacteristics getPointCloudCharacteristics(const PointCloud& pc, int k, double octreeResolution);

  void projectPointCloudOnPlane(PointCloud& pc, const Eigen::Vector3d& p, const Eigen::Vector3d& n);
  void projectPointCloudOnSphere(PointCloud& pc, const Eigen::Vector3d& p, double r);

}

#endif