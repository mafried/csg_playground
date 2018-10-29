#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace lmu
{
  struct Mesh;
  
  using PointCloud = Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>;

  void writePointCloud(const std::string& file, PointCloud& points);
  void writePointCloudXYZ(const std::string& file, PointCloud& points);
  PointCloud readPointCloud(const std::string& file, double scaleFactor=1.0);
  PointCloud readPointCloudXYZ(const std::string& file, double scaleFactor=1.0);
  PointCloud pointCloudFromMesh(const lmu::Mesh & mesh, double delta, double samplingRate, double errorSigma);
  
  Eigen::MatrixXd getSIFTKeypoints(Eigen::MatrixXd& points, double minScale, double minContrast, int numOctaves, int numScalesPerOctave, bool normalsAvailable);

    double computeAABBLength(Eigen::MatrixXd& points);
}

#endif