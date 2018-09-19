#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace lmu
{
  struct Mesh;
  
  void writePointCloud(const std::string& file, Eigen::MatrixXd& points);
  void writePointCloudXYZ(const std::string& file, Eigen::MatrixXd& points);
  Eigen::MatrixXd readPointCloud(const std::string& file, double scaleFactor=1.0);
  Eigen::MatrixXd readPointCloudXYZ(const std::string& file, double scaleFactor=1.0);
  Eigen::MatrixXd pointCloudFromMesh(const lmu::Mesh & mesh, double delta, double samplingRate, double errorSigma);
  
  Eigen::MatrixXd getSIFTKeypoints(Eigen::MatrixXd& points, double minScale, double minContrast, int numOctaves, int numScalesPerOctave, bool normalsAvailable);

    double computeAABBLength(Eigen::MatrixXd& points);
}

#endif