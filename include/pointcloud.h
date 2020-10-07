#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <unordered_map>

#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Simple_cartesian.h>


namespace lmu
{
  struct Mesh;
  class CSGNode;
  
  using PointCloud = Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>;
  using PointCloudWithLabels = Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor>;

  PointCloud empty_pc();

  PointCloud pointCloudFromVector(const std::vector<Eigen::Matrix<double, 1, 6>>& points);
  PointCloud mergePointClouds(const std::vector<PointCloud>& pointClouds);

  void writePointCloud(const std::string& file, const PointCloud& points);
  void writePointCloudXYZ(const std::string& file, PointCloud& points);
  void writePointCloudXYZ(const std::string& file, const std::unordered_map<std::string, PointCloud>& points);

  PointCloud readPointCloud(std::istream& s, double scaleFactor = 1.0);
  PointCloud readPointCloud(const std::string& file, double scaleFactor=1.0);
  PointCloud readPointCloudXYZ(const std::string& file, double scaleFactor=1.0);
  std::unordered_map<std::string, PointCloud> readPointCloudXYZPerFunc(const std::string& file, double scaleFactor = 1.0);
  
  void scalePointCloud(PointCloud& pc, double scaleFactor = 1.0);

  PointCloud add_gaussian_noise(const PointCloud& pc, double pos_std_dev, double n_std_dev);

  PointCloud to_canonical_frame(const PointCloud& pc, const Eigen::Vector3d& min, const Eigen::Vector3d& max);
  PointCloud to_canonical_frame(const PointCloud& pc);

  PointCloud pointCloudFromMesh(const lmu::Mesh & mesh, double delta, double samplingRate, double errorSigma);

  double computeAABBLength(const PointCloud& pc);
  Eigen::Vector3d computeAABBDims(const PointCloud& pc);

  PointCloud farthestPointSampling(const PointCloud& p, int k);

  void transform(PointCloud &p, const Eigen::Affine3d& t);

  void projectPointCloudOnPlane(PointCloud& pc, const Eigen::Vector3d& p, const Eigen::Vector3d& n);
  void projectPointCloudOnSphere(PointCloud& pc, const Eigen::Vector3d& p, double r);
  void projectPointCloudOnCylinder(PointCloud& pc, const Eigen::Vector3d& p, const Eigen::Vector3d& dir, double r);


  typedef CGAL::Simple_cartesian<double> Kernel;
  typedef CGAL::Search_traits_3<Kernel> TreeTraits;
  typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits> Neighbor_search;
  typedef Neighbor_search::Tree PointTree;

  struct NearestNeighborSearch
  {
	  NearestNeighborSearch(const PointCloud& pc);

	  Eigen::Vector3d get_nn(const Eigen::Vector3d& p) const;
	  double get_nn_distance(const Eigen::Vector3d& p) const;

  private:
	  std::shared_ptr<PointTree> create_point_tree(const lmu::PointCloud & pc) const;
	  std::shared_ptr<PointTree> tree;
  };

}

#endif