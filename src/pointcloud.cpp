#include <fstream>
#include <iostream>
#include <random>
#include <numeric>
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <igl/point_mesh_squared_distance.h>

#include "..\include\pointcloud.h"
#include "..\include\mesh.h"
#include "..\include\csgnode.h"

#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>

lmu::PointCloud lmu::pointCloudFromVector(const std::vector<Eigen::Matrix<double, 1, 6>>& points)
{
	PointCloud pc(points.size(), 6);
	
	for (int i = 0; i < points.size(); ++i)
		pc.row(i) << points[i];

	return pc;
}

void lmu::writePointCloud(const std::string& file, PointCloud& points)
{
	//if (points.cols() != 6)
	//	throw std::runtime_error("Number of columns must be 6.");

	std::ofstream s(file); 

	s << points.rows() << " " << points.cols() << std::endl;

	for (int i = 0; i < points.rows(); i++)
	{
		for (int j = 0; j < points.cols(); j++)
		{
			s << points.row(i).col(j) << " ";
		}
		s << std::endl;
	}
}


void lmu::writePointCloudXYZ(const std::string& file, PointCloud& points)
{
  std::ofstream s(file); 

  for (int i = 0; i < points.rows(); i++)
    {
      for (int j = 0; j < points.cols(); j++)
	{
	  s << points.row(i).col(j) << " ";
	}
      s << std::endl;
    }
}

void lmu::writePointCloudXYZ(const std::string& file, const std::unordered_map<std::string, PointCloud>& points)
{
	std::ofstream s(file);

	for (const auto& f : points)
	{
		s << "p" << std::endl;
		s << f.first << std::endl;

		const auto& pts = f.second;

		for (int i = 0; i < pts.rows(); i++)
		{
			for (int j = 0; j < pts.cols(); j++)
			{
				s << pts.row(i).col(j) << " ";
			}
			s << std::endl;
		}
	}
}

lmu::PointCloud createPC(const std::vector<std::vector<double>>& data, double scaleFactor)
{
	size_t numRows = data.size();
	size_t numCols = 6;

	lmu::PointCloud points(numRows, numCols);

	for (int i = 0; i < points.rows(); i++)
	{
		for (int j = 0; j < points.cols(); j++)
		{
			double v = data[i][j];

			if (j < 3)
				v = v * scaleFactor;

			points(i, j) = v;
		}
	}

	return points;
}

std::unordered_map<std::string, lmu::PointCloud> lmu::readPointCloudXYZPerFunc(const std::string& file, double scaleFactor)
{
	std::ifstream s(file);
	std::unordered_map<std::string, lmu::PointCloud> res; 
	std::vector<std::vector<double>> pwn;
	std::string primitiveName;

	std::cout << "Start reading point clouds." << std::endl;

	std::streampos oldpos = 0;

	while (!s.eof() && oldpos != std::streampos(-1)) 
	{
		oldpos = s.tellg();  
		std::string isPrimitive;
		s >> isPrimitive;
		if (isPrimitive != "p")
		{
			s.seekg(oldpos);
			std::vector<double> tmp(6);
			s >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3] >> tmp[4] >> tmp[5];
			pwn.push_back(tmp);
		}
		else
		{
			if (!primitiveName.empty())
			{
				std::cout << "Read points for primitive " << primitiveName << " " << pwn.size() << std::endl;
				res.insert({ primitiveName, createPC(pwn, scaleFactor) });
				pwn.clear();				
			}
			s >> primitiveName;
		}		
	}
	
	res[primitiveName] = createPC(pwn, scaleFactor);

	return res;
}

lmu::PointCloud lmu::readPointCloud(std::istream& s, double scaleFactor)
{
	size_t numRows;
	size_t numCols;

	s >> numRows;
	s >> numCols;

	std::cout << "Read PointCloud: " << numRows << " " << numCols << std::endl;

	Eigen::MatrixXd points(numRows, numCols);


	for (int i = 0; i < points.rows(); i++)
	{
		for (int j = 0; j < points.cols(); j++)
		{
			double v;
			s >> v;

			if (j < 3)
				v = v * scaleFactor;

			points(i, j) = v;
		}
	}

	return points;
}

lmu::PointCloud lmu::readPointCloud(const std::string& file, double scaleFactor)
{
	std::ifstream s(file);
	return readPointCloud(s, scaleFactor);
}


// Assume each line contains
// x y z nx ny nz
lmu::PointCloud lmu::readPointCloudXYZ(const std::string& file, double scaleFactor)
{
  std::ifstream s(file);

  std::vector<std::vector<double>> pwn;
  while (!s.eof()) {
    std::vector<double> tmp(6);
    s >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3] >> tmp[4] >> tmp[5];
    pwn.push_back(tmp);
  }

  size_t numRows = pwn.size(); 
  size_t numCols = 6;

  std::cout << numRows << " " << numCols << std::endl;

  PointCloud points(numRows, numCols);

  for (int i = 0; i < points.rows(); i++)
    {
      for (int j = 0; j < points.cols(); j++)
	{
	  double v = pwn[i][j]; 

	  if (j < 3)
	    v = v * scaleFactor;

	  points(i,j) = v;
	}
    }

  return points;
}


lmu::PointCloud lmu::pointCloudFromMesh(const lmu::Mesh& mesh, double delta, double samplingRate, double errorSigma)
{
	Eigen::Vector3d min = mesh.vertices.colwise().minCoeff();
	Eigen::Vector3d max = mesh.vertices.colwise().maxCoeff();

	std::cout << "min: " << min << " max: " << max << std::endl;

	Eigen::Vector3d d(samplingRate, samplingRate, samplingRate);
	min -= d*2.0;
	max += d*2.0;

	Eigen::Vector3i numSamples((max.x() - min.x()) / samplingRate, (max.y() - min.y()) / samplingRate, (max.z() - min.z()) / samplingRate);

	std::cout << "Samples: " << numSamples << std::endl;

	Eigen::MatrixXd samplingPoints;
	size_t numSamplingPoints = numSamples.x()*numSamples.y()*numSamples.z();
	samplingPoints.resize(numSamplingPoints, 3);

	int i = 0;
	for (int x = 0; x < numSamples.x(); ++x)
		for (int y = 0; y < numSamples.y(); ++y)
			for (int z = 0; z < numSamples.z(); ++z)
			{
				Eigen::Vector3d p = min + Eigen::Vector3d(x, y, z) * samplingRate;
				samplingPoints.row(i) = p;
				i++;
			}

	std::cout << "num samples: " << std::endl << numSamples << std::endl;

	Eigen::VectorXd sqrD;
	Eigen::VectorXi I;
	Eigen::MatrixXd C;

	std::cout << "Get sampling points" << std::endl;

	igl::point_mesh_squared_distance(samplingPoints, mesh.vertices, mesh.indices, sqrD, I, C);

	std::vector<Eigen::Vector3d> remainingPoints;

	std::random_device rd{};
	std::mt19937 gen{ rd() };

	std::normal_distribution<> dx{ 0.0 , errorSigma };
	std::normal_distribution<> dy{ 0.0 , errorSigma };
	std::normal_distribution<> dz{ 0.0 , errorSigma };

	for (int i = 0; i < numSamplingPoints; i++)
	{
		Eigen::Vector3d noise = Eigen::Vector3d(dx(gen), dy(gen), dz(gen));
		Eigen::Vector3d samplingPoint = samplingPoints.row(i).leftCols(3).transpose();
		samplingPoint += noise;
		
		//double sd = node.signedDistance(samplingPoint);

		if (std::sqrt(sqrD(i)) < delta )//&& std::abs(sd) < delta)
		{
			remainingPoints.push_back(samplingPoint);
		}
	}

	std::cout << "Sample points with error." << std::endl;

	PointCloud res;
	res.resize(remainingPoints.size(), 6);

	
	i = 0;
	for (const auto& point : remainingPoints)
	{	
		res.block<1, 3>(i, 0) = point;

		//Eigen::Vector3d normal = node.signedDistanceAndGradient(point).bottomRows(3).transpose();
		
		res.block<1, 3>(i, 3) = Eigen::Vector3d(0, 0, 0);//normal;

		i++;
	}

	return res;
}

lmu::PointCloudCharacteristics lmu::getPointCloudCharacteristics(const PointCloud & pc, int k, double octreeResolution)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	cloud->width = pc.rows();
	cloud->height = 1;
	cloud->is_dense = false;
	cloud->points.resize(pc.rows());
		
	double nodeSize = octreeResolution * computeAABBLength(pc);

	std::cout << "Octree Node Size: " << nodeSize << std::endl;

	pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(nodeSize);
	octree.setInputCloud(cloud);
	octree.addPointsFromInputCloud();

	for (int i = 0; i < pc.rows(); ++i)
	{
		cloud->points[i].x = pc.row(i).x();
		cloud->points[i].y = pc.row(i).y();
		cloud->points[i].z = pc.row(i).z();
	}
	
	std::vector<double> distances(pc.rows());
	for (int i = 0; i < pc.rows(); ++i)
	{
		Eigen::Vector3d searchP = pc.row(i).leftCols(3);

		std::vector<int> indices(k);
		std::vector<float> kdistances(k);
				
		octree.nearestKSearch(i, k, indices, kdistances);
		std::cout << i << " of " << pc.rows() <<  std::endl;

		distances[i] = std::accumulate(kdistances.begin(), kdistances.end(), 0.0) / (double)kdistances.size();
	}

	std::sort(distances.begin(), distances.end());

	PointCloudCharacteristics pcc;
	pcc.maxDistance = distances.back();
	pcc.minDistance = distances.front(); 
	pcc.medianDistance = distances[distances.size() / 2];
	pcc.meanDistance = std::accumulate(distances.begin(), distances.end(), 0.0) / (double)distances.size();

	return pcc;
}

void lmu::projectPointCloudOnPlane(PointCloud & pc, const Eigen::Vector3d & p, const Eigen::Vector3d & n)
{
	for (int i = 0; i < pc.rows(); ++i)
	{
		Eigen::Vector3d pt = pc.block<1, 3>(i, 0).transpose();

		Eigen::Vector3d projPt = p + pt - (pt.dot(n)) * n;

		pc.block<1, 3>(i, 0) << projPt.transpose();
	}
}

void lmu::projectPointCloudOnSphere(PointCloud & pc, const Eigen::Vector3d & p, double r)
{
	for (int i = 0; i < pc.rows(); ++i)
	{
		Eigen::Vector3d pt = pc.block<1, 3>(i, 0).transpose() - p;
		
		pt = (r / pt.norm()) * pt;
		pt += p;

		pc.block<1, 3>(i, 0) << pt.transpose();

		//double d = (pt-p).norm() - r;
		//std::cout << "D: " << d;
	}
}

void lmu::projectPointCloudOnCylinder(PointCloud & pc, const Eigen::Vector3d & p, const Eigen::Vector3d & dir, double r)
{
	for (int i = 0; i < pc.rows(); ++i)
	{
		Eigen::Vector3d pRay = pc.block<1, 3>(i, 0).transpose() - p;
		
		//TODO
	}
}

double lmu::computeAABBLength(const lmu::PointCloud& points)
{
  Eigen::VectorXd min = points.colwise().minCoeff();
  Eigen::VectorXd max = points.colwise().maxCoeff();
  Eigen::VectorXd diag = max - min;
  return diag.norm();
}

Eigen::Vector3d lmu::computeAABBDims(const PointCloud& pc)
{
	Eigen::Vector3d min = pc.leftCols(3).colwise().minCoeff();
	Eigen::Vector3d max = pc.leftCols(3).colwise().maxCoeff();
	
	return (max - min).cwiseAbs();
}

Eigen::VectorXd getDistances(const Eigen::Vector3d &p, const lmu::PointCloud& pc) {
	auto v = Eigen::VectorXd(pc.rows()); 

	for (int i = 0; i < pc.rows(); ++i)
		v.row(i) << (p - pc.row(i).leftCols(3).transpose()).squaredNorm();

	return v;
}

lmu::PointCloud lmu::farthestPointSampling(const PointCloud & pc, int k)
{
	if (pc.rows() <= k)
		return pc;

	std::random_device r;
	std::default_random_engine e1(r());
	std::uniform_int_distribution<int> uniformDist(0, pc.rows()-1);

	lmu::PointCloud spc(k, 6);
	spc.setZero();

	spc.row(0) << pc.row(uniformDist(e1));
	auto distances = getDistances(spc.row(0).leftCols(3).transpose(), pc);

	for (int i = 1; i < k; ++i)
	{
		// Take point with largest distance to point cloud and add it to the new pc.
		Eigen::VectorXd::Index maxDRowIdx;
		distances.maxCoeff(&maxDRowIdx);
		spc.row(i) = pc.row(maxDRowIdx);

		// Recompute distances as the column-wise minimum of old distance vector and distance vector based on current point.
		distances = distances.cwiseMin(getDistances(spc.row(i).leftCols(3).transpose(), pc));
	}

	return spc;
}

#include <pcl/ml/kmeans.h>

std::vector<std::tuple<Eigen::Vector3d, lmu::PointCloud>> lmu::kMeansClustering(const PointCloud & p, int k)
{
	pcl::Kmeans real(p.rows(), 3);
	real.setClusterSize(k);
	for (size_t i = 0; i < p.rows(); i++)
	{
		std::vector<float> data(3);
		data[0] = p.row(i).x();
		data[1] = p.row(i).y();
		data[2] = p.row(i).z();
		real.addDataPoint(data);
	}

	real.kMeans();
	
	pcl::Kmeans::Centroids pclCentroids = real.get_centroids();
	std::cout << "centroid count: " << pclCentroids.size() << std::endl;
	std::vector<Eigen::Vector3d> centroids(pclCentroids.size());
	for (int i = 0; i < pclCentroids.size(); i++)
	{
		centroids[i] = Eigen::Vector3d(pclCentroids[i][0], pclCentroids[i][1], pclCentroids[i][2]);

		std::cout << "Centroid: " << centroids[i] << std::endl;
	}
	std::vector<std::vector<size_t>> clusters(centroids.size());

	for (size_t i = 0; i < p.rows(); i++)
	{
		Eigen::Vector3d point = p.row(i).leftCols(3).transpose();
		double minD = std::numeric_limits<double>::max();
		double cIdxMinD = 0;
		for (int j = 0; j < centroids.size(); ++j)
		{
			double d = (point - centroids[j]).squaredNorm();
			if (minD > d) {
				minD = d;
				cIdxMinD = j;
			}
		}
		clusters[cIdxMinD].push_back(i);
	}

	std::vector<std::tuple<Eigen::Vector3d, lmu::PointCloud>> res;

	for (int i = 0; i < centroids.size(); ++i)
	{
		Eigen::Vector3d c = centroids[i];
		lmu::PointCloud cPc(clusters[i].size(), 6);
		for (int j = 0; j < clusters[i].size(); j++)
		{
			cPc.row(j) << p.row(clusters[i][j]);

			//std::cout << "Cluster " << i << "Row " << j << " " << cPc.row(i) << std::endl;
		}

		//std::cout << cPc << std::endl;

		res.push_back(std::make_tuple(c, cPc));
	}
	
	return res;
}
#include <pcl/common/centroid.h>

pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudToPCLPointCloud(const lmu::PointCloud& pc)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl(new pcl::PointCloud<pcl::PointXYZ>());

	pcl->width = pc.rows();
	pcl->height = 1;
	pcl->is_dense = false;
	pcl->points.resize(pc.rows());

	for (int i = 0; i < pc.rows(); ++i)
	{
		pcl->points[i].x = pc.row(i).x();
		pcl->points[i].y = pc.row(i).y();
		pcl->points[i].z = pc.row(i).z();
	}

	return pcl;
}

#include <pcl/common/eigen.h>

Eigen::Affine3d lmu::getOrientation(const PointCloud & p)
{
	Eigen::Vector4f centroid;
	Eigen::Matrix3f covariance_matrix;

	// Extract the eigenvalues and eigenvectors
	Eigen::Vector3f eigen_values;
	Eigen::Matrix3f eigen_vectors;

	auto pcl = PointCloudToPCLPointCloud(p);

	pcl::compute3DCentroid(*pcl, centroid);

	// Compute the 3x3 covariance matrix
	pcl::computeCovarianceMatrix(*pcl, centroid, covariance_matrix);
	pcl::eigen33(covariance_matrix, eigen_vectors, eigen_values);

    https://stackoverflow.com/questions/28701213/how-to-find-principal-components-and-orientation-of-a-point-cloud-using-point-cl

	auto v1 = eigen_vectors.col(0).normalized();
	auto v2 = eigen_vectors.col(1).normalized();

	//std::cout << "EV:" << std::endl;
	//std::cout << eigen_vectors << std::endl;

	v2 = v2 - (v1 * v1.dot(v2));

  	Eigen::Matrix3f rot;
	rot.col(0) << v1;
	rot.col(1) << v2;
	rot.col(2) << v1.cross(v2);

	auto res = Eigen::Affine3d(Eigen::Translation3d(centroid.cast<double>().topRows(3))) * Eigen::Affine3d(rot.cast<double>());

	return res;
}

void lmu::transform(PointCloud &p, const Eigen::Affine3d& t)
{
	// https://eigen.tuxfamily.org/dox-devel/group__TutorialGeometry.html
	// https://stackoverflow.com/questions/38841606/shorter-way-to-apply-transform-to-matrix-containing-vectors-in-eigen

	for (int i = 0; i < p.rows(); ++i)
	{
		auto point = p.row(i).leftCols(3).transpose();
		auto normal = p.row(i).rightCols(3).transpose();

		point = (t.linear() * point) + t.translation();
		normal = t.linear() * normal;

		p.row(i) << point.transpose(), normal.transpose();
	}
}

Eigen::Vector3d lmu::computeOBBDims(const PointCloud & p)
{
	auto trans = getOrientation(p);
	auto pTransformed = p;

	//std::cout << "Computed Orientation: " << std::endl << trans.matrix()<< std::endl;

	Eigen::Matrix4d inv;
	bool invertible;
	trans.matrix().computeInverseWithCheck(inv, invertible);	
	//std::cout << "Invertible: " << invertible << std::endl;

	transform(pTransformed, trans.inverse());

	//std::cout << "Transformed" << std::endl;

	return computeAABBDims(pTransformed);
}

/*
  Eigen::MatrixXd lmu::getSIFTKeypoints(Eigen::MatrixXd& points, double minScale, double minContrast, int numOctaves, int numScalesPerOctave, bool normalsAvailable)
  {
  pcl::PointCloud<pcl::PointNormal>::Ptr pcWithNormals(new pcl::PointCloud<pcl::PointNormal>());
		
  pcWithNormals->width = points.rows();
  pcWithNormals->height = 1;
  pcWithNormals->is_dense = false;
  pcWithNormals->points.resize(points.rows());

	
	if(!normalsAvailable)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr pcWithoutNormals(new pcl::PointCloud<pcl::PointXYZ>());
		
		pcWithoutNormals->width = points.rows();
		pcWithoutNormals->height = 1;
		pcWithoutNormals->is_dense = false;
		pcWithoutNormals->points.resize(points.rows());

		for (int i = 0; i < points.rows(); ++i)
		{
			pcWithoutNormals->points[i].x = points.row(i).x();
			pcWithoutNormals->points[i].y = points.row(i).y();
			pcWithoutNormals->points[i].z = points.row(i).z();
		}

		pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_n(new pcl::search::KdTree<pcl::PointXYZ>());
	
		ne.setInputCloud(pcWithoutNormals);
		ne.setSearchMethod(tree_n);
		ne.setRadiusSearch(0.2);
		ne.compute(*pcWithNormals);

		for (int i = 0; i < points.rows(); ++i)
		{
			pcWithNormals->points[i].x = points.row(i).x();
			pcWithNormals->points[i].y = points.row(i).y();
			pcWithNormals->points[i].z = points.row(i).z();
		}
		
	}
	else
	{

		for (int i = 0; i < points.rows(); ++i)
		{
			pcWithNormals->points[i].x = points.row(i).x();
			pcWithNormals->points[i].y = points.row(i).y();
			pcWithNormals->points[i].z = points.row(i).z();

			//... add copy of normal coordinates TODO
		}
	}

	
	pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;

	pcl::PointCloud<pcl::PointWithScale>::Ptr result(new pcl::PointCloud<pcl::PointWithScale>());


	pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>());
	sift.setSearchMethod(tree);
	sift.setScales(minScale, numOctaves, numScalesPerOctave);	
	sift.setMinimumContrast(minContrast);	
	sift.setInputCloud(pcWithNormals);
	
	std::cout << "Compute" << std::endl;

	sift.compute(*result);

	std::cout << "Compute Done" << std::endl;

	Eigen::MatrixXd resultMat;
	resultMat.resize(result->size(), 3);
	for (int i = 0; i < resultMat.rows(); i++)
	{
		resultMat.block<1, 3>(i, 0) << result->points[i].x, result->points[i].y, result->points[i].z;
			//pcWithNormals->points[i].normal_x, pcWithNormals->points[i].normal_y, pcWithNormals->points[i].normal_z;
	}

	std::cout << "Copy Done" << std::endl;


	return resultMat;
}*/