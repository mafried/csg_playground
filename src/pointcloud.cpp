#include <fstream>
#include <iostream>
#include <random>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <igl/point_mesh_squared_distance.h>

#include "..\include\pointcloud.h"
#include "..\include\mesh.h"


void lmu::writePointCloud(const std::string& file, Eigen::MatrixXd& points)
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


void lmu::writePointCloudXYZ(const std::string& file, Eigen::MatrixXd& points)
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


Eigen::MatrixXd lmu::readPointCloud(const std::string& file, double scaleFactor)
{
	std::ifstream s(file);
	
	size_t numRows; 
	size_t numCols;

	s >> numRows; 
	s >> numCols; 

	std::cout << numRows << " " << numCols << std::endl;

	Eigen::MatrixXd points(numRows, numCols);


	for (int i = 0; i < points.rows(); i++)
	{
		for (int j = 0; j < points.cols(); j++)
		{
			double v; 
			s >> v;

			if (j < 3)
				v = v * scaleFactor;

			points(i,j) = v;
		}
	}

	return points;
}


// Assume each line contains
// x y z nx ny nz
Eigen::MatrixXd lmu::readPointCloudXYZ(const std::string& file, double scaleFactor)
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

  Eigen::MatrixXd points(numRows, numCols);

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


Eigen::MatrixXd lmu::pointCloudFromMesh(const lmu::Mesh& mesh, double delta, double samplingRate, double errorSigma)
{
	Eigen::Vector3d min = mesh.vertices.colwise().minCoeff();
	Eigen::Vector3d max = mesh.vertices.colwise().maxCoeff();

	std::cout << "min: " << min << " max: " << max << std::endl;

	Eigen::Vector3d d(samplingRate, samplingRate, samplingRate);
	min -= d*2.0;
	max += d*2.0;

	Eigen::Vector3i numSamples((max.x() - min.x()) / samplingRate, (max.y() - min.y()) / samplingRate, (max.z() - min.z()) / samplingRate);

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
	std::vector<Eigen::Vector3d> remainingNormals;

	for (int i = 0; i < numSamplingPoints; i++)
	{
		if (sqrD(i) < delta)
		{
			remainingPoints.push_back(samplingPoints.row(i));

			int faceIndex = I.row(i).x();
			if (faceIndex < 0 || faceIndex >= mesh.indices.rows())
			{
				std::cout << "Invalid face index: " << faceIndex << " available rows: " << mesh.indices.rows() << std::endl;
				remainingNormals.push_back(Eigen::Vector3d(0, 0, 0));
				continue;
			}

			auto face = mesh.indices.row(faceIndex);
			
			int vertexIndex = face.x();

			if (vertexIndex < 0 || vertexIndex >= mesh.normals.rows())
			{
				std::cout << "Invalid vertex index: " << vertexIndex << " available rows: " << mesh.normals.rows() << std::endl;
				remainingNormals.push_back(Eigen::Vector3d(0, 0, 0));
				continue;
			}

			auto normal = mesh.normals.row(vertexIndex);		
			remainingNormals.push_back(normal);
		}
	}

	std::cout << "Sample points with error." << std::endl;

	Eigen::MatrixXd res;
	res.resize(remainingPoints.size(), 6);

	std::random_device rd{};
	std::mt19937 gen{ rd() };

	std::normal_distribution<> dx{ 0.0 , errorSigma };
	std::normal_distribution<> dy{ 0.0 , errorSigma };
	std::normal_distribution<> dz{ 0.0 , errorSigma };


	i = 0;
	for (const auto& point : remainingPoints)
	{
		res.block<1, 3>(i, 0) = point;// +Eigen::Vector3d(dx(gen), dy(gen), dz(gen));

		res.block<1, 3>(i, 3) = remainingNormals[i];

		i++;
	}

	return res;
}


double computeAABBLength(Eigen::MatrixXd& points) {
  Eigen::VectorXd min = points.colwise().minCoeff();
  Eigen::VectorXd max = points.colwise().maxCoeff();
  Eigen::VectorXd diag = max - min;
  return diag.norm();
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