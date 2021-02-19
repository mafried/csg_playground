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

lmu::PointCloud lmu::empty_pc()
{
	return PointCloud();
}

lmu::PointCloud lmu::pointCloudFromVector(const std::vector<Eigen::Matrix<double, 1, 6>>& points)
{
	PointCloud pc(points.size(), 6);
	
	for (int i = 0; i < points.size(); ++i)
		pc.row(i) << points[i];

	return pc;
}

lmu::PointCloud lmu::mergePointClouds(const std::vector<PointCloud>& pointClouds)
{
	if (pointClouds.empty())
		return lmu::PointCloud(0, 0);

	size_t size = 0; 
	for (const auto& pc : pointClouds)
		size += pc.rows();

	PointCloud res_pc(size, pointClouds[0].cols());
	size_t row_offset = 0;
	for (size_t mat_idx = 0; mat_idx < pointClouds.size(); ++mat_idx) {
		long cur_rows = pointClouds[mat_idx].rows();
		res_pc.middleRows(row_offset, cur_rows) = pointClouds[mat_idx];
		row_offset += cur_rows;
	}

	return res_pc;
}

void lmu::writePointCloud(const std::string& file, const PointCloud& points)
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

void lmu::scalePointCloud(PointCloud & pc, double scaleFactor)
{
	auto dims = computeAABBDims(pc);

	double f = dims.maxCoeff();

	for (int i = 0; i < pc.rows(); ++i)
	{
		Eigen::Vector3d p = pc.row(i).leftCols(3).transpose();

		p /= (f * scaleFactor);

		pc.row(i).leftCols(3) << p.transpose();
	}
}

lmu::PointCloud lmu::add_gaussian_noise(const PointCloud& pc, double pos_std_dev, double n_std_dev)
{
	auto new_pc = pc;

	std::random_device rd{};
	std::mt19937 gen{ rd() };

	for (int i = 0; i < pc.rows(); i++)
	{
		for (int j = 0; j < pc.cols(); j++)
		{
			auto std_dev = j < 3 ? pos_std_dev : n_std_dev;
			auto& v = new_pc(i,j);

			std::normal_distribution<> d{ v, std_dev };

			double new_v = d(gen);
			//std::cout << delta << std::endl;

			v = new_v;
		}

		Eigen::Vector3d n(new_pc.row(i).rightCols(3).transpose());
		Eigen::Vector3d p(new_pc.row(i).leftCols(3).transpose());

		new_pc.row(i) << p.transpose(), n.normalized().transpose();
	}

	return new_pc;
}

lmu::PointCloud lmu::to_canonical_frame(const PointCloud& pc)
{	
	Eigen::Vector3d min = pc.leftCols(3).colwise().minCoeff();
	Eigen::Vector3d max = pc.leftCols(3).colwise().maxCoeff();
	
	return to_canonical_frame(pc, min, max);
}

lmu::PointCloud lmu::to_canonical_frame(const PointCloud& pc, const Eigen::Vector3d& min, const Eigen::Vector3d& max)
{	
	double s = (max - min).maxCoeff();

	lmu::PointCloud centered_pc = pc;

	centered_pc.leftCols(3) = centered_pc.leftCols(3).rowwise() - min.transpose();
	centered_pc.leftCols(3) = centered_pc.leftCols(3).array().rowwise() / Eigen::Array<double,1,3>(s, s, s);

	return centered_pc;
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

  std::cout << "IS OPEN: " << s.is_open() << std::endl;

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

#include "igl/signed_distance.h"
#include <igl/per_vertex_normals.h>
#include <igl/per_edge_normals.h>
#include <igl/per_face_normals.h>

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
	
	Eigen::VectorXd sd;
	Eigen::VectorXi idx;
	Eigen::MatrixXd norm, c;
	igl::AABB<Eigen::MatrixXd, 3> tree;
	Eigen::MatrixXd fn, vn, en; //note that _vn is the same as mesh's _normals. TODO
	Eigen::MatrixXi e;
	Eigen::VectorXi emap;
		
	tree.init(mesh.vertices, mesh.indices);

	igl::per_face_normals(mesh.vertices, mesh.indices, fn);
	igl::per_vertex_normals(mesh.vertices, mesh.indices, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, fn, vn);
	igl::per_edge_normals(mesh.vertices, mesh.indices, igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM, fn, en, e, emap);

	igl::signed_distance_pseudonormal(samplingPoints, mesh.vertices, mesh.indices, tree, fn, vn, en, emap, sd, idx, c, norm);

	std::vector<Eigen::Matrix<double,1,6>> remainingPoints;

	std::random_device rd{};
	std::mt19937 gen{ rd() };

	std::normal_distribution<> dx{ 0.0 , errorSigma };
	std::normal_distribution<> dy{ 0.0 , errorSigma };
	std::normal_distribution<> dz{ 0.0 , errorSigma };

	for (int i = 0; i < numSamplingPoints; i++)
	{
		Eigen::Vector3d noise = Eigen::Vector3d(dx(gen), dy(gen), dz(gen));
		Eigen::Matrix<double, 1, 6> p;
		
		p.row(0) << samplingPoints.row(i).leftCols(3), norm.row(i).leftCols(3);
		//p += noise;
		
		//double sd = node.signedDistance(samplingPoint);

		if (std::abs(sd(i)) < delta )//&& std::abs(sd) < delta)
		{
			remainingPoints.push_back(p);
		}
	}

	std::cout << "Sample points with error." << std::endl;

	PointCloud res;
	res.resize(remainingPoints.size(), 6);

	
	i = 0;
	for (const auto& point : remainingPoints)
	{	
		res.row(i) << point;

		i++;
	}

	return res;
}

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;

void lmu::projectPointCloudOnPlane(PointCloud & pc, const Eigen::Vector3d & p, const Eigen::Vector3d & n)
{
	K::Plane_3 plane(K::Point_3(p.x(), p.y(), p.z()), K::Vector_3(n.x(), n.y(), n.z()));

	for (int i = 0; i < pc.rows(); ++i)
	{
		K::Point_3 proj_pt = plane.projection(K::Point_3(pc.row(i).x(), pc.row(i).y(), pc.row(i).z()));
		pc.block<1, 3>(i, 0) << proj_pt.x(), proj_pt.y(), proj_pt.z();
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

lmu::NearestNeighborSearch::NearestNeighborSearch(const lmu::PointCloud& pc) :
		tree(create_point_tree(pc))
{		
}

Eigen::Vector3d lmu::NearestNeighborSearch::get_nn(const Eigen::Vector3d& p) const
{	
	Neighbor_search search(*tree, Kernel::Point_3(p.x(), p.y(), p.z()));

	for (auto it = search.begin(); it != search.end(); ++it)
	{
		auto found_p = it->first;
		return Eigen::Vector3d(found_p.x(), found_p.y(), found_p.z());
	}
	return Eigen::Vector3d(0, 0, 0);
}

double lmu::NearestNeighborSearch::get_nn_distance(const Eigen::Vector3d& p) const
{
	Neighbor_search search(*tree, Kernel::Point_3(p.x(), p.y(), p.z()));

	for (auto it = search.begin(); it != search.end(); ++it)
	{
		return it->second;
	}
	return std::numeric_limits<double>::max();
}

std::shared_ptr<lmu::PointTree> lmu::NearestNeighborSearch::create_point_tree(const lmu::PointCloud& pc) const
{
	std::vector<Kernel::Point_3> points;
	points.reserve(pc.rows());

	for (int i = 0; i < pc.rows(); ++i)
	{
		points.push_back(Kernel::Point_3(pc.coeff(i, 0), pc.coeff(i, 1), pc.coeff(i, 2)));
	}

	return std::make_shared<PointTree>(points.begin(), points.end());
}
