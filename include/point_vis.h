#ifndef POINT_VIS_H
#define POINT_VIS_H

#include "mesh.h"
#include "primitives.h"

#include <Eigen/SparseCore>

namespace lmu
{
	void write_affinity_matrix(const std::string& file, const Eigen::MatrixXd& af);
	void write_affinity_matrix(const std::string& file, const Eigen::SparseMatrix<double>& af);

	Eigen::MatrixXd get_affinity_matrix(const lmu::PointCloud& pc, const lmu::Mesh& surface_mesh, lmu::PointCloud& debug_pc);
	
	Eigen::SparseMatrix<double> get_affinity_matrix(const lmu::PointCloud& pc, const lmu::ManifoldSet& planes, bool normal_check, lmu::PointCloud& debug_pc);

	Eigen::SparseMatrix<double> get_affinity_matrix_with_triangulation(const lmu::PointCloud& pc, const lmu::ManifoldSet& planes, bool normal_check);

	Eigen::SparseMatrix<double> get_affinity_matrix_old(const lmu::PointCloud& pc, const lmu::ManifoldSet& planes, bool normal_check, lmu::PointCloud& debug_pc);

	Eigen::MatrixXd get_affinity_matrix(const lmu::Mesh& m0, const lmu::Mesh& m1);

}

#endif 