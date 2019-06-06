#include "c.h"
#include "primitives.h"
#include "pointcloud.h"
#include <iostream>

int ransac(const void * in_pc /*6D*/, int in_pc_rows, void* out_pc /*6D*/, void* out_pc_prim_ids /*1D*/ )
{
	const int pc_cols = 6;
	
	lmu::PointCloud pc(in_pc_rows, pc_cols);

	const double* in_pc_data = (double *)in_pc;
	for (int i = 0; i < in_pc_rows * pc_cols; ++i)
	{
		int row = i / pc_cols;
		int col = i % pc_cols;
		pc.block<1, 1>(row, col) << in_pc_data[i];
	}

	auto params = lmu::RansacParams();
	params.probability = 0.1;
	params.min_points = 500;
	params.normal_threshold = 0.9;
	params.cluster_epsilon = 0.2;
	params.epsilon = 0.2;
	
	auto ransacRes = lmu::extractManifoldsWithCGALRansac(pc, params);

	double* out_pc_data = (double *)out_pc;
	double* out_pc_prim_ids_data = (double *)out_pc_prim_ids;

	int pc_row_index = 0;
	int manifold_id = 0;
	for (const auto& m : ransacRes.manifolds)
	{
		for (int i = 0; i < m->pc.rows(); ++i)
		{
			out_pc_data[pc_row_index * pc_cols] = m->pc.row(i).col(0).value();
			out_pc_data[pc_row_index * pc_cols + 1] = m->pc.row(i).col(1).value();
			out_pc_data[pc_row_index * pc_cols + 2] = m->pc.row(i).col(2).value();
			out_pc_data[pc_row_index * pc_cols + 3] = m->pc.row(i).col(3).value();
			out_pc_data[pc_row_index * pc_cols + 4] = m->pc.row(i).col(4).value();
			out_pc_data[pc_row_index * pc_cols + 5] = m->pc.row(i).col(5).value();
			
			out_pc_prim_ids_data[pc_row_index] = manifold_id;
			pc_row_index++;
		}

		manifold_id++; 
	}

	return pc_row_index;
}
