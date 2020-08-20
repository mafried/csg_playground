#include "point_vis.h"

#include "igl/signed_distance.h"
#include <igl/per_vertex_normals.h>
#include <igl/per_edge_normals.h>
#include <igl/per_face_normals.h>
#include "igl/copyleft/cgal/mesh_boolean.h"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/intersections.h>
#include <CGAL/squared_distance_3.h>

#include <fstream>

typedef CGAL::Simple_cartesian<double> K;
typedef CGAL::Polyhedron_3<K> Polyhedron;

typedef CGAL::AABB_face_graph_triangle_primitive<Polyhedron> Primitive;
typedef CGAL::AABB_traits<K, Primitive> Traits;
typedef CGAL::AABB_tree<Traits> Tree;

typedef CGAL::Search_traits_3<K> TreeTraits;
typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits> Neighbor_search;
typedef Neighbor_search::Tree PointTree;


std::vector<K::Plane_3> to_cgal_planes(const lmu::ManifoldSet& ms)
{
	std::vector<K::Plane_3> planes;
	planes.reserve(ms.size());

	for (const auto& m : ms)
	{
		if (m->type != lmu::ManifoldType::Plane)
			continue;

		planes.push_back(K::Plane_3(K::Point_3(m->p.x(), m->p.y(), m->p.z()), K::Vector_3(m->n.x(), m->n.y(), m->n.z())));
	}

	return planes;
}

std::vector<K::Point_3> to_cgal_points(const lmu::PointCloud& pc)
{
	std::vector<K::Point_3> points;
	points.reserve(pc.rows());

	for (int i = 0; i < pc.rows(); ++i)
	{
		points.push_back(K::Point_3(pc.coeff(i,0), pc.coeff(i, 1), pc.coeff(i, 2)));
	}

	return points;
}

class PolyhedronBuilder : public CGAL::Modifier_base<Polyhedron::HalfedgeDS> 
{
public:

	typedef typename Polyhedron::Vertex Vertex;
	typedef typename Polyhedron::Point Point;

	PolyhedronBuilder(const lmu::Mesh& m) : mesh(m)
	{
	}

	void operator()(Polyhedron::HalfedgeDS& poly)
	{
		CGAL::Polyhedron_incremental_builder_3<Polyhedron::HalfedgeDS> builder(poly, true);
		
		builder.begin_surface(mesh.vertices.rows(), mesh.indices.rows(), 0);
		
		double f = 1.0;

		for (int i = 0; i < mesh.vertices.rows(); ++i)
			builder.add_vertex(Point(mesh.vertices.row(i).x() * f, mesh.vertices.row(i).y() * f, mesh.vertices.row(i).z() * f));
		
		for (int i = 0; i < mesh.indices.rows(); ++i)
		{
			builder.begin_facet();

			builder.add_vertex_to_facet(mesh.indices.row(i).x());
			builder.add_vertex_to_facet(mesh.indices.row(i).y());
			builder.add_vertex_to_facet(mesh.indices.row(i).z());

			builder.end_facet();
		}
		
		builder.end_surface();
	}

	lmu::Mesh mesh;
};

Eigen::VectorXd get_signed_distances(const lmu::Mesh& surface_mesh, const lmu::PointCloud& points)
{
	Eigen::MatrixXd fn, vn, en;
	Eigen::MatrixXi e;
	Eigen::VectorXi emap;

	igl::AABB<Eigen::MatrixXd, 3> tree;
	tree.init(surface_mesh.vertices, surface_mesh.indices);

	igl::per_face_normals(surface_mesh.vertices, surface_mesh.indices, fn);
	igl::per_vertex_normals(surface_mesh.vertices, surface_mesh.indices, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, fn, vn);
	igl::per_edge_normals(surface_mesh.vertices, surface_mesh.indices, igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM, fn, en, e, emap);
		
	Eigen::VectorXd d;
	Eigen::VectorXi i;
	Eigen::MatrixXd norm, c;

	std::cout << "Get sd and normal. ";

	Eigen::MatrixXd pts(points.leftCols(3));

	igl::signed_distance_pseudonormal(pts, surface_mesh.vertices, surface_mesh.indices, tree, fn, vn, en, emap, d, i, c, norm);

	return d;
}

void lmu::write_affinity_matrix(const std::string & file, const Eigen::MatrixXd & af)
{
	std::ofstream f(file);

	f << af;
}

Eigen::MatrixXd lmu::get_affinity_matrix(const lmu::PointCloud & pc, const lmu::Mesh & surface_mesh, lmu::PointCloud& debug_pc)
{
	K::Plane_3 p;
	
	debug_pc = pc;

	Eigen::MatrixXd am(pc.rows(), pc.rows());
	
	Polyhedron poly;
	PolyhedronBuilder poly_builder(surface_mesh);
	poly.delegate(poly_builder);

	//CGAL::write_off(std::ofstream("mesh_poly.off"), poly);

	Tree tree(faces(poly).first, faces(poly).second, poly);
	
	int c = 0;

	am.setZero();

	Eigen::VectorXd hits(pc.rows());
	hits.setZero();

	Eigen::VectorXd sd = get_signed_distances(surface_mesh, pc);

	for (int i = 0; i < pc.rows(); ++i)
	{
		for (int j = i + 1; j < pc.rows(); ++j)
		{
			K::Point_3 p0(pc.row(i).x(), pc.row(i).y(), pc.row(i).z());
			K::Point_3 p1(pc.row(j).x(), pc.row(j).y(), pc.row(j).z());
			K::Segment_3 s(p0, p1);

			
			int close_inters = 0; 
			close_inters += sd.coeff(i, 0) >= 0.0 ? 1 : 0;
			close_inters += sd.coeff(j, 0) >= 0.0 ? 1 : 0;

			/*
			int n = tree.number_of_intersected_primitives(s);
			if (n == close_inters)
			{
				am.coeffRef(i, j) = 1.0;
				am.coeffRef(j, i) = 1.0;								
			}
			else
			{
				//std::cout << n << "(" << sd.row(i) << ") ";
				hits.coeffRef(i) += n;
			}
			*/

			double n = std::max(0.0, (double)tree.number_of_intersected_primitives(s) - close_inters);

			double v = n == 0.0 ? 1.0 : 1.0 / (1.0 + n);

			am.coeffRef(i, j) = v;
			am.coeffRef(j, i) = v;
			
			c++;

			if (c % 100000 == 0)
				std::cout << c << " of " << (int)(pc.rows() * pc.rows() * 0.5) << std::endl;
		}
	}

	hits.array() /= hits.maxCoeff();
	for (int i = 0; i < debug_pc.rows(); ++i)
	{
		debug_pc.coeffRef(i, 3) = hits.coeff(i, 0);
		debug_pc.coeffRef(i, 4) = hits.coeff(i, 0); 
		debug_pc.coeffRef(i, 5) = hits.coeff(i, 0);
	}

	return am;
}

Eigen::MatrixXd lmu::get_affinity_matrix(const lmu::PointCloud & pc, const lmu::ManifoldSet& p, double max_dist, bool normal_check, lmu::PointCloud& debug_pc)
{
	auto planes = to_cgal_planes(p);
	auto points = to_cgal_points(pc);

	Eigen::MatrixXd am(pc.rows(), pc.rows());
	am.setZero();

	PointTree tree(points.begin(), points.end());

	double epsilon = 0.000001;
	int c = 0;
	int wrong_side_c = 0;

	for (int i = 0; i < pc.rows(); ++i)
	{
		for (int j = i + 1; j < pc.rows(); ++j)
		{
			if (c % 100000 == 0)
				std::cout << c << " of " << (int)(pc.rows() * pc.rows()) * 0.5 << std::endl;
			c++;

			K::Point_3 p0(pc.row(i).x(), pc.row(i).y(), pc.row(i).z());
			K::Point_3 p1(pc.row(j).x(), pc.row(j).y(), pc.row(j).z());
			K::Segment_3 s(p0, p1);

			if (normal_check)
			{
				Eigen::Vector3d n(pc.row(i).rightCols(3));
				Eigen::Vector3d ep0(pc.row(i).leftCols(3));
				Eigen::Vector3d ep1(pc.row(j).leftCols(3));
				
				if ((n * -1.0).normalized().dot((ep1 - ep0).normalized()) < 0.0)
				{
					wrong_side_c++;
					continue;
				}
			}
			
			int hit = 1;
			for (const auto& plane : planes)
			{
				auto result = CGAL::intersection(s, plane);
				if (result)
				{
					if (const K::Point_3* p = boost::get<K::Point_3>(&*result))
					{
						// filter out points exactly on the origin or target plane.
						if (CGAL::squared_distance(*p, p0) < epsilon || CGAL::squared_distance(*p, p1) < epsilon)
							continue;

						Neighbor_search search(tree, *p, 1);

						for (auto it = search.begin(); it != search.end(); ++it) 
						{
							//std::cout << it->second << std::endl;

							if (it->second <= max_dist)
							{
								// If end point is on the plane, ignore it.
								// if (CGAL::squared_distance(plane, p1) < epsilon)
								// {
								//	continue;
								// }

								hit = 0;
								goto OUT;
							}
						}
					}
				}
			}
			OUT:
			am.coeffRef(i, j) = hit;
			am.coeffRef(j, i) = hit;			
		}
	}

	std::cout << "wrong side connections: " << wrong_side_c << std::endl;

	return am;
}

lmu::Mesh merge_meshes(const lmu::Mesh& m0, const lmu::Mesh& m1)
{
	auto new_m1 = m1;
   
	for (int i = 0; i < m0.vertices.rows(); ++i)
	{
		for (int j = 0; j < m1.vertices.rows(); ++j)
		{
			Eigen::Vector3d p0(m0.vertices.row(i).x(), m0.vertices.row(i).y(), m0.vertices.row(i).z());
			Eigen::Vector3d p1(m1.vertices.row(j).x(), m1.vertices.row(j).y(), m1.vertices.row(j).z());

			double d = (p0 - p1).norm();

			if (d < 0.0001)
			{
				new_m1.vertices.row(j) << p0.transpose();
			}
		}
	}

	lmu::Mesh m;
	igl::copyleft::cgal::mesh_boolean(m0.vertices, m0.indices, new_m1.vertices, new_m1.indices, igl::MESH_BOOLEAN_TYPE_UNION, m.vertices, m.indices);

	if (m.vertices.rows() < m0.vertices.rows() + m1.vertices.rows())
	{	
		return m;
	}
	else
	{
		return lmu::Mesh();
	}
}

Eigen::MatrixXd lmu::get_affinity_matrix(const lmu::Mesh& m0, const lmu::Mesh& m1)
{
	auto m = merge_meshes(m0, m1);
	
	Eigen::MatrixXd am(m.vertices.rows(), m.vertices.rows());

	if (m.empty())
		return am;

	am.setOnes();

	static int c = 0;
	//std::string path = "af_mesh_" + std::to_string(c++) + ".obj";
	//igl::writeOBJ(path, m.vertices, m.indices);


	for (int i = 0; i < m.vertices.rows(); ++i)
	{
		for (int j = i + 1; j < m.vertices.rows(); ++j)
		{
			K::Point_3 p0(m.vertices.row(i).x(), m.vertices.row(i).y(), m.vertices.row(i).z());
			K::Point_3 p1(m.vertices.row(j).x(), m.vertices.row(j).y(), m.vertices.row(j).z());
			K::Segment_3 s(p0, p1);
		
			int intersection_count = 0;
			for (int k = 0; k < m.indices.rows(); ++k)
			{
				int i0 = m.indices.row(k).x();
				int i1 = m.indices.row(k).y();
				int i2 = m.indices.row(k).z();
				
				K::Point_3 v0(m.vertices.row(i0).x(), m.vertices.row(i0).y(), m.vertices.row(i0).z());
				K::Point_3 v1(m.vertices.row(i1).x(), m.vertices.row(i1).y(), m.vertices.row(i1).z());
				K::Point_3 v2(m.vertices.row(i2).x(), m.vertices.row(i2).y(), m.vertices.row(i2).z());

				K::Triangle_3 t(v0, v1, v2);

				if (CGAL::coplanar(p0, v0, v1, v2) || CGAL::coplanar(p1, v0, v1, v2))
					continue;

				auto result = CGAL::intersection(s, t);
				if (result)
				{
					// We only consider intersection points (not also segments).
					if (boost::get<K::Point_3>(&*result))
					{
						intersection_count++;
					}
				}
			}			
			if (intersection_count > 0 )
			{
				am.coeffRef(i, j) = 0;
				am.coeffRef(j, i) = 0;
			}
		}
	}

	return am; 
}
