#include "point_vis.h"

#include "igl/signed_distance.h"
#include <igl/per_vertex_normals.h>
#include <igl/per_edge_normals.h>
#include <igl/per_face_normals.h>
#include "igl/copyleft/cgal/mesh_boolean.h"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Triangulation_2.h>


#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Alpha_shape_vertex_base_2.h>
#include <CGAL/Alpha_shape_face_base_2.h>
#include <CGAL/Delaunay_triangulation_2.h>

#include <CGAL/Polyhedron_3.h>

#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/intersections.h>
#include <CGAL/squared_distance_3.h>
#include <CGAL/compute_average_spacing.h>

#include <fstream>

typedef CGAL::Simple_cartesian<double> K;
typedef CGAL::Polyhedron_3<K> Polyhedron;

typedef CGAL::AABB_face_graph_triangle_primitive<Polyhedron> Primitive;
typedef CGAL::AABB_traits<K, Primitive> Traits;
typedef CGAL::AABB_tree<Traits> Tree;

typedef CGAL::Triangulation_2<K> Triangulation;

typedef CGAL::Alpha_shape_vertex_base_2<K>                   Vb;
typedef CGAL::Alpha_shape_face_base_2<K>                     Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb>         Tds;
typedef CGAL::Delaunay_triangulation_2<K, Tds>               Triangulation_2;
typedef CGAL::Alpha_shape_2<Triangulation_2>                 Alpha_shape_2;

typedef Triangulation::Face_handle Face_handle;
//typedef Triangulation::Finite_face_handles Finite_face_handles;
typedef Triangulation::Vertex_handle Vertex_handle;

typedef std::vector<K::Triangle_3>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> TrianglePrimitive;
typedef CGAL::AABB_traits<K, TrianglePrimitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> TriangleTree;

typedef CGAL::Search_traits_3<K> TreeTraits;
typedef CGAL::Orthogonal_k_neighbor_search<TreeTraits> Neighbor_search;
typedef Neighbor_search::Tree PointTree;

std::vector<K::Point_2> get_2DPoints(const lmu::ManifoldPtr& plane) 
{
	K::Plane_3 cPlane(K::Point_3(plane->p.x(), plane->p.y(), plane->p.z()), K::Vector_3(plane->n.x(), plane->n.y(), plane->n.z()));

	std::vector<K::Point_2> points;
	points.reserve(plane->pc.rows());
	for (int i = 0; i < plane->pc.rows(); ++i)
	{
		Eigen::Vector3d p = plane->pc.row(i).leftCols(3).transpose();
		points.push_back(cPlane.to_2d(K::Point_3(p.x(), p.y(), p.z())));
	}

	return points;
}


K::Point_3 get_3DPoint(const lmu::ManifoldPtr& plane, const K::Point_2& point)
{
	K::Plane_3 cPlane(K::Point_3(plane->p.x(), plane->p.y(), plane->p.z()),
		K::Vector_3(plane->n.x(), plane->n.y(), plane->n.z()));

	K::Point_3 p = cPlane.to_3d(point);

	return p;
}

std::vector<K::Triangle_3> get_triangles(const lmu::ManifoldSet&ms, bool save_mesh = false)
{
	// For each manifold:
	//  extract its points, and the corresponding plane information
	//  given the points and the plane information, get 2D points
	//  given the 2d points on one plane, compute its Delaunay triangulation
	//  get the corresponding 3d points and add the 3d triangle to a list
	//
	// Create a triangle tree data-structure from the list of 3d triangles

	std::vector<K::Triangle_3> triangles;

	//std::vector<Eigen::Matrix<double, 1, 6>> pts;

	for (const auto& m : ms) 
	{
		if (m->type != lmu::ManifoldType::Plane) continue;

		std::vector<K::Point_2> points2d = get_2DPoints(m);

		/*
		for (const auto p : points2d)
		{
			Eigen::Matrix<double, 1, 6> pv;
			auto p3d = get_3DPoint(m, p);
			pv << p3d.x(), p3d.y(), p3d.z(), 0.0, 0.0, 0.0;
			pts.push_back(pv);
		}
		*/

		//Triangulation t;
		//t.insert(points2d.begin(), points2d.end());

		Alpha_shape_2 as(points2d.begin(), points2d.end(), 10000, Alpha_shape_2::Mode::GENERAL);

		Alpha_shape_2::Alpha_iterator opt = as.find_optimal_alpha(1);
		as.set_alpha(*opt);
	
		//for (auto fh = t.finite_faces_begin(); fh != t.finite_faces_end(); fh++) 
		for (auto fh = as.finite_faces_begin(); fh != as.finite_faces_end(); fh++)
		{
			switch (as.classify(fh))
			{
			case Alpha_shape_2::REGULAR:
			case Alpha_shape_2::SINGULAR:
			case Alpha_shape_2::INTERIOR:
			{
				auto v0 = fh->vertex(0);
				auto v1 = fh->vertex(1);
				auto v2 = fh->vertex(2);

				K::Point_2 p0 = v0->point();
				K::Point_2 p1 = v1->point();
				K::Point_2 p2 = v2->point();

				K::Point_3 p03d = get_3DPoint(m, p0);
				K::Point_3 p13d = get_3DPoint(m, p1);
				K::Point_3 p23d = get_3DPoint(m, p2);

				triangles.push_back(K::Triangle_3(p03d, p13d, p23d));

			}
			}
		}
	}

	//lmu::writePointCloud("proj_pts.txt", lmu::pointCloudFromVector(pts));

	lmu::Mesh m;

	m.vertices = Eigen::MatrixXd(triangles.size() * 3, 3);
	m.indices = Eigen::MatrixXi(triangles.size(), 3);
	
	lmu::PointCloud pc(m.vertices.rows(), 6);
	
	for (int i = 0; i < triangles.size(); ++i)
	{
		m.vertices.row(i * 3) << triangles[i].vertex(0).x(), triangles[i].vertex(0).y(), triangles[i].vertex(0).z();
		m.vertices.row(i * 3 + 1) << triangles[i].vertex(1).x(), triangles[i].vertex(1).y(), triangles[i].vertex(1).z();
		m.vertices.row(i * 3 + 2) << triangles[i].vertex(2).x(), triangles[i].vertex(2).y(), triangles[i].vertex(2).z();

		m.indices.row(i) << (i * 3), (i * 3 + 1), (i * 3 + 2);
	
		pc.row(i * 3) << triangles[i].vertex(0).x(), triangles[i].vertex(0).y(), triangles[i].vertex(0).z(), 0.0, 0.0, 0.0;
		pc.row(i * 3 + 1) << triangles[i].vertex(1).x(), triangles[i].vertex(1).y(), triangles[i].vertex(1).z(), 0.0, 0.0, 0.0;
		pc.row(i * 3 + 2) << triangles[i].vertex(2).x(), triangles[i].vertex(2).y(), triangles[i].vertex(2).z(), 0.0, 0.0, 0.0;	
	}
	
	/*
	std::vector<lmu::PointCloud> pointclouds;
	for (const auto& m : ms)
	{
		pointclouds.push_back(m->pc);
	}
	lmu::writePointCloud("plane_pc.txt", lmu::mergePointClouds(pointclouds));
	*/
	
	if (save_mesh)
	{
		lmu::writePointCloud("pc_tri.txt", pc);
		igl::writeOBJ("triangulation.obj", m.vertices, m.indices);
	}
	std::cout << "Triangles: " << triangles.size() << std::endl;

	return triangles;
}

std::shared_ptr<TriangleTree> get_triangle_tree(const lmu::ManifoldSet& planes, bool save_mesh = false)
{
	auto triangles = get_triangles(planes, save_mesh);

	return std::make_shared<TriangleTree>(triangles.begin(), triangles.end());
}

std::vector<K::Triangle_3> get_triangles(const lmu::ManifoldPtr& plane)
{
	lmu::ManifoldSet planes;
	planes.push_back(plane);

	return get_triangles(planes);
}

std::shared_ptr<TriangleTree> get_triangle_tree(const lmu::ManifoldPtr& plane)
{
	auto triangles = get_triangles(plane);

	return std::make_shared<TriangleTree>(triangles.begin(), triangles.end());
}

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

void lmu::write_affinity_matrix(const std::string & file, const Eigen::SparseMatrix<double> & af)
{
	std::ofstream f(file);

	for (int k = 0; k < af.outerSize(); ++k) 
	{
		for (Eigen::SparseMatrix<double>::InnerIterator it(af, k); it; ++it)
		{
			//if(it.value() != 0.0)
			f << it.row() << " " << it.col() << " " << it.value() << std::endl;
		}
	}
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

bool is_close(const K::Point_3& p, TriangleTree& tree, double epsilon) 
{
	double d = std::sqrt(tree.squared_distance(p));
	return d <= epsilon;
}

Eigen::SparseMatrix<double> lmu::get_affinity_matrix_with_triangulation(const lmu::PointCloud & pc,	const lmu::ManifoldSet& ms, 
	bool normal_check)
{
	auto planes = to_cgal_planes(ms);
	auto tree = get_triangle_tree(ms, true);

	std::vector<std::shared_ptr<TriangleTree>> per_plane_trees; 
	std::vector<std::vector<K::Triangle_3>> per_plane_triangles;
	std::vector<double> per_plane_avg_spacings;
	for (const auto& plane : ms)
	{
		per_plane_trees.push_back(get_triangle_tree(plane));
		per_plane_triangles.push_back(get_triangles(plane));
		per_plane_avg_spacings.push_back(CGAL::compute_average_spacing<CGAL::Sequential_tag>(to_cgal_points(plane->pc), 1));
		std::cout << "Plane spacing: " << per_plane_avg_spacings.back() << std::endl;
	}

	std::vector<Eigen::Triplet<double>> triplets;
	triplets.reserve(pc.rows()); //TODO: better estimation?
	Eigen::SparseMatrix<double> am(pc.rows(), pc.rows());

	double epsilon = 0.0001; // 0.000001;
	int c = 0;
	int wrong_side_c = 0;
	int point_vis_c = 0;
	int col_c = 0;

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
						
			Eigen::Vector3d en0(pc.row(i).rightCols(3));
			Eigen::Vector3d en1(pc.row(j).rightCols(3));

			Eigen::Vector3d ep0(pc.row(i).leftCols(3));
			Eigen::Vector3d ep1(pc.row(j).leftCols(3));

			if (en0.normalized().dot((ep1 - ep0).normalized()) > epsilon || en1.normalized().dot((ep0 - ep1).normalized()) > epsilon)
			{
				// not front-facing (thus not visible)
				wrong_side_c++;				
			}
			else
			{
				// Test against planes first.
				bool hit = false;
				for (int k = 0; k < planes.size(); ++k)
				{
					auto result = CGAL::intersection(s, planes[k]);
					if (result)
					{

						if (const K::Point_3* p = boost::get<K::Point_3>(&*result))
						{
							//double e = per_plane_avg_spacings[k];

							// filter out points exactly on the origin or target plane.
							if (CGAL::squared_distance(*p, p0) < epsilon ||
								CGAL::squared_distance(*p, p1) < epsilon)
							{
								continue;
							}
							else
							{							
								// Triangle list intersection test.
								for (int l = 0; l < per_plane_triangles[k].size(); ++l)
								{
									auto result = CGAL::intersection(s, per_plane_triangles[k][l]);
									if (result && boost::get<K::Point_3>(&*result))
									{
										col_c++;
										hit = true;
										goto HIT;
									}
								}
							}
						}
					}
				}
				HIT:
				if (!hit)
				{
					triplets.push_back(Eigen::Triplet<double>(i, j, 1));
					triplets.push_back(Eigen::Triplet<double>(j, i, 1));
					point_vis_c++;
				}

			}
		}
	}

	std::cout << "wrong side connections: " << wrong_side_c << std::endl;
	std::cout << "point visibilities: " << point_vis_c << std::endl;
	std::cout << "collisions: " << col_c << std::endl;

	am.setFromTriplets(triplets.begin(), triplets.end());

	return am;
}

struct PointTriangle
{
	PointTriangle(const K::Triangle_3& t, int point_idx) :
		triangle(t),
		point_idx(point_idx)
	{	
	}

	K::Triangle_3 triangle;
	int point_idx;
};

int get_global_point_idx(int plane_idx, int inner_plane_point_idx, const lmu::ManifoldSet& ms)
{
	int point_idx = 0;
	for (int i = 0; i < plane_idx; ++i)
		point_idx += ms[i]->pc.rows();

	return point_idx + inner_plane_point_idx;
}

std::vector<PointTriangle> get_splat_triangles(const lmu::ManifoldSet& planes, bool save_mesh = false)
{
	std::vector<PointTriangle> triangles;
	
	for (int i = 0; i < planes.size(); ++i)
	{
		auto plane = planes[i];

		if (plane->type != lmu::ManifoldType::Plane)
			continue;

		double splat_size = CGAL::compute_average_spacing<CGAL::Sequential_tag>(to_cgal_points(plane->pc), 10);
		double w = splat_size / 2.0;

		for (int j = 0; j < plane->pc.rows(); ++j)
		{
			K::Point_3  p0(plane->pc.coeff(j, 0), plane->pc.coeff(j, 1), plane->pc.coeff(j, 2));
			K::Vector_3 n0(plane->pc.coeff(j, 3), plane->pc.coeff(j, 4), plane->pc.coeff(j, 5));

			K::Plane_3 c_plane(p0, n0);

			auto c = c_plane.to_2d(p0);

			auto v0 = c_plane.to_3d(K::Point_2(c.x() - w, c.y() + w));
			auto v1 = c_plane.to_3d(K::Point_2(c.x() - w, c.y() - w));
			auto v2 = c_plane.to_3d(K::Point_2(c.x() + w, c.y() - w));
			auto v3 = c_plane.to_3d(K::Point_2(c.x() + w, c.y() + w));

			int point_idx = j;

			triangles.push_back(PointTriangle(K::Triangle_3(v0, v1, v2), point_idx));
			triangles.push_back(PointTriangle(K::Triangle_3(v0, v2, v3), point_idx));
		}
	}

	//lmu::writePointCloud("proj_pts.txt", lmu::pointCloudFromVector(pts));

	lmu::Mesh m;
	m.vertices = Eigen::MatrixXd(triangles.size() * 3, 3);
	m.indices = Eigen::MatrixXi(triangles.size(), 3);
	lmu::PointCloud pc(m.vertices.rows(), 6);

	for (int i = 0; i < triangles.size(); ++i)
	{
		m.vertices.row(i * 3) << triangles[i].triangle.vertex(0).x(), triangles[i].triangle.vertex(0).y(), triangles[i].triangle.vertex(0).z();
		m.vertices.row(i * 3 + 1) << triangles[i].triangle.vertex(1).x(), triangles[i].triangle.vertex(1).y(), triangles[i].triangle.vertex(1).z();
		m.vertices.row(i * 3 + 2) << triangles[i].triangle.vertex(2).x(), triangles[i].triangle.vertex(2).y(), triangles[i].triangle.vertex(2).z();

		m.indices.row(i) << (i * 3), (i * 3 + 1), (i * 3 + 2);

		pc.row(i * 3) << triangles[i].triangle.vertex(0).x(), triangles[i].triangle.vertex(0).y(), triangles[i].triangle.vertex(0).z(), 0.0, 0.0, 0.0;
		pc.row(i * 3 + 1) << triangles[i].triangle.vertex(1).x(), triangles[i].triangle.vertex(1).y(), triangles[i].triangle.vertex(1).z(), 0.0, 0.0, 0.0;
		pc.row(i * 3 + 2) << triangles[i].triangle.vertex(2).x(), triangles[i].triangle.vertex(2).y(), triangles[i].triangle.vertex(2).z(), 0.0, 0.0, 0.0;
	}

	/*
	std::vector<lmu::PointCloud> pointclouds;
	for (const auto& m : ms)
	{
		pointclouds.push_back(m->pc);
	}
	lmu::writePointCloud("plane_pc.txt", lmu::mergePointClouds(pointclouds));
	*/

	if (save_mesh)
	{
		lmu::writePointCloud("pc_tri.txt", pc);
		igl::writeOBJ("triangulation.obj", m.vertices, m.indices);
	}
	std::cout << "Triangles: " << triangles.size() << std::endl;

	return triangles;
}


std::vector<PointTriangle> get_splat_triangles(const lmu::ManifoldPtr& plane, bool save_mesh = false)
{
	lmu::ManifoldSet planes;
	planes.push_back(plane);

	return get_splat_triangles(planes, save_mesh);
}

Eigen::MatrixXd lmu::g_p1, lmu::g_p2, lmu::g_c;

Eigen::SparseMatrix<double> lmu::get_affinity_matrix_with_rays(const lmu::ManifoldSet& ms, bool normal_check)
{
	auto planes = to_cgal_planes(ms);

	get_splat_triangles(ms, true);

	std::vector<std::vector<PointTriangle>> per_plane_triangles;
	std::vector<double> per_plane_avg_spacings;
	int n = 0;
	for (const auto& plane : ms)
	{
		per_plane_triangles.push_back(get_splat_triangles(plane, false));
		per_plane_avg_spacings.push_back(CGAL::compute_average_spacing<CGAL::Sequential_tag>(to_cgal_points(plane->pc), 1));
		std::cout << "Plane spacing: " << per_plane_avg_spacings.back() << std::endl;

		n += plane->pc.rows();
	}
	
	std::vector<Eigen::Triplet<double>> triplets;
	Eigen::SparseMatrix<double> am(n, n);

	double epsilon = 0.0001; // 0.000001;
	int c = 0;
	int wrong_side_c = 0;
	int point_vis_c = 0;
	int col_c = 0;

	int max_rays = 30;
	double cone_angle = 60.0 / 180.0 * M_PI;
	double d = 0.1;
	double R = std::tan(cone_angle / 2.0) * d;

	std::random_device rd;  
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0.0, 1.0);

	g_p1 = Eigen::MatrixXd(max_rays, 3);
	g_p2 = Eigen::MatrixXd(max_rays, 3);
	g_c = Eigen::MatrixXd(max_rays, 3);
			
	for (int i = 0; i < ms.size(); ++i)
	{
		for (int j = 0; j < ms[i]->pc.rows(); ++j)
		{
			int start_point_idx = get_global_point_idx(i, j, ms);

			if (c % 100 == 0)
				std::cout << c << " of " << n << std::endl;
			c++;

			// A point is always visible to itself.
			triplets.push_back(Eigen::Triplet<double>(start_point_idx, start_point_idx, 1));

			K::Point_3  p0(ms[i]->pc.coeff(j, 0), ms[i]->pc.coeff(j, 1), ms[i]->pc.coeff(j, 2));
			K::Vector_3 n0(ms[i]->pc.coeff(j, 3), ms[i]->pc.coeff(j, 4), ms[i]->pc.coeff(j, 5));
			
			for (int k = 0; k < max_rays; k++)
			{				
				double r = R * std::sqrt(dis(gen));
				double theta = dis(gen) * 2 * M_PI;

				double x = r * std::cos(theta);
				double y = r * std::sin(theta);

				K::Vector_3 p_n = -1.0  * (n0 / CGAL::sqrt(n0.squared_length()));

				K::Point_3 p_origin_3 = p0 + d * p_n;
				K::Plane_3 plane(p_origin_3, p_n);
				K::Point_2 p_origin_2 = plane.to_2d(p_origin_3);

				K::Point_3 p1 = plane.to_3d(p_origin_2 + K::Vector_2(x, y));
			
				K::Ray_3 view_ray(p0, p1 - p0);

				g_p1.row(k) << p0.x(), p0.y(), p0.z();
				g_c.row(k) << 1.0, 0.0, 0.0;

				// The idea is: 
				// take the ray view_ray and see if it intersects with one of the planes.
				// If it intersects with one of the planes, it is checked if it intersects with one of the plane triangles (there are two triangles per point).
				// If it intersects with a triangle, the corresponding point (with index m/2) is marked as visible and first_hit is set to false.
				// If the very same ray hits another plane (and triangle), the corresponding point is marked as invisible. This is controlled by the first_hit variable.
				bool first_hit = true; 
				for (int l = 0; l < planes.size(); ++l)
				{
					auto result = CGAL::intersection(view_ray, planes[l]);
					if (result)
					{
						if (const K::Point_3* p = boost::get<K::Point_3>(&*result))
						{
							// filter out points exactly on the origin plane.
							if (CGAL::squared_distance(*p, p0) < epsilon)
							{
								continue;
							}
							else
							{
								// Triangle list intersection test.
								bool triangle_hit = false;
								for (int m = 0; m < per_plane_triangles[l].size(); ++m)
								{
									//std::cout << per_plane_triangles[l][m] << std::endl;

									auto result = CGAL::intersection(view_ray, per_plane_triangles[l][m].triangle);
									if (result)
									{
										if (const K::Point_3* pi = boost::get<K::Point_3>(&*result))
										{
											g_p2.row(k) << pi->x(), pi->y(), pi->z();
											triangle_hit = true; 

											int inter_point_idx = get_global_point_idx(l, per_plane_triangles[l][m].point_idx, ms);

											if (first_hit)
											{
												triplets.push_back(Eigen::Triplet<double>(start_point_idx, inter_point_idx, 1));
												triplets.push_back(Eigen::Triplet<double>(inter_point_idx, start_point_idx, 1));
											}
											else
											{
												triplets.push_back(Eigen::Triplet<double>(start_point_idx, inter_point_idx, 0));
												triplets.push_back(Eigen::Triplet<double>(inter_point_idx, start_point_idx, 0));
											}
										}
									}
								}
								if (triangle_hit) first_hit = false;
							}
						}
					}
				}
			}
		}
	}
	
	std::cout << "wrong side connections: " << wrong_side_c << std::endl;
	std::cout << "point visibilities: " << point_vis_c << std::endl;
	std::cout << "collisions: " << col_c << std::endl;

	am.setFromTriplets(triplets.begin(), triplets.end());

	
	for (int k = 0; k < am.outerSize(); ++k)
	{
		for (Eigen::SparseMatrix<double>::InnerIterator it(am, k); it; ++it)
		{
			if (it.value() != 0.0)
				it.valueRef() = 1.0;
		}
	}
	

	return am;
}

// -----

Eigen::SparseMatrix<double> lmu::get_affinity_matrix_with_rays_2(const lmu::ManifoldSet& ms, bool normal_check)
{
	auto planes = to_cgal_planes(ms);

	get_splat_triangles(ms, true);

	std::vector<std::vector<PointTriangle>> per_plane_triangles;
	std::vector<double> per_plane_avg_spacings;
	int n = 0;
	for (const auto& plane : ms) 
	{
		per_plane_triangles.push_back(get_splat_triangles(plane, false));
		per_plane_avg_spacings.push_back(CGAL::compute_average_spacing<CGAL::Sequential_tag>(to_cgal_points(plane->pc), 1));
		std::cout << "Plane spacing: " << per_plane_avg_spacings.back() << std::endl;

		n += plane->pc.rows();
	}

	std::vector<Eigen::Triplet<double>> triplets;
	Eigen::SparseMatrix<double> am(n, n);

	double epsilon = 0.0001; // 0.000001;
	int c = 0;
	int wrong_side_c = 0;
	int point_vis_c = 0;
	int col_c = 0;

	int max_rays = 50;
	double cone_angle = 60.0 / 180.0 * M_PI;
	double d = 0.1;
	double R = std::tan(cone_angle / 2.0) * d;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0.0, 1.0);

	g_p1 = Eigen::MatrixXd(max_rays, 3);
	g_p2 = Eigen::MatrixXd(max_rays, 3);
	g_c = Eigen::MatrixXd(max_rays, 3);

	for (int i = 0; i < ms.size(); ++i) 
	{
		for (int j = 0; j < ms[i]->pc.rows(); ++j)
		{

			int start_point_idx = get_global_point_idx(i, j, ms);

			// A point is visible by itself
			triplets.push_back(Eigen::Triplet<double>(start_point_idx, start_point_idx, 1));

			if (c % 100 == 0)
				std::cout << c << " of " << n << std::endl;

			c++;

			K::Point_3  p0(ms[i]->pc.coeff(j, 0), ms[i]->pc.coeff(j, 1), ms[i]->pc.coeff(j, 2));
			K::Vector_3 n0(ms[i]->pc.coeff(j, 3), ms[i]->pc.coeff(j, 4), ms[i]->pc.coeff(j, 5));

			for (int k = 0; k < max_rays; k++) {
				double r = R * std::sqrt(dis(gen));
				double theta = dis(gen) * 2 * M_PI;

				double x = r * std::cos(theta);
				double y = r * std::sin(theta);

				K::Vector_3 p_n = -1.0  * (n0 / CGAL::sqrt(n0.squared_length()));

				K::Point_3 p_origin_3 = p0 + d * p_n;
				K::Plane_3 plane(p_origin_3, p_n);
				K::Point_2 p_origin_2 = plane.to_2d(p_origin_3);

				K::Point_3 p1 = plane.to_3d(p_origin_2 + K::Vector_2(x, y));

				K::Ray_3 view_ray(p0, p1 - p0);

				g_p1.row(k) << p0.x(), p0.y(), p0.z();
				g_c.row(k) << 1.0, 0.0, 0.0;

				int closest_idx = -1;
				float closest_d_sq = std::numeric_limits<float>::max();

				std::vector<int> invisible_point_indices;

				for (int l = 0; l < planes.size(); ++l)
				{
					auto result = CGAL::intersection(view_ray, planes[l]);
					if (!result) continue;

					if (const K::Point_3* p = boost::get<K::Point_3>(&*result)) 
					{
						// filter out points exactly on the origin plane.
						if (CGAL::squared_distance(*p, p0) < epsilon)
							continue;

						// Triangle list intersection test.
						for (int m = 0; m < per_plane_triangles[l].size(); ++m) 
						{
							auto result = CGAL::intersection(view_ray, per_plane_triangles[l][m].triangle);
							if (!result) continue;

							if (const K::Point_3* pi = boost::get<K::Point_3>(&*result)) 
							{
								g_p2.row(k) << pi->x(), pi->y(), pi->z();

								int inter_point_idx = get_global_point_idx(l, per_plane_triangles[l][m].point_idx, ms);

								float d_sq = CGAL::squared_distance(p0, *pi);
								if (d_sq < closest_d_sq) 
								{
									closest_d_sq = d_sq;
									
									if (closest_idx != -1)
										invisible_point_indices.push_back(closest_idx);
									
									closest_idx = inter_point_idx;
								}
							}
						}
					}
				}

				if (closest_idx != -1)
				{
					triplets.push_back(Eigen::Triplet<double>(start_point_idx, closest_idx, 1));
					triplets.push_back(Eigen::Triplet<double>(closest_idx, start_point_idx, 1));

					for (int ipi : invisible_point_indices)
					{
						triplets.push_back(Eigen::Triplet<double>(start_point_idx, ipi, 0));
						triplets.push_back(Eigen::Triplet<double>(ipi, start_point_idx, 0));
					}
				}

			}
		}
	}

	am.setFromTriplets(triplets.begin(), triplets.end());

	for (int k = 0; k < am.outerSize(); ++k)
	{
		for (Eigen::SparseMatrix<double>::InnerIterator it(am, k); it; ++it)
		{
			if (it.value() != 0.0)
				it.valueRef() = 1.0;
		}
	}

	return am;
}

// -----



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

Eigen::SparseMatrix<double> lmu::get_affinity_matrix_old(const lmu::PointCloud & pc, const lmu::ManifoldSet& p, bool normal_check, lmu::PointCloud& debug_pc)
{
	auto planes = to_cgal_planes(p);
	auto points = to_cgal_points(pc);

	std::vector<Eigen::Triplet<double>> triplets;
	triplets.reserve(pc.rows()); //TODO: better estimation?
	Eigen::SparseMatrix<double> am(pc.rows(), pc.rows());

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

						Neighbor_search search(tree, *p);

						for (auto it = search.begin(); it != search.end(); ++it)
						{
							//std::cout << it->second << std::endl;

							if (it->second <= 0.01)
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
			if (hit == 1)
			{
				triplets.push_back(Eigen::Triplet<double>(i, j, 1));
				triplets.push_back(Eigen::Triplet<double>(j, i, 1));

			}
		}
	}

	std::cout << "wrong side connections: " << wrong_side_c << std::endl;

	am.setFromTriplets(triplets.begin(), triplets.end());

	return am;
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
