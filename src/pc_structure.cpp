#include "pc_structure.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/IO/write_xyz_points.h>
#include <CGAL/property_map.h>
#include <CGAL/Shape_detection_3.h>
#include <CGAL/structure_point_set.h>
#include <CGAL/grid_simplify_point_set.h>
#include <CGAL/bounding_box.h>
#include <CGAL/estimate_scale.h>
#include <CGAL/random_simplify_point_set.h>

#include "boost/graph/graphviz.hpp"


// Type declarations
typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef K::Point_3                                      Point;

typedef std::tuple<K::Point_3, K::Vector_3, int> Point_with_normal_and_plane_idx;


typedef std::vector<Point_with_normal_and_plane_idx>         Pwn_vector;
typedef CGAL::Nth_of_tuple_property_map<0, Point_with_normal_and_plane_idx>  Point_map;
typedef CGAL::Nth_of_tuple_property_map<1, Point_with_normal_and_plane_idx> Normal_map;
// Efficient RANSAC types
typedef CGAL::Shape_detection_3::Shape_detection_traits
<K, Pwn_vector, Point_map, Normal_map>					 Traits;
typedef CGAL::Shape_detection_3::Efficient_RANSAC<Traits>    Efficient_ransac;
typedef CGAL::Shape_detection_3::Plane<Traits>				 Plane;

struct PlaneEx : public Plane
{
	void set_indices(const std::vector<std::size_t>& indices)
	{
		m_indices = indices;
	}
};

Pwn_vector to_cgal_points(const lmu::ManifoldSet& ms)
{
	Pwn_vector v;

	int plane_idx = 0;
	for (const auto& m : ms)
	{
		if (m->type != lmu::ManifoldType::Plane)
			continue;

		for (int i = 0; i < m->pc.rows(); ++i)
		{
			v.push_back(Point_with_normal_and_plane_idx(
				K::Point_3(m->pc.coeff(i, 0), m->pc.coeff(i, 1), m->pc.coeff(i, 2)),
				K::Vector_3(m->pc.coeff(i, 3), m->pc.coeff(i, 4), m->pc.coeff(i, 5)),
				plane_idx)
			);
		}

		plane_idx++;
	}

	return v;
}

lmu::PointCloud from_cgal_points(const Pwn_vector& points)
{
	lmu::PointCloud pc(points.size(), 6);

	for (int i = 0; i < points.size(); ++i)
	{
		const auto& p = points[i];
		auto pos = std::get<0>(p);
		auto n = std::get<1>(p);

		pc.row(i) << pos.x(), pos.y(), pos.z(), n.x(), n.y(), n.z();
	}

	return pc;
}


K::Vector_3 findNormalOrientation(const Pwn_vector& points, const std::vector<std::size_t>& indices,
	const K::Vector_3& normal)
{
	int same_orientation = 0;
	int opposite_orientation = 0;

	for (std::size_t i = 0; i < indices.size(); ++i) {
		K::Vector_3 curr_normal = std::get<1>(points[indices[i]]);
		if (curr_normal*normal > 0) {
			same_orientation++;
		}
		else {
			opposite_orientation++;
		}
	}

	if (same_orientation > opposite_orientation)
		return normal;

	K::Vector_3 opposite_normal = K::Vector_3(-normal.x(), -normal.y(), -normal.z());
	return opposite_normal;
}


Efficient_ransac::Plane_range to_cgal_planes(const lmu::ManifoldSet& ms, const Pwn_vector& points)
{
	auto planes = boost::make_shared<std::vector<boost::shared_ptr<Plane> > >();

	for (int i = 0; i < ms.size(); ++i)
	{
		if (ms[i]->type != lmu::ManifoldType::Plane)
			continue;

		std::vector<std::size_t> indices;
		for (int j = 0; j < points.size(); ++j)
		{
			if (std::get<2>(points[j]) == i)
				indices.push_back(j);
		}

		if (indices.empty())
			continue;

		auto plane = boost::make_shared<PlaneEx>();

		plane->set_indices(indices);

		/*
		plane->update(Traits::Plane_3(
		Kernel::Point_3(ms[i]->p.x(), ms[i]->p.y(), ms[i]->p.z()),
		Kernel::Vector_3(ms[i]->n.x(), ms[i]->n.y(), ms[i]->n.z())));
		*/

		/*
		Kernel::Vector_3 normal = std::get<1>(points[indices[0]]);
		plane->update(Traits::Plane_3(
		Kernel::Point_3(ms[i]->p.x(), ms[i]->p.y(), ms[i]->p.z()),
		Kernel::Vector_3(normal.x(), normal.y(), normal.z())));
		*/

		K::Vector_3 normal = K::Vector_3(ms[i]->n.x(), ms[i]->n.y(), ms[i]->n.z());
		K::Vector_3 out_normal = findNormalOrientation(points, indices, normal);

		plane->update(Traits::Plane_3(
			K::Point_3(ms[i]->p.x(), ms[i]->p.y(), ms[i]->p.z()),
			out_normal));

		planes->push_back(plane);
	}

	return Efficient_ransac::Plane_range(planes);
}

lmu::ManifoldSet get_plane_manifolds(const std::vector<K::Plane_3>& planes, std::vector<std::pair<K::Plane_3, lmu::ManifoldPtr>>& planes_to_manifolds)
{
	lmu::ManifoldSet ms;

	for (const auto& p : planes)
	{
		bool plane_available = false;
		for (const auto& pm : planes_to_manifolds)
		{
			if (pm.first == p)
			{
				ms.push_back(pm.second);
				plane_available = true;
				break;
			}
		}

		if (!plane_available)
		{
			auto new_p = std::make_shared<lmu::Manifold>(
				lmu::ManifoldType::Plane,
				Eigen::Vector3d(p.point().x(), p.point().y(), p.point().z()),
				Eigen::Vector3d(p.orthogonal_vector().x(), p.orthogonal_vector().y(), p.orthogonal_vector().z()),
				Eigen::Vector3d(),
				lmu::PointCloud()
				);

			planes_to_manifolds.push_back(std::make_pair(p, new_p));
			ms.push_back(new_p);
		}
	}

	return ms;
}

std::pair<lmu::PointCloud, std::vector<int>> lmu::resample_pointcloud(const lmu::PlaneGraph& pg)
{
	std::vector<lmu::PointCloud> pcs;
	std::vector<int> pc_to_plane_idx;

	int plane_idx = 0;
	for (const auto& plane : pg.planes())
	{
		pcs.push_back(lmu::farthestPointSampling(plane->pc, 200));
		std::fill_n(std::back_inserter(pc_to_plane_idx), plane->pc.rows(), plane_idx);
		plane_idx++;
	}

	return std::make_pair(lmu::mergePointClouds(pcs), pc_to_plane_idx);
}

std::pair<lmu::PointCloud, std::vector<int>> lmu::resample_pointcloud(const lmu::PlaneGraph& pg, double range_scale_factor)
{
	int max_points = 3000;

	std::vector<Point_with_normal_and_plane_idx> points;
	int plane_idx = 0;
	for (const auto& plane : pg.planes())
	{
		for (int i = 0; i < plane->pc.rows(); ++i)
		{
			points.push_back(Point_with_normal_and_plane_idx(
				K::Point_3(plane->pc.coeff(i, 0), plane->pc.coeff(i, 1), plane->pc.coeff(i, 2)),
				K::Vector_3(plane->pc.coeff(i, 3), plane->pc.coeff(i, 4), plane->pc.coeff(i, 5)),
				plane_idx
			));
		}
		plane_idx++;
	}

	if (points.size() > max_points)
	{
		// Re-sample point cloud.
		std::vector<K::Point_3> pts;
		pts.reserve(points.size());
		std::transform(points.begin(), points.end(), std::back_inserter(pts), [](const auto& pn) {return std::get<0>(pn); });
		K::FT range_scale = CGAL::estimate_global_range_scale(pts);

		double removed_percentage = (double)(points.size() - max_points) / (double)points.size() * 100.0;

		std::cout << "Before: " << points.size() << " range scale: " << range_scale << " removed percentage: " << removed_percentage << std::endl;
		//points.erase(CGAL::grid_simplify_point_set(points, range_scale * range_scale_factor), points.end());
		points.erase(CGAL::random_simplify_point_set(points, removed_percentage), points.end());

		std::cout << "After: " << points.size() << std::endl;
	}

	lmu::PointCloud resampled_pc(points.size(), 6);
	std::vector<int> pc_to_plane_idx;
	pc_to_plane_idx.resize(points.size());
	for (int i = 0; i < points.size(); ++i)
	{
		auto pos = std::get<0>(points[i]);
		auto n = std::get<1>(points[i]);

		resampled_pc.row(i) <<
			pos.x(), pos.y(), pos.z(), n.x(), n.y(), n.z();

		pc_to_plane_idx[i] = std::get<2>(points[i]);
	}

	return std::make_pair(resampled_pc, pc_to_plane_idx);
}

lmu::PlaneGraph lmu::create_plane_graph(const lmu::ManifoldSet& ms, lmu::PointCloud& debug_pc,
	lmu::PointCloud& pcwn, double epsilon)
{
	Pwn_vector points = to_cgal_points(ms); //always the same number of points (tested).

	if (epsilon == 0.0)
	{
		// factor 2.0 was found out empirically.
		epsilon = 2.0 * CGAL::estimate_global_range_scale(points.begin(), points.end(), CGAL::Nth_of_tuple_property_map<0, Point_with_normal_and_plane_idx>());
		std::cout << "Using " << epsilon << " as epsilon." << std::endl;
	}

	Efficient_ransac::Plane_range planes = to_cgal_planes(ms, points);

	CGAL::Point_set_with_structure<K> psws(points,
		planes,
		epsilon,
		CGAL::parameters::point_map(Point_map()).
		normal_map(Normal_map()).
		plane_map(CGAL::Shape_detection_3::Plane_map<Traits>()).
		plane_index_map(CGAL::Shape_detection_3::Point_to_shape_index_map<Traits>(points, planes)));


	lmu::PlaneGraph graph;
	std::vector<std::pair<K::Plane_3, lmu::ManifoldPtr>> planes_to_manifolds;
	std::unordered_map<lmu::ManifoldPtr, std::vector<Eigen::Matrix<double, 1, 6>>> manifolds_to_points;

	std::vector<Eigen::Matrix<double, 1, 6>> struct_points;
	std::vector<Eigen::Matrix<double, 1, 6>> struct_points_wn;

	for (int i = 0; i < psws.size(); ++i)
	{
		std::vector<K::Plane_3> adjacent_planes;

		psws.adjacency(i, std::back_inserter(adjacent_planes));

		if (!adjacent_planes.empty())
		{
			auto planes = get_plane_manifolds(adjacent_planes, planes_to_manifolds);

			for (const auto& plane : planes)
			{
				Eigen::Matrix<double, 1, 6> pn;
				K::FT m = psws.normal(i).squared_length();
				m = sqrt(m);
				pn << psws.point(i).x(), psws.point(i).y(), psws.point(i).z(), psws.normal(i).x() / m, psws.normal(i).y() / m, psws.normal(i).z() / m;

				graph.add_plane(plane);
				manifolds_to_points[plane].push_back(pn);
			}

			for (int j = 0; j < planes.size(); ++j)
			{
				for (int k = j + 1; k < planes.size(); ++k)
				{
					if (!graph.is_connected(planes[j], planes[k]))
						graph.add_connection(planes[j], planes[k]);
				}
			}
		}

		if (!adjacent_planes.empty()) {
			Eigen::Matrix<double, 1, 6> struct_pt;
			Eigen::Matrix<double, 1, 6> struct_pt_wn;

			switch (adjacent_planes.size())
			{
			case 0:
				struct_pt << psws.point(i).x(), psws.point(i).y(), psws.point(i).z(), 1.0, 0.0, 0.0;
				struct_points.push_back(struct_pt);
				//std::cout << "Freeform at " << i << std::endl;
				break;
			case 1:
				struct_pt << psws.point(i).x(), psws.point(i).y(), psws.point(i).z(), 0.0, 0.0, 1.0;
				struct_points.push_back(struct_pt);
				break;
			case 2:
				struct_pt << psws.point(i).x(), psws.point(i).y(), psws.point(i).z(), 0.0, 1.0, 0.0;
				struct_points.push_back(struct_pt);
				break;
			case 3:
				struct_pt << psws.point(i).x(), psws.point(i).y(), psws.point(i).z(), 0.0, 0.0, 0.0;
				struct_points.push_back(struct_pt);
				break;

			default:
				std::cout << "Number of adjacent planes different than: 0, 1, 2 or 3 ("
					<< adjacent_planes.size() << ")." << std::endl;
			}

			K::FT m = psws.normal(i).squared_length();
			m = sqrt(m);
			struct_pt_wn << psws.point(i).x(), psws.point(i).y(), psws.point(i).z(),
				psws.normal(i).x() / m, psws.normal(i).y() / m, psws.normal(i).z() / m;

			struct_points_wn.push_back(struct_pt_wn);
		}
	}


	//debug_pc = lmu::pointCloudFromVector(struct_points);
	pcwn = lmu::pointCloudFromVector(struct_points_wn);


	// fill plane point clouds.

	int points_c = 0;
	for (const auto& plane : planes_to_manifolds)
	{
		plane.second->pc = pointCloudFromVector(manifolds_to_points[plane.second]);
		points_c += plane.second->pc.rows();
	}

	std::cout << "Planes: " << planes.size() << std::endl;
	std::cout << "Points after structuring (distributed among all planes): " << points_c << std::endl;

	return graph;
}

void lmu::resample_proportionally(const lmu::ManifoldSet& planes, int total_max_points)
{
	int points_c = 0;
	for (const auto& plane : planes)
		points_c += plane->pc.rows();

	for (const auto& plane : planes)
	{
		int num_points = std::round((double)plane->pc.rows() / (double)points_c * (double)total_max_points);
		plane->pc = lmu::farthestPointSampling(plane->pc, num_points);
	}
}

void lmu::PlaneGraph::add_plane(const ManifoldPtr& plane)
{
	if (vertex_map.find(plane) != vertex_map.end())
		return;

	auto v = boost::add_vertex(graph);
	graph[v] = plane;

	vertex_map[plane] = v;
}

void lmu::PlaneGraph::add_connection(const ManifoldPtr & p1, const ManifoldPtr & p2)
{
	auto v1_it = vertex_map.find(p1);
	auto v2_it = vertex_map.find(p2);

	if (v1_it == vertex_map.end() || v2_it == vertex_map.end())
		return;

	boost::add_edge(v1_it->second, v2_it->second, graph);
}

lmu::ManifoldSet lmu::PlaneGraph::planes() const
{
	ManifoldSet ms;
	for (const auto& m : vertex_map)
		ms.push_back(m.first);

	return ms;
}

lmu::PointCloud lmu::PlaneGraph::plane_points() const
{
	std::vector<PointCloud> pcs;
	const auto& p = planes();
	std::transform(p.begin(), p.end(), std::back_inserter(pcs),
		[](const ManifoldPtr m) {return m->pc; });

	return lmu::mergePointClouds(pcs);
}

template <class Name>
class VertexWriter
{
public:
	VertexWriter(Name _name) : name(_name) {}
	template <class VertexOrEdge>
	void operator()(std::ostream& out, const VertexOrEdge& v) const
	{
		out << "[label=\"" << lmu::manifoldTypeToString(name[v]->type) << "\"]";
	}
private:
	Name name;
};

void lmu::PlaneGraph::to_file(const std::string & file) const
{
	std::ofstream f(file);
	boost::write_graphviz(f, graph, VertexWriter<PlaneGraphStructure>(graph));
	f.close();
}

lmu::ManifoldSet lmu::PlaneGraph::connected(const ManifoldPtr & plane) const
{
	ManifoldSet neighbors;

	auto plane_it = vertex_map.find(plane);

	if (plane_it != vertex_map.end())
	{
		boost::graph_traits<PlaneGraphStructure>::adjacency_iterator neighbour, neighbour_end;

		for (boost::tie(neighbour, neighbour_end) = boost::adjacent_vertices(plane_it->second, graph); neighbour != neighbour_end; ++neighbour)
			neighbors.push_back(graph[*neighbour]);
	}

	return neighbors;
}

bool lmu::PlaneGraph::is_connected(const ManifoldPtr & p1, const ManifoldPtr & p2) const
{
	auto v1_it = vertex_map.find(p1);
	auto v2_it = vertex_map.find(p2);

	if (v1_it == vertex_map.end() || v2_it == vertex_map.end())
		return false;

	return boost::edge(v1_it->second, v2_it->second, graph).second;
}
