#include "pc_structure.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/IO/write_xyz_points.h>
#include <CGAL/property_map.h>
#include <CGAL/Shape_detection_3.h>
#include <CGAL/structure_point_set.h>

#include "boost/graph/graphviz.hpp"


// Type declarations
typedef CGAL::Exact_predicates_inexact_constructions_kernel  Kernel;
typedef Kernel::Point_3                                      Point;
typedef std::pair<Kernel::Point_3, Kernel::Vector_3>         Point_with_normal;
typedef std::vector<Point_with_normal>                       Pwn_vector;
typedef CGAL::First_of_pair_property_map<Point_with_normal>  Point_map;
typedef CGAL::Second_of_pair_property_map<Point_with_normal> Normal_map;
// Efficient RANSAC types
typedef CGAL::Shape_detection_3::Shape_detection_traits
<Kernel, Pwn_vector, Point_map, Normal_map>					 Traits;
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

	for (const auto& m : ms)
	{
		if (m->type != lmu::ManifoldType::Plane)
			continue;

		for (int i = 0; i < m->pc.rows(); ++i)
		{
			v.push_back(Point_with_normal(
				Kernel::Point_3(m->pc.coeff(i, 0), m->pc.coeff(i, 1), m->pc.coeff(i, 2)),
				Kernel::Vector_3(m->pc.coeff(i, 3), m->pc.coeff(i, 4), m->pc.coeff(i, 5)))
			);
		}
	}

	return v; 
}

lmu::PointCloud from_cgal_points(const Pwn_vector& points)
{
	lmu::PointCloud pc(points.size(), 6);

	for (int i = 0; i < points.size(); ++i)
	{
		const auto& p = points[i];
		pc.row(i) << p.first.x(), p.first.y(), p.first.z(), p.second.x(), p.second.y(), p.second.z();
	}

	return pc;
}

Efficient_ransac::Plane_range to_cgal_planes(const lmu::ManifoldSet& ms)
{
	auto planes = boost::make_shared<std::vector<boost::shared_ptr<Plane> > >();
	size_t point_offset = 0; 
	
	for (const auto& m : ms)
	{
		if (m->type != lmu::ManifoldType::Plane)
			continue;

		auto plane = boost::make_shared<PlaneEx>();

		std::vector<std::size_t> indices;
		for (int i = 0; i < m->pc.rows(); ++i)
			indices.push_back(i + point_offset);
		plane->set_indices(indices);
		
		plane->update(Traits::Plane_3(Kernel::Point_3(m->p.x(), m->p.y(), m->p.z()), Kernel::Vector_3(m->n.x(), m->n.y(), m->n.z())));
		
		planes->push_back(plane);

		point_offset += m->pc.rows();
	}

	return Efficient_ransac::Plane_range(planes);  
}

lmu::ManifoldSet get_plane_manifolds(const std::vector<Kernel::Plane_3>& planes, std::vector<std::pair<Kernel::Plane_3, lmu::ManifoldPtr>>& planes_to_manifolds)
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

lmu::PlaneGraph lmu::structure_pointcloud(const lmu::ManifoldSet& ms, double epsilon, lmu::PointCloud& debug_pc)
{
	Efficient_ransac::Plane_range planes = to_cgal_planes(ms);

	Pwn_vector points = to_cgal_points(ms);

	CGAL::Point_set_with_structure<Kernel> psws(points,
		planes,
		epsilon,
		CGAL::parameters::point_map(Point_map()).
		normal_map(Normal_map()).
		plane_map(CGAL::Shape_detection_3::Plane_map<Traits>()).
		plane_index_map(CGAL::Shape_detection_3::Point_to_shape_index_map<Traits>(points, planes)));


	lmu::PlaneGraph graph;
	std::vector<std::pair<Kernel::Plane_3, lmu::ManifoldPtr>> planes_to_manifolds;
	std::unordered_map<lmu::ManifoldPtr, std::vector<Eigen::Matrix<double, 1, 6>>> manifolds_to_points;

	std::vector<Eigen::Matrix<double, 1, 6>> struct_points;

	for (int i = 0; i < psws.size(); ++i)
	{
		std::vector<Kernel::Plane_3> adjacent_planes;

		psws.adjacency(i, std::back_inserter(adjacent_planes));

		if (!adjacent_planes.empty())
		{
			auto planes = get_plane_manifolds(adjacent_planes, planes_to_manifolds);

			for (const auto& plane : planes)
			{
				Eigen::Matrix<double, 1, 6> pn;
				pn << psws.point(i).x(), psws.point(i).y(), psws.point(i).z(), psws.normal(i).x(), psws.normal(i).y(), psws.normal(i).z();

				graph.add_plane(plane);
				manifolds_to_points[plane].push_back(pn);
			}

			for (int i = 0; i < planes.size(); ++i)
			{
				for (int j = i + 1; j < planes.size(); ++j)
				{
					if (!graph.is_connected(planes[i], planes[j]))
						graph.add_connection(planes[i], planes[j]);
				}
			}
		}

		if (!adjacent_planes.empty()) {
			Eigen::Matrix<double, 1, 6> struct_pt;

			switch (adjacent_planes.size())
			{
			case 0:
				struct_pt << psws.point(i).x(), psws.point(i).y(), psws.point(i).z(), 1.0, 0.0, 0.0;
				struct_points.push_back(struct_pt);
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
				std::cout << "lmu::structure_point_cloud(): number of adjacent planes different than: 0, 1, 2 or 3" << std::endl;
			}
		}
	}

	debug_pc = lmu::pointCloudFromVector(struct_points);

	// fill plane point clouds.
	for (const auto& plane : planes_to_manifolds)
	{
		plane.second->pc = pointCloudFromVector(manifolds_to_points[plane.second]);
		std::cout << "PTS: " << plane.second->pc.rows() << std::endl;
	}

	return graph;
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
