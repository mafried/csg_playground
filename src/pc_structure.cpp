#include "pc_structure.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/IO/write_xyz_points.h>
#include <CGAL/property_map.h>
#include <CGAL/Shape_detection_3.h>
#include <CGAL/structure_point_set.h>

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


void lmu::structure_pointcloud(const lmu::PointCloud& pc, const lmu::ManifoldSet& ms, double epsilon)
{
	Efficient_ransac::Plane_range planes = to_cgal_planes(ms);

	Pwn_vector points = to_cgal_points(ms);
	Pwn_vector structured_pts;

	CGAL::structure_point_set(points,
		planes,
		std::back_inserter(structured_pts),
		epsilon,//0.015, // epsilon for structuring points
		CGAL::parameters::point_map(Point_map()).
		normal_map(Normal_map()).
		plane_map(CGAL::Shape_detection_3::Plane_map<Traits>()).
		plane_index_map(CGAL::Shape_detection_3::Point_to_shape_index_map<Traits>(points, planes)));

	std::cout << structured_pts.size()
		<< " structured point(s) generated." << std::endl;

}

