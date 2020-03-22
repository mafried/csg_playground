#include "..\include\mesh.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <memory>
#include <algorithm>

#include <igl/readOBJ.h>
#include <igl/readOFF.h>
#include <igl/signed_distance.h>
#include <igl/upsample.h>

#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/organized_fast_mesh.h>
#include <pcl/surface/gp3.h>
#include <pcl/kdtree/kdtree_flann.h>

//#include <CGAL/Simple_cartesian.h>
//#include <CGAL/Advancing_front_surface_reconstruction.h>
//#include <CGAL/tuple.h>

#include <pcl/common/common.h>

#include "../include/constants.h"

#include <setoper.h>
#include <cdd.h>

#include "helper.h"

//typedef CGAL::Simple_cartesian<double> K;
//typedef K::Point_3  Point;
//typedef CGAL::cpp11::array<std::size_t, 3> Facet;

using namespace lmu;

void lmu::transform(Mesh& mesh)
{
	//TODO: faster https://stackoverflow.com/questions/38841606/shorter-way-to-apply-transform-to-matrix-containing-vectors-in-eigen 
	for (int i = 0; i < mesh.vertices.rows(); i++)
	{
		Eigen::Vector3d v = mesh.vertices.row(i);
		v = mesh.transform * v;
		mesh.vertices.row(i) = v;

	}
}

//
// Following code for geometry generation derived from: 
// 
// https://github.com/jjuiddong/Introduction-to-3D-Game-Programming-With-DirectX11/blob/master/Common/GeometryGenerator.cpp
// Under MIT License 
//

Mesh lmu::createBox(const Eigen::Affine3d& transform, const Eigen::Vector3d& size, int numSubdivisions)
{
	Mesh mesh; 

	double w2 = 0.5*size.x();
	double h2 = 0.5*size.y();
	double d2 = 0.5*size.z();

	mesh.vertices.resize(8, 3);

	mesh.vertices.row(0) = Eigen::RowVector3d(-w2, -h2, d2);
	mesh.vertices.row(1) = Eigen::RowVector3d(w2, -h2, d2);
	mesh.vertices.row(2) = Eigen::RowVector3d(w2, h2, d2);
	mesh.vertices.row(3) = Eigen::RowVector3d(-w2, h2, d2);

	mesh.vertices.row(4) = Eigen::RowVector3d(-w2, -h2, -d2);
	mesh.vertices.row(5) = Eigen::RowVector3d(w2, -h2, -d2);
	mesh.vertices.row(6) = Eigen::RowVector3d(w2, h2, -d2);
	mesh.vertices.row(7) = Eigen::RowVector3d(-w2, h2, -d2);

	mesh.indices.resize(12, 3);

	mesh.indices.row(0) = Eigen::RowVector3i(0, 1, 2);
	mesh.indices.row(1) = Eigen::RowVector3i(2, 3, 0);
	
	mesh.indices.row(2) = Eigen::RowVector3i(1, 5, 6);
	mesh.indices.row(3) = Eigen::RowVector3i(6, 2, 1);

	mesh.indices.row(4) = Eigen::RowVector3i(7, 6, 5);
	mesh.indices.row(5) = Eigen::RowVector3i(5, 4, 7);

	mesh.indices.row(6) = Eigen::RowVector3i(4, 0, 3);
	mesh.indices.row(7) = Eigen::RowVector3i(3, 7, 4);

	mesh.indices.row(8) = Eigen::RowVector3i(4, 5, 1);
	mesh.indices.row(9) = Eigen::RowVector3i(1, 0, 4);

	mesh.indices.row(10) = Eigen::RowVector3i(3, 2, 6);
	mesh.indices.row(11) = Eigen::RowVector3i(6, 7, 3);

	/*mesh.indices.row(2) = Eigen::RowVector3i(4, 5, 6);
	mesh.indices.row(3) = Eigen::RowVector3i(4, 6, 7);

	mesh.indices.row(4) = Eigen::RowVector3i(8, 9, 10);
	mesh.indices.row(5) = Eigen::RowVector3i(8, 10, 11);
	mesh.indices.row(6) = Eigen::RowVector3i(12, 13, 14);
	mesh.indices.row(7) = Eigen::RowVector3i(12, 14, 15);

	mesh.indices.row(8) = Eigen::RowVector3i(16, 17, 18);
	mesh.indices.row(9) = Eigen::RowVector3i(16, 18, 19);
	mesh.indices.row(10) = Eigen::RowVector3i(20, 21, 22);
	mesh.indices.row(11) = Eigen::RowVector3i( 20, 22, 23);
	*/

	mesh.transform = transform;

	//std::cout << "Before:" << std::endl;
	//std::cout << mesh.vertices << std::endl;

	lmu::transform(mesh);

	Mesh upsampledMesh;// = mesh;

	igl::upsample(mesh.vertices, mesh.indices, upsampledMesh.vertices, upsampledMesh.indices, numSubdivisions);

	//std::cout << "After:" << std::endl;
	//std::cout << mesh.vertices << std::endl;

	//Eigen::MatrixXd fn; 
	//igl::per_face_normals(upsampledMesh.vertices, upsampledMesh.indices, fn);
	igl::per_vertex_normals(upsampledMesh.vertices, upsampledMesh.indices, upsampledMesh.normals);

	return upsampledMesh;
}

Mesh meshFromGeometry(const std::vector<Eigen::RowVector3d>& vertices, const std::vector<int>& indices, const Eigen::Affine3d& transform)
{
	Mesh mesh;

	mesh.vertices.resize(vertices.size(), 3);

	int i = 0;
	for (const auto& vs : vertices)
	{
		mesh.vertices.row(i) = vs;
		i++;
	}

	mesh.indices.resize(indices.size() / 3, 3);

	for (int j = 0; j < indices.size() / 3; j++)
	{
		mesh.indices.row(j) = Eigen::RowVector3i(indices[j * 3], indices[j * 3 + 1], indices[j * 3 + 2]);
		//std::cout << "HERE";
	}

	mesh.transform = transform;

	lmu::transform(mesh);

	igl::per_vertex_normals(mesh.vertices, mesh.indices, mesh.normals);
	
	return mesh;
}

//http://richardssoftware.net/Home/Post/7
Mesh lmu::createSphere(const Eigen::Affine3d & transform, double radius, int stacks, int slices)
{
	int stackCount = stacks;
	int sliceCount = slices;

	const double pi = 3.14159265358979323846;

	double phiStep = pi / stackCount;
	double thetaStep = 2.0*pi / sliceCount;

	std::vector<Eigen::RowVector3d> verticesVector;
	std::vector<int> indicesVector;

	verticesVector.push_back(Eigen::RowVector3d(0, radius, 0));

	for (int i = 1; i <= stackCount - 1; i++) {
		double phi = i*phiStep;
		for (int j = 0; j <= sliceCount; j++) {
			double theta = j*thetaStep;
			Eigen::RowVector3d p(
				(radius*sin(phi)*cos(theta)),
				(radius*cos(phi)),
				(radius*sin(phi)*sin(theta))
			);

			verticesVector.push_back(p);
		}
	}
	
	verticesVector.push_back(Eigen::RowVector3d(0, -radius, 0));

	for (int i = 1; i <= sliceCount; i++) {
		indicesVector.push_back(0);
		indicesVector.push_back(i + 1);
		indicesVector.push_back(i);
	}

	int baseIndex = 1;
	int ringVertexCount = sliceCount + 1;
	for (int i = 0; i < stackCount - 2; i++) {
		for (int j = 0; j < sliceCount; j++) {
			indicesVector.push_back(baseIndex + i*ringVertexCount + j);
			indicesVector.push_back(baseIndex + i*ringVertexCount + j + 1);
			indicesVector.push_back(baseIndex + (i + 1)*ringVertexCount + j);

			indicesVector.push_back(baseIndex + (i + 1)*ringVertexCount + j);
			indicesVector.push_back(baseIndex + i*ringVertexCount + j + 1);
			indicesVector.push_back(baseIndex + (i + 1)*ringVertexCount + j + 1);
		}
	}

	size_t southPoleIndex = verticesVector.size() - 1;
	baseIndex = southPoleIndex - ringVertexCount;
	for (int i = 0; i < sliceCount; i++) {
		indicesVector.push_back(southPoleIndex);
		indicesVector.push_back(baseIndex + i);
		indicesVector.push_back(baseIndex + i + 1);
	}

	return meshFromGeometry(verticesVector, indicesVector, transform);
}

void buildCylinderTopCap(float topRadius, float height, int sliceCount, std::vector<Eigen::RowVector3d>& vertices, std::vector<int>& indices)
{
	const double pi = 3.14159265358979323846;

	int baseIndex = vertices.size();

	double y = 0.5*height;
	double dTheta = 2.0*pi / sliceCount;

	for (int i = 0; i <= sliceCount; i++)
	{
		double x = topRadius*cos(i*dTheta);
		double z = topRadius*sin(i*dTheta);

		//double u = x / height + 0.5;
		//double v = z / height + 0.5;
		vertices.push_back(Eigen::RowVector3d(x, y, z));
	}
	vertices.push_back(Eigen::RowVector3d(0, y, 0));
	int centerIndex = vertices.size() - 1;
	for (int i = 0; i < sliceCount; i++)
	{
		indices.push_back(centerIndex);
		indices.push_back(baseIndex + i + 1);
		indices.push_back(baseIndex + i);
	}
}

void buildCylinderBottomCap(float bottomRadius, float height, int sliceCount, std::vector<Eigen::RowVector3d>& vertices, std::vector<int>& indices)
{
	const double pi = 3.14159265358979323846;

	int baseIndex = vertices.size();

	double y = -0.5 * height;
	double dTheta = 2.0 * pi / sliceCount;

	for (int i = 0; i <= sliceCount; i++) {
		double x = bottomRadius * cos(i * dTheta);
		double z = bottomRadius * sin(i * dTheta);

		double u = x / height + 0.5;
		double v = z / height + 0.5;
		vertices.push_back(Eigen::RowVector3d(x, y, z));
	}
	vertices.push_back(Eigen::RowVector3d(0, y, 0));
	int centerIndex = vertices.size() - 1;
	for (int i = 0; i < sliceCount; i++) {
		indices.push_back(centerIndex);
		indices.push_back(baseIndex + i);
		indices.push_back(baseIndex + i + 1);
	}
}

Mesh lmu::createCylinder(const Eigen::Affine3d& transform, float bottomRadius, float topRadius, float height, int slices, int stacks)
{
	const double pi = 3.14159265358979323846;

	double stackHeight = height / stacks;
	double radiusStep = (topRadius - bottomRadius) / stacks;
	double ringCount = stacks + 1;

	std::vector<Eigen::RowVector3d> vertices;
	std::vector<int> indices;


	for (int i = 0; i < ringCount; i++) {
		double y = -0.5*height + i*stackHeight;
		double r = bottomRadius + i*radiusStep;
		double dTheta = 2.0*pi / slices;
		for (int j = 0; j <= slices; j++) 
		{
			double c = cos(j*dTheta);
			double s = sin(j*dTheta);

			Eigen::Vector3d v(r*c, y, r*s);
			//Eigen::Vector2d uv((double)j / sliceCount, 1.0 - (double)i / stackCount);
			//Eigen::Vector3d t(-s, 0.0, c);

			//double dr = bottomRadius - topRadius;
			//Eigen::Vector3d bitangent(dr*c, -height, dr*s);

			//auto n = t.cross(bitangent);
			//n.normalize();
			vertices.push_back(Eigen::RowVector3d(v));

		}
	}
	int ringVertexCount = slices + 1;
	for (int i = 0; i < stacks; i++) {
		for (int j = 0; j < slices; j++) {
			indices.push_back(i*ringVertexCount + j);
			indices.push_back((i + 1)*ringVertexCount + j);
			indices.push_back((i + 1)*ringVertexCount + j + 1);

			indices.push_back(i*ringVertexCount + j);
			indices.push_back((i + 1)*ringVertexCount + j + 1);
			indices.push_back(i*ringVertexCount + j + 1);
		}
	}

	buildCylinderTopCap(topRadius, height, slices, vertices, indices);
	buildCylinderBottomCap(bottomRadius, height, slices, vertices, indices);
	
	return meshFromGeometry(vertices, indices, transform);
}


#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/convex_hull_3.h>

#include <CGAL/license/Polyhedron.h>
#include <CGAL/boost/graph/named_function_params.h>
#include <CGAL/boost/graph/named_params_helper.h>

#include <CGAL/boost/graph/named_params_helper.h>
#include <CGAL/boost/graph/named_function_params.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Advancing_front_surface_reconstruction.h>
#include <CGAL/tuple.h>


#include <CGAL/Polyhedron_3.h>
#include <CGAL/IO/print_OFF.h>
#include <CGAL/IO/scan_OFF.h>
#include <CGAL/boost/graph/named_params_helper.h>
#include <CGAL/boost/graph/named_function_params.h>
#include <boost/graph/graph_traits.hpp>

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef CGAL::Polyhedron_3<K>                     Polyhedron_3;
typedef K::Point_3                                Point_3;
typedef K::Segment_3                              Segment_3;
typedef K::Triangle_3                             Triangle_3;
typedef K::Vector_3								  Vector_3;
typedef CGAL::cpp11::array<std::size_t, 3> Facet;

typedef CGAL::Advancing_front_surface_reconstruction<> Reconstruction;
typedef Reconstruction::Triangulation_3 Triangulation_3;
typedef Reconstruction::Triangulation_data_structure_2 TDS_2;

void lmu::initializePolytopeCreator()
{
	dd_set_global_constants();  /* First, this must be called to use cddlib. */
}

Mesh lmu::createPolytope(const Eigen::Affine3d& transform, const std::vector<Eigen::Vector3d>& p, const std::vector<Eigen::Vector3d>& n)
{	
	dd_PolyhedraPtr poly;
	dd_MatrixPtr A, G;
	dd_ErrorType err;
	
	A = dd_CreateMatrix(p.size(), 4);
	A->representation = dd_Inequality;

	for (int i = 0; i < p.size(); ++i)
	{
		double d = n[i].normalized().dot(p[i]);
		dd_set_d(A->matrix[i][0], d); dd_set_d(A->matrix[i][1], -n[i].x());  dd_set_d(A->matrix[i][2], -n[i].y());  dd_set_d(A->matrix[i][3], -n[i].z());
	}

   //dd_WriteMatrix(stdout, A);

	poly = dd_DDMatrix2Poly(A, &err);  /* compute the second (generator) representation */
	if (err != dd_NoError)
	{
		//std::cout << "Error: " << (int)err << std::endl;
		return Mesh();
	}
		
	G = dd_CopyGenerators(poly);
	if (G->rowsize == 0)
	{
		//std::cout << "Error: Row size is 0" << std::endl;
		return Mesh();
	}
	
	//dd_WriteMatrix(stdout, A);
	//dd_WriteMatrix(stdout, G);
	
	std::vector<Point_3> points;
	points.reserve(G->rowsize);
	for (int i = 0; i < G->rowsize; i++)
	{
		points.push_back(Point_3(dd_get_d(G->matrix[i][1]), dd_get_d(G->matrix[i][2]), dd_get_d(G->matrix[i][3])));
		//std::cout << "Poly point: " << points.back().x() << " " << points.back().y() << " " << points.back().z() << std::endl;
	}

	/*std::cout << "--------------------------------------------" << std::endl;
	if (points.size() < 8)
	{
		for (int i = 0; i < p.size(); ++i)
		{
			std::cout
				<< "p: " << p[i].x() << " " << p[i].y() << " " << p[i].z()
				<< " n: " << n[i].x() << " " << n[i].y() << " " << n[i].z() << std::endl;
		}
	}*/

	dd_FreeMatrix(A);
	dd_FreeMatrix(G);	
		
	//CGAL::advancing_front_surface_reconstruction(points.begin(),
	//	points.end(),
	//	std::back_inserter(facets));

	CGAL::Object obj;
	CGAL::convex_hull_3(points.begin(), points.end(), obj);	
	const Polyhedron_3* ph = CGAL::object_cast<Polyhedron_3>(&obj);
	if (!ph) 
	{
		//std::cout << "Error: Polyhedron object is null." << std::endl;
		return Mesh();
	}
	
	Eigen::MatrixXd verts(ph->size_of_vertices(), 3);
	auto np = CGAL::parameters::all_default();
	auto vpm = choose_param(get_param(np, CGAL::internal_np::vertex_point),
		CGAL::get_const_property_map(CGAL::vertex_point, *ph));
	size_t vertexIdx = 0;
	for (auto vi : vertices(*ph)) {
		double x = CGAL::to_double(get(vpm, vi).x());
		double y = CGAL::to_double(get(vpm, vi).y());
		double z = CGAL::to_double(get(vpm, vi).z());
		verts.row(vertexIdx++) << x, y, z;
	}

	Eigen::MatrixXi indices(ph->size_of_facets(), 3);
	CGAL::Inverse_index<Polyhedron_3::Vertex_const_iterator>  index(ph->vertices_begin(), ph->vertices_end());
	size_t faceIdx = 0;
	for (auto fi = ph->facets_begin(); fi != ph->facets_end(); ++fi) {

		auto hc = fi->facet_begin();
		auto hc_end = hc;
		std::size_t n = circulator_size(hc);
		size_t indexIdx = 0;
		do {
			indices.block<1,1>(faceIdx, indexIdx) << index[hc->vertex()];
			indexIdx++;
			++hc;
		} while (hc != hc_end);
		faceIdx++;
	}

	/*Eigen::MatrixXi indices(sm.number_of_faces(), 3);
	int row = 0;
	for (Surface_mesh::Face_index fi : sm.faces()) 
	{
		Surface_mesh::Halfedge_index hf = sm.halfedge(fi);
		unsigned int indicesRow[3];
		int i = 0;
		for (Surface_mesh::Vertex_index vi : vertices_around_face(hf, sm))
		{
			if (i > 2) break;
			indicesRow[i++] = (unsigned int)vi;
		}
		if (i < 3) break;
		indices.row(row++) << indicesRow[0], indicesRow[1], indicesRow[2];
	}

	Eigen::MatrixXd vertices(sm.number_of_vertices(), 3);
	row = 0;
	for (auto vd : sm.vertices())
	{
		auto p = sm.point(vd);
		vertices.row(row++) << p.x(), p.y(), p.z();
	}

	std::cout << "mesh ready" << std::endl;

	return Mesh(vertices, indices);*/

	return Mesh(verts, indices);
}

#include <igl/writeOBJ.h>

Mesh lmu::createFromPointCloud(const PointCloud & pc)
{
	std::vector<Facet> facets;

	std::vector<Point_3> points;
	points.reserve(pc.rows());
	for (int i = 0; i < pc.rows(); ++i)
	{
		Eigen::RowVector3d p = pc.row(i).leftCols(3);
		points.push_back(Point_3(p.x(), p.y(), p.z()));
	}
	CGAL::advancing_front_surface_reconstruction(points.begin(), points.end(),std::back_inserter(facets));
	
	Eigen::MatrixXd vertices(points.size(), 3);
	for (int i = 0; i < points.size(); ++i)
		vertices.row(i) << points[i].x(), points[i].y(), points[i].z();

	std::cout << "Facets: " << facets.size() << std::endl;
	Eigen::MatrixXi indices(facets.size(), 3);
	for (int i = 0; i < facets.size(); ++i)
		indices.row(i) << facets[i][0], facets[i][1], facets[i][2];

	
	igl::writeOBJ("mesh_out.obj", vertices, indices);

	
	return Mesh(vertices, indices);
}

double lmu::computeMeshArea(const Mesh & m)
{
	double area = 0.0;
	for (int i = 0; i < m.indices.rows(); ++i)
	{
		Eigen::RowVector3d v0 = m.vertices.row(m.indices.row(i).x());
		Eigen::RowVector3d v1 = m.vertices.row(m.indices.row(i).y());
		Eigen::RowVector3d v2 = m.vertices.row(m.indices.row(i).z());

		Triangle_3 tr(Point_3(v0.x(), v0.y(), v0.z()), Point_3(v1.x(), v1.y(), v1.z()), Point_3(v2.x(), v2.y(), v2.z()));

		//std::cout << tr << std::endl;

		area += std::sqrt(tr.squared_area());
	}

	return area;
}

Mesh lmu::fromOBJFile(const std::string & file)
{
	Eigen::MatrixXd vertices;
	Eigen::MatrixXi indices;
	
	Mesh mesh;

	igl::readOBJ(file, vertices, indices);
	
	return Mesh(vertices, indices);
}

Mesh lmu::fromOFFFile(const std::string & file)
{
	Eigen::MatrixXd vertices;
	Eigen::MatrixXi indices;

	Mesh mesh;

	if (!igl::readOFF(file, vertices, indices))
	{
		std::cout << "Could not read off file '" << file << "'." << std::endl;
		return Mesh();
	}
	return Mesh(vertices, indices);
}

//Mesh must not have any transform other than identity.
void lmu::scaleMesh(Mesh& mesh, double largestDim)
{
	double factor = (mesh.vertices.colwise().maxCoeff() - mesh.vertices.colwise().minCoeff()).cwiseAbs().maxCoeff();
	
	mesh.vertices = mesh.vertices / factor * largestDim;
}

lmu::Mesh lmu::to_canonical_frame(const Mesh& m)
{
	Eigen::Vector3d min = m.vertices.colwise().minCoeff();
	Eigen::Vector3d max = m.vertices.colwise().maxCoeff();

	lmu::Mesh centered_m = m;

	centered_m.vertices  = centered_m.vertices.rowwise() - min.transpose();
	centered_m.vertices = centered_m.vertices.array().rowwise() / (max - min).transpose().array();

	return centered_m;
}

std::string lmu::iFTypeToString(ImplicitFunctionType type)
{
	switch (type)
	{
	case ImplicitFunctionType::Sphere:
		return "Sphere";
	case ImplicitFunctionType::Cylinder:
		return "Cylinder";
	case ImplicitFunctionType::Box:
		return "Box";
	case ImplicitFunctionType::Polytope:
		return "Polytope";
	case ImplicitFunctionType::Cone:
		return "Cone";
	case ImplicitFunctionType::Torus:
		return "Torus";
	case ImplicitFunctionType::Plane:
		return "Plane";
	case ImplicitFunctionType::Null:
		return "Null";
	default:
		return "Undefined Type";
	}
}

Eigen::Matrix3d rotationMatrixFrom(const Eigen::Vector3d& x, Eigen::Vector3d& y)
{	
	Eigen::Matrix3d m;
	m.col(0) = x.normalized();
	m.col(2) = x.cross(y).normalized();
	m.col(1) = m.col(2).cross(x).normalized();

	return m; 
}

std::vector<std::shared_ptr<ImplicitFunction>> lmu::fromFile(const std::string & file, double scaling)
{
	std::ifstream s(file);
	
	int cylCount = 0; 
	int boxCount = 0;
	std::vector<std::shared_ptr<ImplicitFunction>> res;

	while (!s.eof())
	{
		std::string type;
		s >> type; 
		
		std::cout << "Type: " << type << std::endl;

		if (type == "cylinder")
		{		
			double x, y, z, ax, ay, az, radius, height;
			s >> ax >> ay >> az >> x >> y >> z >> radius >> height;

			x *= scaling;
			y *= scaling;
			z *= scaling;
			radius *= scaling; 
			height *= scaling;

			Eigen::Translation3d t(x, z + height / 2, y  );
			Eigen::AngleAxisd r(M_PI / 2 , Eigen::Vector3d(1,0,0));
			//Eigen::Matrix3d r = rotationMatrixFrom(Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(ax,ay,az));

			//std::vector<double> params = { x,y,z,ax,ay,az,radius, height };

			res.push_back(std::make_shared<IFCylinder>((Eigen::Affine3d)r*t,radius, height, "Cylinder" + std::to_string(cylCount++)));
		}
		else if (type == "box")
		{
			double xmin, ymin, zmin, xmax, ymax, zmax;
			s >> xmin >> xmax >> ymin >> ymax >> zmin >> zmax;

			xmin *= scaling;
			ymin *= scaling;
			zmin *= scaling;
			xmax *= scaling;
			ymax *= scaling;
			zmax *= scaling;

			Eigen::Translation3d t(xmin + (xmax-xmin) * 0.5, ymin + (ymax - ymin) * 0.5, zmin + (zmax - zmin) * 0.5);
					
			std::cout << "BOX: " << xmin + (xmax - xmin) * 0.5 << " " << ymin + (ymax - ymin) * 0.5 << " " << zmin + (zmax - zmin) * 0.5 << std::endl;
			
			std::cout << ymax << " min: " << ymin << std::endl;
			//std::cout << xmin << ymin << zmin << "Max: " << xmax << ymax << zmax << std::endl;

			std::vector<double> params = { xmin, ymin, zmin, xmax, ymax, zmax };

			res.push_back(std::make_shared<IFBox>((Eigen::Affine3d)t, Eigen::Vector3d(xmax-xmin, ymax-ymin, zmax-zmin), 1,"Box" + std::to_string(boxCount++)));
		}
	}

	return res;
}


std::vector<std::shared_ptr<ImplicitFunction>> lmu::fromFilePRIM(const std::string& file) 
{
  std::ifstream s(file);
  
  std::vector<std::shared_ptr<ImplicitFunction>> res;

  while (!s.eof()) {
    std::string name;
    s >> name; 
    
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    std::cout << "Name: " << name << std::endl;
    
    if (name.find("cylinder") != std::string::npos) {
      // transform radius height
      Eigen::Affine3d t;
      // row 1
      s >> t(0,0) >> t(0,1) >> t(0,2) >> t(0,3);
      // row 2
      s >> t(1,0) >> t(1,1) >> t(1,2) >> t(1,3);
      // row 3
      s >> t(2,0) >> t(2,1) >> t(2,2) >> t(2,3);
      // row 4
      s >> t(3,0) >> t(3,1) >> t(3,2) >> t(3,3);

      double radius;
      s >> radius;

      double height;
      s >> height;

      res.push_back(std::make_shared<IFCylinder>(t, radius, height, name));

    } else if (name.find("sphere") != std::string::npos) {
      // transform radius displacement
      Eigen::Affine3d t;
      // row 1
      s >> t(0,0) >> t(0,1) >> t(0,2) >> t(0,3);
      // row 2
      s >> t(1,0) >> t(1,1) >> t(1,2) >> t(1,3);
      // row 3
      s >> t(2,0) >> t(2,1) >> t(2,2) >> t(2,3);
      // row 4
      s >> t(3,0) >> t(3,1) >> t(3,2) >> t(3,3);

      double radius;
      s >> radius;

      double disp;
      s >> disp;

      res.push_back(std::make_shared<IFSphere>(t, radius, name, disp));

    } else if (name.find("box") != std::string::npos || name.find("cube") != std::string::npos) {
      // transform size displacement
      Eigen::Affine3d t;
      // row 1
      s >> t(0,0) >> t(0,1) >> t(0,2) >> t(0,3);
      // row 2
      s >> t(1,0) >> t(1,1) >> t(1,2) >> t(1,3);
      // row 3
      s >> t(2,0) >> t(2,1) >> t(2,2) >> t(2,3);
      // row 4
      s >> t(3,0) >> t(3,1) >> t(3,2) >> t(3,3);

      Eigen::Vector3d size;
      s >> size[0] >> size[1] >> size[2];

      double disp;
      s >> disp;

      res.push_back(std::make_shared<IFBox>(t, size, 1, name, disp));

    } else if (name.find("cone") != std::string::npos) {
      // transform c
      Eigen::Affine3d t;
      // row 1
      s >> t(0,0) >> t(0,1) >> t(0,2) >> t(0,3);
      // row 2
      s >> t(1,0) >> t(1,1) >> t(1,2) >> t(1,3);
      // row 3
      s >> t(2,0) >> t(2,1) >> t(2,2) >> t(2,3);
      // row 4
      s >> t(3,0) >> t(3,1) >> t(3,2) >> t(3,3);

      Eigen::Vector3d c;
      s >> c[0] >> c[1] >> c[2];

	  //TODO
      //res.push_back(std::make_shared<IFCone>(t, c, name));

    }
  }		
  
  return res;
}


void lmu::movePointsToSurface(const std::vector<std::shared_ptr<ImplicitFunction>>& functions, bool filter, double threshold)
{	
	for (auto& func : functions)
	{
		std::vector<Eigen::Matrix<double, 1, 6>> points;

		for (int j = 0; j < func->pointsCRef().rows(); ++j)
		{
			Eigen::Matrix<double, 1, 6> pn = func->pointsCRef().row(j);
			Eigen::Vector3d sampleP = pn.leftCols(3);
			Eigen::Vector3d sampleN = pn.rightCols(3);

			Eigen::Vector4d sampleDistGradFunction = func->signedDistanceAndGradient(sampleP);

			double sampleDistFunction = sampleDistGradFunction[0];
			Eigen::Vector3d sampleGradFunction = sampleDistGradFunction.bottomRows(3);
			
			Eigen::Matrix<double, 1, 6> newPN;
			Eigen::Vector3d newP = (sampleP - (sampleDistFunction * sampleGradFunction));
			newPN << newP.transpose() , sampleN.transpose();
								
			double distAfter = func->signedDistance(newP);

			if (std::abs(distAfter) <  threshold)
			{				
				points.push_back(newPN);
			}
			else
			{
				//std::cout << func->name() << ": filter " << distAfter << std::endl;

				//func->points().row(j) = newPN;
			}
		}

		if (filter)
		{
			Eigen::MatrixXd m(points.size(), 6); 

			int i = 0;
			for (const auto& point : points)
			{
				m.row(i) = point;				
				i++;
			}				
			func->points() = m;
		}
	}
}



void lmu::writePrimitives(const std::string& filename, 
			  const std::vector<std::shared_ptr<ImplicitFunction>>& shapes)
{
  std::ofstream of(filename.c_str());
  
  for (const auto& shape : shapes) {
    of << shape->name(); 
    of << " " + shape->serializeTransform(); 
    of << " " + shape->serializeParameters();
    of << std::endl;
  }
}

lmu::IFPolytope::IFPolytope(const Eigen::Affine3d & transform, const std::vector<Eigen::Vector3d>& p, const std::vector<Eigen::Vector3d>& n, const std::string & name) :
	ImplicitFunction(transform, createPolytope(transform, p, n), name),
	_n(n),
	_p(p)
{
	// Make sure normal vectors are normalized.
	for (auto& nv : _n)
		nv.normalize();

	// Create AABB tree for fast signed distance calculations. 
	if (!_mesh.empty())
	{
		_tree.init(_mesh.vertices, _mesh.indices);
		_hier.set_mesh(_mesh.vertices, _mesh.indices);
		_hier.grow();
	}
}

ImplicitFunctionType lmu::IFPolytope::type() const
{
	return ImplicitFunctionType::Polytope;
}

std::shared_ptr<ImplicitFunction> lmu::IFPolytope::clone() const
{
	return std::make_shared<IFPolytope>(*this);
}

std::string lmu::IFPolytope::serializeParameters() const
{
	return "";
}

bool lmu::IFPolytope::empty() const
{
	return _mesh.empty();
}

Mesh lmu::IFPolytope::createMesh() const
{
	return _mesh;//createPolytope(_transform, _p, _n);
}

Eigen::Vector3d lmu::IFPolytope::gradientLocal(const Eigen::Vector3d & localP, double h)
{
	double dx = (signedDistanceLocal(Eigen::Vector3d(localP.x() + h, localP.y(), localP.z())) - signedDistanceLocal(Eigen::Vector3d(localP.x() - h, localP.y(), localP.z()))) / (2.0 * h);
	double dy = (signedDistanceLocal(Eigen::Vector3d(localP.x(), localP.y() + h, localP.z())) - signedDistanceLocal(Eigen::Vector3d(localP.x(), localP.y() - h, localP.z()))) / (2.0 * h);
	double dz = (signedDistanceLocal(Eigen::Vector3d(localP.x(), localP.y(), localP.z() + h)) - signedDistanceLocal(Eigen::Vector3d(localP.x(), localP.y(), localP.z() - h))) / (2.0 * h);

	return Eigen::Vector3d(dx, dy, dz);
}

double lmu::IFPolytope::signedDistanceLocal(const Eigen::Vector3d & localP)
{	
	double d = std::numeric_limits<double>::max(); 

	for (int i = 0; i < _p.size(); ++i)
	{
		double pn = -_n[i].dot(localP - _p[i]);
		d = pn < d ? pn : d;
	}
	
	return -d;
}


inline lmu::IFCylinder::IFCylinder(const Eigen::Affine3d& transform, double radius, double height, const std::string & name) :
	ImplicitFunction(transform, createCylinder(transform, radius, radius, height, 200, 200), name),
	_radius(radius),
	_height(height)
{
	// Original (pre-transform) vertices of the AABB
	Eigen::Vector3d v1(-radius, -height / 2, -radius);
	Eigen::Vector3d v2(radius, -height / 2, -radius);
	Eigen::Vector3d v3(-radius, -height / 2, radius);
	Eigen::Vector3d v4(radius, -height / 2, radius);
	Eigen::Vector3d v5(-radius, height / 2, -radius);
	Eigen::Vector3d v6(radius, height / 2, -radius);
	Eigen::Vector3d v7(-radius, height / 2, radius);
	Eigen::Vector3d v8(radius, height / 2, radius);

	// Transformed vertices of the AABB
	Eigen::Vector3d tv1 = _transform * v1;
	Eigen::Vector3d tv2 = _transform * v2;
	Eigen::Vector3d tv3 = _transform * v3;
	Eigen::Vector3d tv4 = _transform * v4;
	Eigen::Vector3d tv5 = _transform * v5;
	Eigen::Vector3d tv6 = _transform * v6;
	Eigen::Vector3d tv7 = _transform * v7;
	Eigen::Vector3d tv8 = _transform * v8;

	Eigen::Vector3d min = tv1;
	Eigen::Vector3d max = tv1;

	// Recompute the AABB (approximation)
	min = min.array().min(tv2.array());
	min = min.array().min(tv3.array());
	min = min.array().min(tv4.array());
	min = min.array().min(tv5.array());
	min = min.array().min(tv6.array());
	min = min.array().min(tv7.array());
	min = min.array().min(tv8.array());

	max = max.array().max(tv2.array());
	max = max.array().max(tv3.array());
	max = max.array().max(tv4.array());
	max = max.array().max(tv5.array());
	max = max.array().max(tv6.array());
	max = max.array().max(tv7.array());
	max = max.array().max(tv8.array());

	// Center of the AABB is recomputed
	Eigen::Vector3d tpos(0.5*(min.x() + max.x()), 0.5*(min.y() + max.y()), 0.5*(min.z() + max.z()));

	_aabb = AABB(tpos, 0.5 * Eigen::Vector3d(max.x() - min.x(), max.y() - min.y(), max.z() - min.z()));

}

inline lmu::IFBox::IFBox(const Eigen::Affine3d & transform, const Eigen::Vector3d & size, int numSubdivisions, const std::string & name, double displacement) :
	ImplicitFunction(transform, createBox(transform, size, numSubdivisions), name),
	_size(size),
	_displacement(displacement),
	_numSubdivisions(numSubdivisions)
{
	Eigen::Vector3d min = -size * 0.5;
	Eigen::Vector3d max = size * 0.5;

	Eigen::Vector3d v1(min.x(), min.y(), min.z());
	Eigen::Vector3d v2(max.x(), min.y(), min.z());
	Eigen::Vector3d v3(min.x(), min.y(), max.z());
	Eigen::Vector3d v4(max.x(), min.y(), max.z());
	Eigen::Vector3d v5(min.x(), max.y(), min.z());
	Eigen::Vector3d v6(max.x(), max.y(), min.z());
	Eigen::Vector3d v7(min.x(), max.y(), max.z());
	Eigen::Vector3d v8(max.x(), max.y(), max.z());

	// Transformed vertices of the AABB
	Eigen::Vector3d tv1 = _transform * v1;
	Eigen::Vector3d tv2 = _transform * v2;
	Eigen::Vector3d tv3 = _transform * v3;
	Eigen::Vector3d tv4 = _transform * v4;
	Eigen::Vector3d tv5 = _transform * v5;
	Eigen::Vector3d tv6 = _transform * v6;
	Eigen::Vector3d tv7 = _transform * v7;
	Eigen::Vector3d tv8 = _transform * v8;

	// Recompute the AABB (approximation)
	min = tv1;
	min = min.array().min(tv2.array());
	min = min.array().min(tv3.array());
	min = min.array().min(tv4.array());
	min = min.array().min(tv5.array());
	min = min.array().min(tv6.array());
	min = min.array().min(tv7.array());
	min = min.array().min(tv8.array());

	max = tv1;
	max = max.array().max(tv2.array());
	max = max.array().max(tv3.array());
	max = max.array().max(tv4.array());
	max = max.array().max(tv5.array());
	max = max.array().max(tv6.array());
	max = max.array().max(tv7.array());
	max = max.array().max(tv8.array());

	// Center of the AABB is recomputed
	Eigen::Vector3d tpos(0.5*(min.x() + max.x()), 0.5*(min.y() + max.y()), 0.5*(min.z() + max.z()));

	_aabb = AABB(tpos, 0.5 * Eigen::Vector3d(max.x() - min.x(), max.y() - min.y(), max.z() - min.z()));
}

lmu::IFTorus::IFTorus(const Eigen::Affine3d& transform,	double minor, double major, const std::string& name) :
	ImplicitFunction(transform, lmu::Mesh(), name),	
	_major(major),
	_minor(minor)
{
}

ImplicitFunctionType lmu::IFTorus::type() const
{
	return ImplicitFunctionType::Torus;
}

std::shared_ptr<ImplicitFunction> lmu::IFTorus::clone() const
{
	return std::make_shared<IFTorus>(*this);
}

std::string lmu::IFTorus::serializeParameters() const
{
	return "";
}

Mesh lmu::IFTorus::createMesh() const
{
	return _mesh;
}

Eigen::Vector3d lmu::IFTorus::gradientLocal(const Eigen::Vector3d & localP, double h)
{
	double dx = (signedDistanceLocal(Eigen::Vector3d(localP.x() + h, localP.y(), localP.z())) - signedDistanceLocal(Eigen::Vector3d(localP.x() - h, localP.y(), localP.z()))) / (2.0 * h);
	double dy = (signedDistanceLocal(Eigen::Vector3d(localP.x(), localP.y() + h, localP.z())) - signedDistanceLocal(Eigen::Vector3d(localP.x(), localP.y() - h, localP.z()))) / (2.0 * h);
	double dz = (signedDistanceLocal(Eigen::Vector3d(localP.x(), localP.y(), localP.z() + h)) - signedDistanceLocal(Eigen::Vector3d(localP.x(), localP.y(), localP.z() - h))) / (2.0 * h);

	return Eigen::Vector3d(dx, dy, dz);
}

double lmu::IFTorus::signedDistanceLocal(const Eigen::Vector3d& localP)
{
	const auto n = Eigen::Vector3d(1.0, 0.0, 0.0);

	auto s = localP;
	float spin1 = n.dot(s);
	float spin0 = (s - spin1 * n).norm();
	spin0 -= _major;
	
	return std::sqrt(spin0 * spin0 + spin1 * spin1) - _minor;

	//if (!m_appleShaped)
	//	return std::sqrt(spin0 * spin0 + spin1 * spin1) - m_rminor;
	
	// apple shaped torus distance
	//float minorAngle = std::atan2(spin1, spin0); // minor angle
	//if (fabs(minorAngle) < m_cutOffAngle)
	//	return std::sqrt(spin0 * spin0 + spin1 * spin1) - m_rminor;
	//spin0 += 2 * m_rmajor - m_rminor;
	//if (minorAngle < 0)
	//	spin1 += m_appleHeight;
	//else
	//	spin1 -= m_appleHeight;
	//return -std::sqrt(spin0 * spin0 + spin1 * spin1);
}

lmu::IFCone::IFCone(const Eigen::Affine3d& transform, double angle, double height, const std::string& name) :
	ImplicitFunction(transform, lmu::Mesh(), name),
	_angle(angle),
	_height(height)
{	
}

ImplicitFunctionType lmu::IFCone::type() const
{
	return ImplicitFunctionType::Cone;
}

std::shared_ptr<ImplicitFunction> lmu::IFCone::clone() const
{
	return std::make_shared<IFCone>(*this);
}

std::string lmu::IFCone::serializeParameters() const
{
	return "";
}

Mesh lmu::IFCone::createMesh() const
{
	return _mesh;
}

Eigen::Vector3d lmu::IFCone::gradientLocal(const Eigen::Vector3d& localP, double h)
{
	double dx = (signedDistanceLocal(Eigen::Vector3d(localP.x() + h, localP.y(), localP.z())) - signedDistanceLocal(Eigen::Vector3d(localP.x() - h, localP.y(), localP.z()))) / (2.0 * h);
	double dy = (signedDistanceLocal(Eigen::Vector3d(localP.x(), localP.y() + h, localP.z())) - signedDistanceLocal(Eigen::Vector3d(localP.x(), localP.y() - h, localP.z()))) / (2.0 * h);
	double dz = (signedDistanceLocal(Eigen::Vector3d(localP.x(), localP.y(), localP.z() + h)) - signedDistanceLocal(Eigen::Vector3d(localP.x(), localP.y(), localP.z() - h))) / (2.0 * h);

	return Eigen::Vector3d(dx, dy, dz);
}

double lmu::IFCone::signedDistanceLocal(const Eigen::Vector3d& localP)
{
	const auto n = Eigen::Vector3d(1.0, 0.0, 0.0);
	// this is for one sided cone!
	auto s = localP;
	float g = s.dot(n); // distance to plane orhogonal to
								// axisdir through center
								// distance to axis
	float sqrS = s.squaredNorm();
	float f = sqrS - (g * g);
	if (f <= 0)
		f = 0;
	else
		f = std::sqrt(f);
	float da = std::cos(_angle) * f;
	float db = -std::sin(_angle) * g;
	if (g < 0 && da - db < 0) // is inside other side of cone -> disallow
		return std::sqrt(sqrS);
	return da + db;
}


lmu::IFPlane::IFPlane(const Eigen::Affine3d& transform, const std::string& name) :
	ImplicitFunction(transform, lmu::Mesh(), name)
{
}

ImplicitFunctionType lmu::IFPlane::type() const
{
	return ImplicitFunctionType::Plane;
}

std::shared_ptr<ImplicitFunction> lmu::IFPlane::clone() const
{
	return std::make_shared<IFPlane>(*this);
}

std::string lmu::IFPlane::serializeParameters() const
{
	return "";
}

Mesh lmu::IFPlane::createMesh() const
{
	return _mesh;
}

Eigen::Vector3d lmu::IFPlane::gradientLocal(const Eigen::Vector3d& localP, double h)
{
	double dx = (signedDistanceLocal(Eigen::Vector3d(localP.x() + h, localP.y(), localP.z())) - signedDistanceLocal(Eigen::Vector3d(localP.x() - h, localP.y(), localP.z()))) / (2.0 * h);
	double dy = (signedDistanceLocal(Eigen::Vector3d(localP.x(), localP.y() + h, localP.z())) - signedDistanceLocal(Eigen::Vector3d(localP.x(), localP.y() - h, localP.z()))) / (2.0 * h);
	double dz = (signedDistanceLocal(Eigen::Vector3d(localP.x(), localP.y(), localP.z() + h)) - signedDistanceLocal(Eigen::Vector3d(localP.x(), localP.y(), localP.z() - h))) / (2.0 * h);

	return Eigen::Vector3d(dx, dy, dz);
}

double lmu::IFPlane::signedDistanceLocal(const Eigen::Vector3d& localP)
{
	const auto n = Eigen::Vector3d(1.0, 0.0, 0.0);
	double d =  n.dot(localP);

	//std::cout << "D: " << d << std::endl;
	return d; 
}


