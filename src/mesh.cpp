#include "..\include\mesh.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <memory>
#include <algorithm>

#include <igl/readOBJ.h>
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


#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/convex_hull_3.h>

#include <CGAL/license/Polyhedron.h>
#include <CGAL/boost/graph/named_function_params.h>
#include <CGAL/boost/graph/named_params_helper.h>

#include <CGAL/boost/graph/named_params_helper.h>
#include <CGAL/boost/graph/named_function_params.h>


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


Mesh lmu::createPolytope(const Eigen::Affine3d& transform, const std::vector<Eigen::Vector3d>& p, const std::vector<Eigen::Vector3d>& n)
{	

	dd_PolyhedraPtr poly;
	dd_MatrixPtr A, G;
	dd_ErrorType err;

	dd_set_global_constants();  /* First, this must be called to use cddlib. */

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
		return Mesh();
		
	G = dd_CopyGenerators(poly);
	if (G->rowsize == 0)
		return Mesh();
	
	//dd_WriteMatrix(stdout, A);
	//dd_WriteMatrix(stdout, G);
	
	std::vector<Point_3> points;
	points.reserve(G->rowsize);
	for (int i = 0; i < G->rowsize; i++)
	{
		points.push_back(Point_3(dd_get_d(G->matrix[i][1]), dd_get_d(G->matrix[i][2]), dd_get_d(G->matrix[i][3])));
		//std::cout << "Poly point: " << points.back().x() << " " << points.back().y() << " " << points.back().z() << std::endl;
	}

	dd_FreeMatrix(A);
	dd_FreeMatrix(G);	
	dd_free_global_constants();
		
	//CGAL::advancing_front_surface_reconstruction(points.begin(),
	//	points.end(),
	//	std::back_inserter(facets));

	CGAL::Object obj;
	CGAL::convex_hull_3(points.begin(), points.end(), obj);	
	const Polyhedron_3* ph = CGAL::object_cast<Polyhedron_3>(&obj);
	if (!ph) {
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

Mesh lmu::fromOBJFile(const std::string & file)
{
	Eigen::MatrixXd vertices;
	Eigen::MatrixXi indices;
	
	Mesh mesh;

	igl::readOBJ(file, vertices, indices);
	
	return Mesh(vertices, indices);
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

      res.push_back(std::make_shared<IFCone>(t, c, name));

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

void lmu::reducePoints(const std::vector<std::shared_ptr<ImplicitFunction>>& functions, const lmu::Graph& graph, double h)
{
	std::unordered_map<std::shared_ptr<ImplicitFunction>, std::vector<Eigen::Matrix<double, 1, 6>>> selectedPoints;

	for (auto& f : functions)
	{
		std::vector<Eigen::Matrix<double, 1, 6>> selectedPoints;

		//Check orientation
		int numSameSide = 0;
		for (int i = 0; i < f->pointsCRef().rows(); ++i)
		{
			Eigen::Vector3d p = f->pointsCRef().row(i).leftCols(3);
			Eigen::Vector3d n = f->pointsCRef().row(i).rightCols(3);
			Eigen::Vector3d g = f->signedDistanceAndGradient(p, h).bottomRows(3);
			numSameSide += g.dot(n) > 0.0;
		}
		bool outside = numSameSide >= f->pointsCRef().rows() / 2;
			
		for (auto& f2 : functions)
		{
			if (f == f2 || !areConnected(graph, f, f2))
				continue;

			int furthestAwayPointIdx = 0;
			double curLargestDistance = 0.0;
			for (int i = 0; i < f->pointsCRef().rows(); ++i)
			{
				Eigen::Vector3d p = f->pointsCRef().row(i).leftCols(3);
				Eigen::Vector3d n = f->pointsCRef().row(i).rightCols(3);
				Eigen::Vector3d g = f->signedDistanceAndGradient(p, h).bottomRows(3);

				double d = std::abs(f2->signedDistance(p));
				if (d > curLargestDistance)
				{
					furthestAwayPointIdx = i;
					curLargestDistance = d;
				}
			}
			
			auto point = f->points().row(furthestAwayPointIdx);
			Eigen::Vector3d p = point.leftCols(3);
			Eigen::Vector3d n = point.rightCols(3);
			Eigen::Vector3d g = f->signedDistanceAndGradient(p, h).bottomRows(3);

			Eigen::Matrix<double, 1, 6> newPoint;
			newPoint << p.transpose(), (outside ? g : -g).transpose();

			selectedPoints.push_back(newPoint);
		}

		f->setScoreWeight(f->points().rows() / selectedPoints.size());

		PointCloud pc;
		pc.resize(selectedPoints.size(), 6);
		for (int i = 0; i < pc.rows(); ++i)
		{
			pc.row(i) = selectedPoints[i];
		}
		f->points() = pc;

		//std::cout << f->name() << ": " << (outside ? "Outside" : "Inside") << " " << (g.dot(n) > 0.0 ? "Outside" : "Inside") << std::endl;
	}
}




void lmu::arrangeGradients(const std::vector<std::shared_ptr<ImplicitFunction>>& functions, const lmu::Graph& graph, double h)
{
	std::unordered_map<std::shared_ptr<ImplicitFunction>, std::vector<Eigen::Matrix<double, 1, 6>>> selectedPoints;

	for (auto& f : functions)
	{
		std::vector<Eigen::Matrix<double, 1, 6>> selectedPoints;

		//Check orientation
		int numSameSide = 0;
		for (int i = 0; i < f->pointsCRef().rows(); ++i)
		{
			Eigen::Vector3d p = f->pointsCRef().row(i).leftCols(3);
			Eigen::Vector3d n = f->pointsCRef().row(i).rightCols(3);
			Eigen::Vector3d g = f->signedDistanceAndGradient(p, h).bottomRows(3);
			numSameSide += g.dot(n) > 0.0;
		}
		bool outside = numSameSide >= f->pointsCRef().rows() / 2;
		//std::cout << f->name() << ": " << numSameSide << " " << f->pointsCRef().rows() << std::endl;
		
		for (int i = 0; i < f->pointsCRef().rows(); ++i)
		{
			Eigen::Vector3d p = f->pointsCRef().row(i).leftCols(3);
			Eigen::Vector3d g = f->signedDistanceAndGradient(p, h).bottomRows(3);

			Eigen::Matrix<double, 1, 6> newPoint;
			newPoint << p.transpose(), (outside ? g : -g).transpose();

			f->points().row(i) = newPoint;
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
	ImplicitFunction(transform, createPolytope(transform, p, n), name)
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

Eigen::Vector3d lmu::IFPolytope::gradientLocal(const Eigen::Vector3d & localP, double h)
{
	auto worldP = _transform * localP;

	int i;
	Eigen::RowVector3d c;
	_tree.squared_distance(_mesh.vertices, _mesh.indices, worldP.transpose(), i, c);
	
	return _mesh.normals.row(i);
}

double lmu::IFPolytope::signedDistanceLocal(const Eigen::Vector3d & localP)
{
	auto worldP = _transform * localP;

	double s;
	int i;
	Eigen::RowVector3d c;
	
	double sqrd = _tree.squared_distance(_mesh.vertices, _mesh.indices, worldP.transpose(), i, c);
	
	s = 1. - 2.*_hier.winding_number(worldP);
	//s = 1;
	
	return std::sqrt(sqrd) * s;
}
