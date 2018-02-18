#include "..\include\mesh.h"

#include <vector>
#include <iostream>
#include <cmath>

#include <igl/readOBJ.h>
#include <igl/signed_distance.h>
#include <igl/upsample.h>

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

	lmu::transform(mesh);

	Mesh upsampledMesh;// = mesh;

	igl::upsample(mesh.vertices, mesh.indices, upsampledMesh.vertices, upsampledMesh.indices, numSubdivisions);


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
	case ImplicitFunctionType::Null:
		return "Null";
	default:
		return "Undefined Type";
	}
}

Eigen::Vector4d lmu::IFMeshSupported::signedDistanceAndGradient(const Eigen::Vector3d & p)
{
	Eigen::MatrixXd points(1,3);
	points.row(0) << p.x(), p.y(), p.z();

	Eigen::VectorXd d;
	Eigen::VectorXi i;
	Eigen::MatrixXd n, c;

	igl::signed_distance_pseudonormal(points, _mesh.vertices, _mesh.indices, _tree, _fn, _vn, _en, _emap, d, i, c, n);

	return Eigen::Vector4d(d(0), n.row(0).x(), n.row(0).y(), n.row(0).z());
}
