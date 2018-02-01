#include "..\include\mesh.h"

#include <vector>
#include <iostream>
#include <cmath>

#include <igl/readOBJ.h>

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


Mesh lmu::createBox(const Eigen::Affine3d& transform, const Eigen::Vector3d& size)
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



	Mesh mesh;

	mesh.vertices.resize(verticesVector.size(), 3);

	int i = 0;
	for (const auto& vs : verticesVector)
	{
		mesh.vertices.row(i) = vs;		
		i++;
	}

	mesh.indices.resize(indicesVector.size() / 3, 3);

	for (int j = 0; j < indicesVector.size() / 3; j++)
	{
		mesh.indices.row(j) = Eigen::RowVector3i(indicesVector[j*3], indicesVector[j*3+1], indicesVector[j*3+2]);
		//std::cout << "HERE";
	}

	mesh.transform = transform;

	igl::per_vertex_normals(mesh.vertices, mesh.indices, mesh.normals);

	lmu::transform(mesh);

	return mesh;
}

Mesh lmu::createCylinder(const Eigen::Affine3d & transform, double radius, double height, int stacks, int slices)
{
	return Mesh(); //TODO
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
	case ImplicitFunctionType::Null:
		return "Null";
	default:
		return "Undefined Type";
	}
}


