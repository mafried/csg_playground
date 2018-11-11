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

#include "../include/constants.h"


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

    } else if (name.find("box") != std::string::npos) {
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

			//std::cout << sampleP << std::endl << (sampleP - (sampleGradFunction * sampleDistFunction)) << std::endl;
			//std::cout << sampleGradFunction << std::endl;
			//std::cout << "----------" << std::endl;

			Eigen::Matrix<double, 1, 6> newPN;

			Eigen::Vector3d newP = (sampleP - (sampleDistFunction * sampleGradFunction));

			newPN << newP.transpose() , sampleN.transpose();
								
			double distAfter = func->signedDistance(sampleP - (sampleDistFunction * sampleGradFunction));

			if (filter && std::abs(distAfter) <  threshold)
			{
				points.push_back(newPN);
			}
			else
			{
				func->points().row(j) = newPN;
			}
		}

		if (filter)
		{
			Eigen::MatrixXd m(points.size(), 6); 

			int i = 0;
			for (const auto& point : points)
				m.row(i++) = point;
				
			func->points() = m;
		}
	}
}

void lmu::reducePoints(const std::vector<std::shared_ptr<ImplicitFunction>>& functions, const lmu::Graph& graph, double h)
{
	for (auto& f : functions)
	{
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
			
		//Find most significant point 
		int furthestAwayPointIdx = -1;
		double curLargestDistance = 0.0;
		for (int i = 0; i < f->pointsCRef().rows(); ++i)
		{
			Eigen::Vector3d p = f->pointsCRef().row(i).leftCols(3);
			Eigen::Vector3d n = f->pointsCRef().row(i).rightCols(3);
			Eigen::Vector3d g = f->signedDistanceAndGradient(p, h).bottomRows(3);

			if (outside && g.dot(n) < 0.0)
				continue; 

			if (!outside && g.dot(n) >= 0.0)
				continue;

			double accDistance = 0.0;
			for (auto& f2 : functions)
			{
				if (f == f2 || !areConnected(graph, f, f2))
					continue; 

				accDistance += std::abs(f2->signedDistance(p));				
			}

			if (accDistance > curLargestDistance)
			{
				furthestAwayPointIdx = i;
				curLargestDistance = accDistance;
			}
		}

		if (furthestAwayPointIdx == -1)
			throw std::runtime_error("Shape has no most significant point.");

		auto point = f->pointsCRef().row(furthestAwayPointIdx);
		
		Eigen::Vector3d p = point.leftCols(3);
		Eigen::Vector3d n = point.rightCols(3);
		Eigen::Vector3d g = f->signedDistanceAndGradient(p, h).bottomRows(3);

		Eigen::Matrix<double, 1, 6> newPoint; 
		newPoint << p.transpose(), (outside ? g : -g).transpose(); 

		f->setScoreWeight(f->points().rows());
		f->points() = newPoint;

		std::cout << f->name() << ": " << (outside ? "Outside" : "Inside") << " " << (g.dot(n) > 0.0 ? "Outside" : "Inside") << std::endl;


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
