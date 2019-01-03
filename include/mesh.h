#ifndef MESH_H
#define MESH_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <igl/per_vertex_normals.h>
#include <igl/per_edge_normals.h>
#include <igl/per_face_normals.h>
#include <igl/AABB.h>


#include <iostream>
#include <memory>

#include <string>
#include <vector>

#include "pointcloud.h"
#include "congraph.h"

namespace lmu
{
	struct Mesh
	{
		Mesh(const Eigen::MatrixXd& vertices, const Eigen::MatrixXi& indices, const Eigen::MatrixXd& normals = Eigen::MatrixXd()) :
			indices(indices),
			vertices(vertices),		
			normals(normals),
			transform(Eigen::Affine3d::Identity())
		{
			if(normals.rows() == 0)
				igl::per_vertex_normals(this->vertices, this->indices, this->normals);				

			//std::cout << normals << std::endl;
		}

		Mesh() :			
			transform(Eigen::Affine3d::Identity())
		{
		}

		Eigen::MatrixXd vertices;
		Eigen::MatrixXd normals;
		Eigen::MatrixXi indices;
		Eigen::Affine3d transform;
	};

	void transform(Mesh& mesh);

	Mesh createBox(const Eigen::Affine3d& transform, const Eigen::Vector3d& size, int numSubdivisions);
	Mesh createSphere(const Eigen::Affine3d& transform, double radius, int stacks, int slices);
	Mesh createCylinder(const Eigen::Affine3d& transform, float bottomRadius, float topRadius, float height, int stacks, int slices);

	Mesh fromOBJFile(const std::string& file);

	enum class ImplicitFunctionType
	{
		Sphere = 0,
		Cylinder,
		Cone,
		Box,
		Null
	};

	std::string iFTypeToString(ImplicitFunctionType type);

	struct AABB
	{
		AABB()
		{
		}

		AABB(const Eigen::Vector3d& center, const Eigen::Vector3d& size) : 
			c(center), s(size)
		{
		}

		bool overlapsWith(const AABB& b, double e = 0.0)
		{
			Eigen::Vector3d ev(e, e, e);

			Eigen::Vector3d amin = c - s.cwiseAbs() - ev; 
			Eigen::Vector3d amax = c + s.cwiseAbs() + ev;

			Eigen::Vector3d bmin = b.c - b.s.cwiseAbs() - ev;
			Eigen::Vector3d bmax = b.c + b.s.cwiseAbs() + ev;

			//std::cout << amin.z() << " " << amax.z() << " " << bmin.z() << " " << bmax.z() << " s: " << b.s.z() << std::endl;
			//std::cout << ((amin.x() <= bmax.x() && amax.x() >= bmin.x()) || (amin.x() >= bmin.x() && amax.x() <= bmax.x()) || (bmin.x() >= amin.x() && bmax.x() <= amax.x())) << std::endl;
			//std::cout << ((amin.y() <= bmax.y() && amax.y() >= bmin.y()) || (amin.y() >= bmin.y() && amax.y() <= bmax.y()) || (bmin.y() >= amin.y() && bmax.y() <= amax.y())) << std::endl;
			//std::cout << ((amin.z() <= bmax.z() && amax.z() >= bmin.z()) || (amin.z() >= bmin.z() && amax.z() <= bmax.z()) || (bmin.z() >= amin.z() && bmax.z() <= amax.z())) << std::endl;

			return ((amin.x() <= bmax.x() && amax.x() >= bmin.x()) || (amin.x() >= bmin.x() && amax.x() <= bmax.x()) || (bmin.x() >= amin.x() && bmax.x() <= amax.x())) &&
				   ((amin.y() <= bmax.y() && amax.y() >= bmin.y()) || (amin.y() >= bmin.y() && amax.y() <= bmax.y()) || (bmin.y() >= amin.y() && bmax.y() <= amax.y())) &&
				   ((amin.z() <= bmax.z() && amax.z() >= bmin.z()) || (amin.z() >= bmin.z() && amax.z() <= bmax.z()) || (bmin.z() >= amin.z() && bmax.z() <= amax.z()));
		}

		AABB intersection(const AABB& b)
		{
			Eigen::Vector3d min1 = c - s;
			Eigen::Vector3d min2 = b.c - b.s;
			Eigen::Vector3d max1 = c + s;
			Eigen::Vector3d max2 = b.c + b.s;

			Eigen::Vector3d min(std::max(min1.x(), min2.x()), std::max(min1.y(), min2.y()), std::max(min1.z(), min2.z()));
			Eigen::Vector3d max(std::min(max1.x(), max2.x()), std::min(max1.y(), max2.y()), std::min(max1.z(), max2.z()));

			Eigen::Vector3d sinter = 0.5 * (max - min);
			Eigen::Vector3d cinter = min + sinter; 

			return AABB(cinter, sinter);
		}

		Eigen::Vector3d c; 
		Eigen::Vector3d s;
	};

	struct ImplicitFunction 
	{
		ImplicitFunction(const Eigen::Affine3d& transform, const Mesh& mesh, const std::string& name) :
			_transform(transform),
			_pos(0.0, 0.0, 0.0),
			_mesh(mesh),
			_name(name),
			_scoreWeight(1.0)
		{
			_pos = _transform * _pos;
			_invTrans = transform.inverse();
		}

		Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& worldP, double h = 0.001)
		{
			//world -> local
			auto pLocal = _invTrans * worldP;

			double d = signedDistanceLocal(pLocal);
					
			Eigen::Vector3d gLocal = gradientLocal(pLocal, h);

			//See also : https://math.stackexchange.com/questions/767369/how-does-the-transformation-on-a-point-affect-the-normal-at-that-point
			auto transposedTrans = _invTrans.matrix().block<3, 3>(0, 0).transpose();
			auto gWorld = transposedTrans * gLocal;//(_transform * (pLocal + gLocal) - p).normalized();

			Eigen::Vector4d res;
			res << (d), gWorld;

			return res;
		}

		double signedDistance(const Eigen::Vector3d& worldP)
		{
			auto localP = _invTrans * worldP;
			return signedDistanceLocal(localP);
		}

		Mesh& meshRef()
		{
			return _mesh;
		}

		const Mesh& meshCRef() const 
		{
			return _mesh;
		}

		PointCloud& points()
		{
			return _points;
		}

		const PointCloud& pointsCRef() const
		{
			return _points;
		}

		void setPoints(const PointCloud& points)
		{
			_points = points;
		}

		virtual ImplicitFunctionType type() const = 0;

		Eigen::Vector3d pos() const
		{
			return _pos;
		}

		std::string name() const
		{
			return _name;
		}

		Eigen::Affine3d transform() const
		{
			return _transform;
		}

	  // row-order: row1 " " row2 " " row3 " " row4
	  std::string serializeTransform() const {
	    std::string row1 = std::to_string(_transform(0,0)) + " " 
	      + std::to_string(_transform(0,1)) + " " 
	      + std::to_string(_transform(0,2)) + " " 
	      + std::to_string(_transform(0,3));
	    
	    std::string row2 = std::to_string(_transform(1,0)) + " " 
	      + std::to_string(_transform(1,1)) + " " 
	      + std::to_string(_transform(1,2)) + " " 
	      + std::to_string(_transform(1,3));
	    
	    std::string row3 = std::to_string(_transform(2,0)) + " " 
	      + std::to_string(_transform(2,1)) + " " 
	      + std::to_string(_transform(2,2)) + " " 
	      + std::to_string(_transform(2,3));
	    
	    std::string row4 = std::to_string(_transform(3,0)) + " " 
	      + std::to_string(_transform(3,1)) + " " 
	      + std::to_string(_transform(3,2)) + " " 
	      + std::to_string(_transform(3,3));
	    

	    return row1 + " " + row2 + " " + row3 + " " + row4;
	  }

	  virtual std::shared_ptr<ImplicitFunction> clone() const = 0;

	  virtual std::string serializeParameters() const = 0;

	  double scoreWeight() const
	  {
		  return _scoreWeight;
	  }

	  void setScoreWeight(double w)
	  {
		  _scoreWeight = w;
	  }

	  bool normalsPointOutside() const
	  {
		  return _normalsPointOutside;
	  }

	  void setNormalsPointOutside(bool outside)
	  {
		  _normalsPointOutside = outside;
	  }

	  std::vector<double>& pointWeights()
	  {
		  return _pointWeights;
	  }

	  AABB aabb() const
	  {
		  return _aabb;
	  }

	protected: 
		virtual Eigen::Vector3d gradientLocal(const Eigen::Vector3d& localP, double h) = 0;
		virtual double signedDistanceLocal(const Eigen::Vector3d& localP) = 0;

		Eigen::Affine3d _transform;
		Eigen::Affine3d _invTrans;

		AABB _aabb;
		Eigen::Vector3d _pos;
		Mesh _mesh;
		PointCloud _points;
		std::string _name;
		std::vector<double> _pointWeights;

		double _scoreWeight;

		bool _normalsPointOutside;
	};

	struct IFSphere : public ImplicitFunction 
	{
		IFSphere(const Eigen::Affine3d& transform, double radius, const std::string& name, double displacement = 0.0) : 
			ImplicitFunction(transform, createSphere(transform, radius, 50, 50), name),
			_radius(radius),
			_displacement(displacement)
		{
			_aabb = AABB(_pos, Eigen::Vector3d(radius, radius, radius));
		}
	
		virtual ImplicitFunctionType type() const override
		{
			return ImplicitFunctionType::Sphere;
		}

		double radius() const
		{
			return _radius; 
		}

		double displacement() const 
		{
		  return _displacement;
		}

		virtual std::shared_ptr<ImplicitFunction> clone() const override
		{
			return std::make_shared<IFSphere>(*this);
		}

		virtual std::string serializeParameters() const {
		  return std::to_string(_radius) + " " + std::to_string(_displacement);
		}

	protected:

		virtual double signedDistanceLocal(const Eigen::Vector3d& localP) override
		{
			double d = localP.norm() - _radius;
			
			double d2 = std::sin(_displacement * localP.x())*sin(_displacement * localP.y())*sin(_displacement * localP.z());

			return d + d2;
		}

		virtual Eigen::Vector3d gradientLocal(const Eigen::Vector3d& localP, double h) override
		{
			return localP.normalized();
		}

	private: 
		double _radius;
		double _displacement;
	};


	struct IFCylinder : ImplicitFunction
	{
		IFCylinder(const Eigen::Affine3d& transform, double radius, double height, const std::string& name) :
			ImplicitFunction(transform, createCylinder(transform, radius, radius, height, 200, 200), name),
			_radius(radius),
			_height(height)
		{
			_invTrans = transform.inverse();

			Eigen::Vector3d min(-radius, -height / 2.0, -radius);
			Eigen::Vector3d max(radius, height / 2.0, radius);

			min = _transform * min;
			max = _transform * max;
			_aabb = AABB(_pos, 0.5 * Eigen::Vector3d(max.x() - min.x(), max.y() - min.y(), max.z() - min.z()));
		}

		virtual ImplicitFunctionType type() const override
		{
			return ImplicitFunctionType::Cylinder;
		}

		virtual std::shared_ptr<ImplicitFunction> clone() const override
		{
			return std::make_shared<IFCylinder>(*this);
		}

		double radius() const 
		{
		  return _radius;
		}

		double height() const 
		{
		  return _height;
		}

		virtual std::string serializeParameters() const {
		  return std::to_string(_radius) + " " + std::to_string(_height);
		}

	protected:

		virtual Eigen::Vector3d gradientLocal(const Eigen::Vector3d& localP, double h) override
		{
			double dx = (signedDistanceLocalInline(Eigen::Vector3d(localP.x() + h, localP.y(), localP.z())) - signedDistanceLocalInline(Eigen::Vector3d(localP.x() - h, localP.y(), localP.z()))) / (2.0 * h);
			double dy = (signedDistanceLocalInline(Eigen::Vector3d(localP.x(), localP.y() + h, localP.z())) - signedDistanceLocalInline(Eigen::Vector3d(localP.x(), localP.y() - h, localP.z()))) / (2.0 * h);
			double dz = (signedDistanceLocalInline(Eigen::Vector3d(localP.x(), localP.y(), localP.z() + h)) - signedDistanceLocalInline(Eigen::Vector3d(localP.x(), localP.y(), localP.z() - h))) / (2.0 * h);

			return Eigen::Vector3d(dx, dy, dz);
		}

		virtual double signedDistanceLocal(const Eigen::Vector3d& localP) override
		{
			return signedDistanceLocalInline(localP);
		}
	
	private:

		inline double signedDistanceLocalInline(const Eigen::Vector3d& localP)
		{
			double l = Eigen::Vector2d(localP.x(), localP.z()).norm();
			//double dx = std::abs(l) - _height;
			//double dy = std::abs(localP.y()) - _height;
			return std::max(l - _radius, abs(localP.y()) - _height / 2.0);// std::min(std::max(dx, dy), 0.0) + Eigen::Vector2d(std::max(dx, 0.0), std::max(dy, 0.0)).norm();
		}

		double _radius;
		double _height;
	};

	struct IFBox : public ImplicitFunction
	{
		IFBox(const Eigen::Affine3d& transform, const Eigen::Vector3d& size, int numSubdivisions, const std::string& name, double displacement = 0.0) :
			ImplicitFunction(transform, createBox(transform, size, numSubdivisions), name),
			_size(size),
			_displacement(displacement)
		{
			Eigen::Vector3d min = -size * 0.5;
			Eigen::Vector3d max = size * 0.5;

			min = _transform * min;
			max = _transform * max;
			_aabb = AABB(_pos, 0.5 * Eigen::Vector3d(max.x() - min.x(), max.y() - min.y(), max.z() - min.z()));
		}

		virtual ImplicitFunctionType type() const override
		{
			return ImplicitFunctionType::Box;
		}

		virtual std::shared_ptr<ImplicitFunction> clone() const override
		{
			return std::make_shared<IFBox>(*this);
		}

		Eigen::Vector3d size() const {
		  return _size;
		}

		double displacement() const {
		  return _displacement;
		}

		virtual std::string serializeParameters() const {
		  return std::to_string(_size[0]) + " " 
		    + std::to_string(_size[1]) + " "
		    + std::to_string(_size[2]) + " "
		    + std::to_string(_displacement);
		}
				
	protected:

		virtual Eigen::Vector3d gradientLocal(const Eigen::Vector3d& localP, double h) override
		{
			double dx = (signedDistanceLocalInline(Eigen::Vector3d(localP.x() + h, localP.y(), localP.z())) - signedDistanceLocalInline(Eigen::Vector3d(localP.x() - h, localP.y(), localP.z()))) / (2.0 * h);
			double dy = (signedDistanceLocalInline(Eigen::Vector3d(localP.x(), localP.y() + h, localP.z())) - signedDistanceLocalInline(Eigen::Vector3d(localP.x(), localP.y() - h, localP.z()))) / (2.0 * h);
			double dz = (signedDistanceLocalInline(Eigen::Vector3d(localP.x(), localP.y(), localP.z() + h)) - signedDistanceLocalInline(Eigen::Vector3d(localP.x(), localP.y(), localP.z() - h))) / (2.0 * h);

			return Eigen::Vector3d(dx, dy, dz);
		}

		virtual double signedDistanceLocal(const Eigen::Vector3d& localP) override
		{
			return signedDistanceLocalInline(localP);
		}

	private:

		inline double signedDistanceLocalInline(const Eigen::Vector3d& localP)
		{
			double d1 = std::max(abs(localP.x()) - _size.x() / 2.0, std::max(abs(localP.y()) - _size.y() / 2.0, abs(localP.z()) - _size.z() / 2.0));

			double d2 = std::sin(_displacement * localP.x())*sin(_displacement * localP.y())*sin(_displacement * localP.z());

			return d1 + d2;			
		}

		Eigen::Vector3d _size;
		double _displacement;
	};

	struct IFNull : public ImplicitFunction
	{
		IFNull(const std::string& name) :
			ImplicitFunction(Eigen::Affine3d::Identity(), lmu::Mesh(), name)
		{
		}

		virtual ImplicitFunctionType type() const override
		{
			return ImplicitFunctionType::Null;
		}

		std::shared_ptr<ImplicitFunction> clone() const override
		{
			return std::make_shared<IFNull>(*this);
		}
		
		virtual std::string serializeParameters() const 
		{
		  return "";
		}

	protected:

		virtual Eigen::Vector3d gradientLocal(const Eigen::Vector3d& localP, double h) override
		{
			return Eigen::Vector3d(0, 0, 0);
		}

		virtual double signedDistanceLocal(const Eigen::Vector3d& localP) override
		{
			return 0.0;
		}
	};

	struct IFCone : public ImplicitFunction
	{
		IFCone(const Eigen::Affine3d& transform, const Eigen::Vector3d& c, const std::string& name) :
			ImplicitFunction(transform, createCylinder(transform, c.x(), c.y(), c.z(), 200, 200), name),
			_c(c)
		{
		}

		virtual ImplicitFunctionType type() const override
		{
			return ImplicitFunctionType::Cone;
		}

		virtual std::shared_ptr<ImplicitFunction> clone() const override
		{
			return std::make_shared<IFCone>(*this);
		}

		inline double sign(double v)
		{
			if (v < 0.0)
				return -1.0;
			else if (v > 0.0)
				return 1.0;
			else
				return 0.0;
		}

		Eigen::Vector3d c() const {
		  return _c;
		}

		virtual std::string serializeParameters() const {
		  return std::to_string(_c[0]) + " " 
		    + std::to_string(_c[1]) + " "
		    + std::to_string(_c[2]);
		}

	protected:

		virtual Eigen::Vector3d gradientLocal(const Eigen::Vector3d& localP, double h) override
		{
			double dx = (signedDistanceLocalInline(Eigen::Vector3d(localP.x() + h, localP.y(), localP.z())) - signedDistanceLocalInline(Eigen::Vector3d(localP.x() - h, localP.y(), localP.z()))) / (2.0 * h);
			double dy = (signedDistanceLocalInline(Eigen::Vector3d(localP.x(), localP.y() + h, localP.z())) - signedDistanceLocalInline(Eigen::Vector3d(localP.x(), localP.y() - h, localP.z()))) / (2.0 * h);
			double dz = (signedDistanceLocalInline(Eigen::Vector3d(localP.x(), localP.y(), localP.z() + h)) - signedDistanceLocalInline(Eigen::Vector3d(localP.x(), localP.y(), localP.z() - h))) / (2.0 * h);

			return Eigen::Vector3d(dx, dy, dz);
		}

		virtual double signedDistanceLocal(const Eigen::Vector3d& localP) override
		{
			return signedDistanceLocalInline(localP);
		}

	private:

		inline double signedDistanceLocalInline(const Eigen::Vector3d& localP) 
		{
			Eigen::Vector2d q = Eigen::Vector2d(Eigen::Vector2d(localP.x(), localP.z()).norm(), localP.y());
			Eigen::Vector2d v = Eigen::Vector2d(_c.z()*_c.y() / _c.x(), -_c.z());
			Eigen::Vector2d w = v - q;
			Eigen::Vector2d vv = Eigen::Vector2d(v.dot(v), v.x()*v.x());
			Eigen::Vector2d qv = Eigen::Vector2d(v.dot(w), v.x()*w.x());
			Eigen::Vector2d d;
			d.x() = std::max(qv.x(), 0.0)*qv.x() / vv.x();
			d.y() = std::max(qv.y(), 0.0)*qv.y() / vv.y();

			return sqrt(w.dot(w) - std::max(d.x(), d.y())) * sign(std::max(q.y()*v.x() - q.x()*v.y(), w.y()));
		}

		Eigen::Vector3d _c;
	};
	
	// Read primitives saved with the .FIT file format
	std::vector<std::shared_ptr<ImplicitFunction>> fromFile(const std::string& file, double scaling = 1.0);

	void movePointsToSurface(const std::vector<std::shared_ptr<ImplicitFunction>>& functions, bool filter = false, double threshold = 0.0);

	void reducePoints(const std::vector<std::shared_ptr<ImplicitFunction>>& functions, const lmu::Graph& graph, double h);
	void arrangeGradients(const std::vector<std::shared_ptr<ImplicitFunction>>& functions, const lmu::Graph& graph, double h);



	// Save primitives with the .PRIM file format
	void writePrimitives(const std::string& filename, 
			     const std::vector<std::shared_ptr<ImplicitFunction>>& shapes);

	// Read primitives saved with the .PRIM file format
	std::vector<std::shared_ptr<ImplicitFunction>> fromFilePRIM(const std::string& file);
}

#endif