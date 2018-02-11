#ifndef MESH_H
#define MESH_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <igl/per_vertex_normals.h>

#include <iostream>
#include <memory>

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

	Mesh createBox(const Eigen::Affine3d& transform, const Eigen::Vector3d& size);
	Mesh createSphere(const Eigen::Affine3d& transform, double radius, int stacks, int slices);
	Mesh createCylinder(const Eigen::Affine3d& transform, double radius, double height, int stacks, int slices);

	Mesh fromOBJFile(const std::string& file);

	enum class ImplicitFunctionType
	{
		Sphere = 0,
		Cylinder,
		Null
	};

	std::string iFTypeToString(ImplicitFunctionType type);
	
	struct ImplicitFunction
	{
		ImplicitFunction(const Eigen::Affine3d& transform, const Mesh& mesh, const std::string& name) :
			_transform(transform),
			_pos(0.0,0.0,0.0),
			_mesh(mesh),
			_name(name)
		{
			_pos = _transform * _pos;
		}

		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p) = 0;

		Mesh& mesh()
		{
			return _mesh;
		}

		Eigen::MatrixXd& points()
		{
			return _points;
		}

		void setPoints(const Eigen::MatrixXd& points)
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

		virtual std::shared_ptr<ImplicitFunction> clone() const = 0;

	protected: 
		Eigen::Affine3d _transform;
		Eigen::Vector3d _pos;
		Mesh _mesh;
		Eigen::MatrixXd _points;
		std::string _name;
	};

	struct IFSphere : public ImplicitFunction 
	{
		IFSphere(const Eigen::Affine3d& transform, double radius, const std::string& name) : 
			ImplicitFunction(transform, createSphere(transform, radius, 50, 50), name),
			_radius(radius)
		{
		}

		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p) override
		{
			double d = (
				((p(0) - _pos(0)) * (p(0) - _pos(0)) +
				 (p(1) - _pos(1)) * (p(1) - _pos(1)) +
				 (p(2) - _pos(2)) * (p(2) - _pos(2))) -
				_radius*_radius);

			//std::cout << _name << " distance to " << p << ": " << d << std::endl;

			Eigen::Vector3d gradient = (2.0 * (p - _pos));
			gradient.normalize();

			Eigen::Vector4d res;
			res << d, gradient;

			return res;
		}

		virtual ImplicitFunctionType type() const override
		{
			return ImplicitFunctionType::Sphere;
		}

		double radius() const
		{
			return _radius; 
		}

		std::shared_ptr<ImplicitFunction> clone() const override
		{
			return std::make_shared<IFSphere>(*this);
		}

	private: 
		double _radius;
	};

	struct IFCylinder : public ImplicitFunction
	{
		IFCylinder(const Eigen::Affine3d& transform, double radius, double height, const std::string& name) :
			ImplicitFunction(transform, createCylinder(transform, radius, height, 50, 50), name),
			_radius(radius),
			_height(height)
		{
		}

		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p) override
		{
			return Eigen::Vector4d(); //TODO
		}

		virtual ImplicitFunctionType type() const override
		{
			return ImplicitFunctionType::Cylinder;
		}

		std::shared_ptr<ImplicitFunction> clone() const override
		{
			return std::make_shared<IFCylinder>(*this);
		}

	private:
		double _radius;
		double _height;
	};

	struct IFNull : public ImplicitFunction
	{
		IFNull(const std::string& name) :
			ImplicitFunction(Eigen::Affine3d::Identity(), lmu::Mesh(), name)
		{
		}

		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p) override
		{
			return Eigen::Vector4d(0,0,0,0);
		}

		virtual ImplicitFunctionType type() const override
		{
			return ImplicitFunctionType::Null;
		}

		std::shared_ptr<ImplicitFunction> clone() const override
		{
			return std::make_shared<IFNull>(*this);
		}

	private:
		double _radius;
		double _height;
	};
}

#endif