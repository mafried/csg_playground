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

		Mesh& meshRef()
		{
			return _mesh;
		}

		const Mesh& meshCRef() const 
		{
			return _mesh;
		}

		Eigen::MatrixXd& points()
		{
			return _points;
		}

		const Eigen::MatrixXd& pointsCRef()
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
		IFSphere(const Eigen::Affine3d& transform, double radius, const std::string& name, double displacement = 0.0) : 
			ImplicitFunction(transform, createSphere(transform, radius, 50, 50), name),
			_radius(radius),
			_displacement(displacement)
		{
		}

		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p) override
		{
			double d = (
				((p(0) - _pos(0)) * (p(0) - _pos(0)) +
				 (p(1) - _pos(1)) * (p(1) - _pos(1)) +
				 (p(2) - _pos(2)) * (p(2) - _pos(2))) -
				_radius*_radius);

			
			double d2 = std::sin(_displacement * p.x())*sin(_displacement * p.y())*sin(_displacement * p.z());
			

			//std::cout << _name << " distance to " << p << ": " << d << std::endl;

			Eigen::Vector3d gradient = (2.0 * (p - _pos));
			gradient.normalize();

			Eigen::Vector4d res;
			res << (d+d2), gradient;

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
		double _displacement;
	};

	class IFMeshSupported : public ImplicitFunction
	{
	public: 
		IFMeshSupported(const Eigen::Affine3d& transform, const Mesh& mesh, const std::string& name) :
			ImplicitFunction(transform, mesh, name)
		{
			_tree.init(_mesh.vertices, _mesh.indices);

			igl::per_face_normals(_mesh.vertices, _mesh.indices, _fn);
			igl::per_vertex_normals(_mesh.vertices, _mesh.indices, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, _fn, _vn);
			igl::per_edge_normals(_mesh.vertices, _mesh.indices, igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM, _fn, _en, _e, _emap);
		}

		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p) override;

	private: 
		//We need all this stuff (especially the AABB tree for accelerating distance lookups.
		igl::AABB<Eigen::MatrixXd, 3> _tree;
		Eigen::MatrixXd _fn, _vn, _en; //note that _vn is the same as mesh's _normals. TODO
		Eigen::MatrixXi _e;
		Eigen::VectorXi _emap;
	};

	struct IFCylinder : public IFMeshSupported
	{
		IFCylinder(const Eigen::Affine3d& transform, double radius, double height, const std::string& name) :
			IFMeshSupported(transform, createCylinder(transform, radius, radius, height, 200, 200), name),
			_radius(radius),
			_height(height)
		{
			_invTrans = transform.inverse();
		}

		virtual ImplicitFunctionType type() const override
		{
			return ImplicitFunctionType::Cylinder;
		}

		std::shared_ptr<ImplicitFunction> clone() const override
		{
			return std::make_shared<IFCylinder>(*this);
		}

		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p) override
		{		

			auto ps = _invTrans * p;

			double delta = 0.001;

			double d = distance(ps);
			//double d1 = distance(Eigen::Vector3d(ps.x() + delta, ps.y(), ps.z()));
			//double d2 = distance(Eigen::Vector3d(ps.x(), ps.y() + delta, ps.z()));

			//Eigen::Vector3d v1(ps.x()) + delta, d1


			//std::cout << _name << " distance to " << p << ": " << d << std::endl;

			Eigen::Vector3d gradient = gradientPerCentralDifferences(ps, 0.001);
			gradient.normalize();

			Eigen::Vector4d res;
			res << (d), gradient;

			return res;
		}

		inline double distance(const Eigen::Vector3d& ps)
		{
			double l = Eigen::Vector2d(ps.x(), ps.z()).norm();
			double dx = std::abs(l) - _height;
			double dy = std::abs(ps.y()) - _height;
			return std::max(l - _radius, abs(ps.y()) - _height / 2.0);// std::min(std::max(dx, dy), 0.0) + Eigen::Vector2d(std::max(dx, 0.0), std::max(dy, 0.0)).norm();
		}

		inline Eigen::Vector3d gradientPerCentralDifferences(const Eigen::Vector3d& ps, double h)
		{
			double dx = (distance(Eigen::Vector3d(ps.x() + h, ps.y(), ps.z())) - distance(Eigen::Vector3d(ps.x() - h, ps.y(), ps.z()))) / (2.0 * h);
			double dy = (distance(Eigen::Vector3d(ps.x(), ps.y() + h, ps.z())) - distance(Eigen::Vector3d(ps.x(), ps.y() - h, ps.z()))) / (2.0 * h);
			double dz = (distance(Eigen::Vector3d(ps.x(), ps.y(), ps.z() + h)) - distance(Eigen::Vector3d(ps.x(), ps.y(), ps.z() - h))) / (2.0 * h);

			return Eigen::Vector3d(dx, dy, dz);
		}

	private:
		double _radius;
		double _height;
		Eigen::Affine3d _invTrans;
	};

	struct IFBox : public IFMeshSupported
	{
		IFBox(const Eigen::Affine3d& transform, const Eigen::Vector3d& size, int numSubdivisions, const std::string& name, double displacement = 0.0) :
			IFMeshSupported(transform, createBox(transform, size, numSubdivisions), name),
			_size(size),
			_displacement(displacement)
		{
			_invTrans = transform.inverse();
		}

		virtual ImplicitFunctionType type() const override
		{
			return ImplicitFunctionType::Box;
		}

		std::shared_ptr<ImplicitFunction> clone() const override
		{
			return std::make_shared<IFBox>(*this);
		}

		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p) override
		{
			auto ps = _invTrans * p;

			double d = distance(ps);
		
			Eigen::Vector3d gradient = gradientPerCentralDifferences(ps, 0.001);
			gradient.normalize();

			Eigen::Vector4d res;
			res << (d), gradient;

			return res;
		}


		inline double distance(const Eigen::Vector3d& ps)
		{
			double d1 = std::max(abs(ps.x()) - _size.x() / 2.0, std::max(abs(ps.y()) - _size.y() / 2.0, abs(ps.z()) - _size.z() / 2.0));

			double d2 = std::sin(_displacement * ps.x())*sin(_displacement * ps.y())*sin(_displacement * ps.z());

			return d1 + d2;			
		}

		inline Eigen::Vector3d gradientPerCentralDifferences(const Eigen::Vector3d& ps, double h)
		{
			double dx = (distance(Eigen::Vector3d(ps.x() + h, ps.y(), ps.z())) - distance(Eigen::Vector3d(ps.x() - h, ps.y(), ps.z()))) / (2.0 * h);
			double dy = (distance(Eigen::Vector3d(ps.x(), ps.y() + h, ps.z())) - distance(Eigen::Vector3d(ps.x(), ps.y() - h, ps.z()))) / (2.0 * h);
			double dz = (distance(Eigen::Vector3d(ps.x(), ps.y(), ps.z() + h)) - distance(Eigen::Vector3d(ps.x(), ps.y(), ps.z() - h))) / (2.0 * h);

			return Eigen::Vector3d(dx, dy, dz);
		}

	private:
		Eigen::Vector3d _size;
		Eigen::Affine3d _invTrans;
		float _displacement;
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

	static double compute_dot_product(double v1[], double v2[]) {
		return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
	}

	static double compute_norm2(double v[]) {
		return sqrt(compute_dot_product(v, v));
	}

	struct IFCylinder2 : public ImplicitFunction
	{
		IFCylinder2(const std::string& name, const std::vector<double>& parameters) :
			ImplicitFunction(Eigen::Affine3d::Identity(), lmu::Mesh(), name),
			_parameters(parameters)
		{
		}

		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p) override
		{
			double d = distance(p);

			Eigen::Vector3d gradient = gradientPerCentralDifferences(p, 0.001);
			gradient.normalize();

			Eigen::Vector4d res;
			res << (d), gradient;

			return res;
		}

		inline double distance(const Eigen::Vector3d& p)
		{
			double axis_dir[] = { _parameters[0], _parameters[1], _parameters[2] };
			double axis_pos[] = { _parameters[3], _parameters[4], _parameters[5] };
			double radius = _parameters[6];
			double diff[] = { p.x() - axis_pos[0], p.y() - axis_pos[1], p.z() - axis_pos[2] };
			double lamb = compute_dot_product(axis_dir, diff);
			double v[] = { diff[0] - lamb*axis_dir[0], diff[1] - lamb*axis_dir[1],
				diff[2] - lamb*axis_dir[2] };
			double axis_dist = compute_norm2(v);
			double d = axis_dist - radius;
			
			return -d; //???
		}

		inline Eigen::Vector3d gradientPerCentralDifferences(const Eigen::Vector3d& ps, double h)
		{
			double dx = (distance(Eigen::Vector3d(ps.x() + h, ps.y(), ps.z())) - distance(Eigen::Vector3d(ps.x() - h, ps.y(), ps.z()))) / (2.0 * h);
			double dy = (distance(Eigen::Vector3d(ps.x(), ps.y() + h, ps.z())) - distance(Eigen::Vector3d(ps.x(), ps.y() - h, ps.z()))) / (2.0 * h);
			double dz = (distance(Eigen::Vector3d(ps.x(), ps.y(), ps.z() + h)) - distance(Eigen::Vector3d(ps.x(), ps.y(), ps.z() - h))) / (2.0 * h);

			return Eigen::Vector3d(dx, dy, dz);
		}

		virtual ImplicitFunctionType type() const override
		{
			return ImplicitFunctionType::Cylinder;
		}

		std::shared_ptr<ImplicitFunction> clone() const override
		{
			return std::make_shared<IFCylinder2>(*this);
		}

	private:
		std::vector<double> _parameters;
	};

	struct IFBox2 : public ImplicitFunction
	{
		IFBox2(const std::string& name, const std::vector<double>& parameters) :
			ImplicitFunction(Eigen::Affine3d::Identity(), lmu::Mesh(), name),
			_parameters(parameters)
		{
		}

		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p) override
		{
			double d = distance(p);

			Eigen::Vector3d gradient = gradientPerCentralDifferences(p, 0.001);
			gradient.normalize();

			Eigen::Vector4d res;
			res << (d), gradient;

			return res;
		}

		inline double distance(const Eigen::Vector3d& p)
		{
			double xmin = _parameters[0];
			double xmax = _parameters[1];
			double ymin = _parameters[2];
			double ymax = _parameters[3];
			double zmin = _parameters[4];
			double zmax = _parameters[5];

			double xtmp = std::min(p.x() - xmin, xmax - p.x());
			double ytmp = std::min(p.y() - ymin, ymax - p.y());
			double ztmp = std::min(p.z() - zmin, zmax - p.z());

			return std::min(xtmp, std::min(ytmp, ztmp));
		}

		inline Eigen::Vector3d gradientPerCentralDifferences(const Eigen::Vector3d& ps, double h)
		{
			double dx = (distance(Eigen::Vector3d(ps.x() + h, ps.y(), ps.z())) - distance(Eigen::Vector3d(ps.x() - h, ps.y(), ps.z()))) / (2.0 * h);
			double dy = (distance(Eigen::Vector3d(ps.x(), ps.y() + h, ps.z())) - distance(Eigen::Vector3d(ps.x(), ps.y() - h, ps.z()))) / (2.0 * h);
			double dz = (distance(Eigen::Vector3d(ps.x(), ps.y(), ps.z() + h)) - distance(Eigen::Vector3d(ps.x(), ps.y(), ps.z() - h))) / (2.0 * h);

			return Eigen::Vector3d(dx, dy, dz);
		}

		virtual ImplicitFunctionType type() const override
		{
			return ImplicitFunctionType::Box;
		}

		std::shared_ptr<ImplicitFunction> clone() const override
		{
			return std::make_shared<IFBox2>(*this);
		}

	private:
		std::vector<double> _parameters;
	};

	std::vector<std::shared_ptr<ImplicitFunction>> fromFile(const std::string& file, double scaling = 1.0);
}

#endif