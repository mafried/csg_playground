#include "csgnode_helper.h"

#include <fstream>

using namespace lmu;

CSGNode lmu::geometry(ImplicitFunctionPtr function)
{	
	return CSGNode(std::make_shared<CSGNodeGeometry>(function));
}
CSGNode lmu::opUnion(const std::vector<CSGNode>& childs)
{
	return CSGNode(std::make_shared<UnionOperation>("", childs));
}
CSGNode lmu::opDiff(const std::vector<CSGNode>& childs)
{
	return CSGNode(std::make_shared<DifferenceOperation>("", childs));
}
CSGNode lmu::opInter(const std::vector<CSGNode>& childs)
{
	return CSGNode(std::make_shared<IntersectionOperation>("", childs));
}
CSGNode lmu::opComp(const std::vector<CSGNode>& childs)
{
	return CSGNode(std::make_shared<ComplementOperation>("", childs));
}
CSGNode lmu::opNo(const std::vector<CSGNode>& childs)
{
	return CSGNode(std::make_shared<NoOperation>("", childs));
}

Eigen::Vector3d readVec3(const json& json)
{
	return Eigen::Vector3d(
		json.at(0).get<double>(),
		json.at(1).get<double>(),
		json.at(2).get<double>());
}

lmu::CSGNode readNode(const json& json)
{
	auto opTypeIt = json.find("op"); 
	if (opTypeIt != json.cend())
	{
		auto opType = operationTypeFromString(opTypeIt->get<std::string>());
		auto node = createOperation(opType);		
		auto childs = json.at("childs");
		for (json::iterator it = childs.begin(); it != childs.end(); ++it) 
		{
			node.addChild(readNode(*it));
		}

		return node;
	}
	else
	{
		auto geoTypeIt = json.find("geo");
		if (geoTypeIt != json.cend())
		{
			auto geoType = geoTypeIt->get<std::string>();
			std::transform(geoType.begin(), geoType.end(), geoType.begin(), ::tolower);

			auto name = json.at("name").get<std::string>();

			auto params = json.at("params");

			auto center = readVec3(params.at("center"));

			auto rotIter = params.find("rotation");
			auto rotation = rotIter == params.end() ? Eigen::Vector3d(0, 0, 0) : readVec3(*rotIter);
			Eigen::AngleAxisd rotx(rotation.x(), Eigen::Vector3d(1.0, 0.0, 0.0));
			Eigen::AngleAxisd roty(rotation.y(), Eigen::Vector3d(0.0, 1.0, 0.0));
			Eigen::AngleAxisd rotz(rotation.z(), Eigen::Vector3d(0.0, 0.0, 1.0));

			auto trans =
				(Eigen::Affine3d)Eigen::Translation3d(center.x(), center.y(), center.z()) * rotx * roty * rotz;

			if (geoType == "sphere")
			{
				return geo<IFSphere>(
					trans,
					params.at("radius").get<double>(),
					name);
			}
			else if (geoType == "box" || geoType == "cube")
			{
				return geo<IFBox>(
					trans,
					readVec3(params.at("radius")) * 2.0,
					2,
					name);
			}
			else if (geoType == "cylinder")
			{
				return geo<IFCylinder>(
					trans,
					params.at("radius").get<double>(),
					params.at("height").get<double>(),
					name);
			}
		}
		else
		{
			std::cout << json;
		}
	}

	return CSGNode::invalidNode;
}

CSGNode lmu::fromJSON(const json& json)
{
	return readNode(json);
}

CSGNode lmu::fromJSONFile(const std::string& file)
{
	std::ifstream str(file);
	json json;
	str >> json;

	return fromJSON(json);
}

void lmu::toJSONFile(const CSGNode& node, const std::string& file)
{
	std::ofstream str(file);
	json json = toJSON(node);
	str << json;
	str.close();
}

lmu::json lmu::toJSON(const lmu::CSGNode& node)
{
	lmu::json json;

	switch (node.type())
	{
	case CSGNodeType::Geometry:
	{
		switch (node.function()->type())
		{
		case ImplicitFunctionType::Cylinder:
			json["geo"] = "cylinder";
			json["params"]["radius"] = dynamic_cast<IFCylinder*>(node.function().get())->radius();
			json["params"]["height"] = dynamic_cast<IFCylinder*>(node.function().get())->height();

			break;
		case ImplicitFunctionType::Sphere:
			json["geo"] = "sphere";
			json["params"]["radius"] = dynamic_cast<IFSphere*>(node.function().get())->radius();
			break;
		case ImplicitFunctionType::Box:
			json["geo"] = "cube";
			json["params"]["radius"] = json::array();
			json["params"]["radius"].push_back(dynamic_cast<IFBox*>(node.function().get())->size().x() / 2.0);
			json["params"]["radius"].push_back(dynamic_cast<IFBox*>(node.function().get())->size().y() / 2.0);
			json["params"]["radius"].push_back(dynamic_cast<IFBox*>(node.function().get())->size().z() / 2.0);
			break;
		}
		
		//std::cout << "NAME: " << node.function()->name() << std::endl;
		json["name"] = node.function()->name();

		auto pos = node.function()->pos();
		json["params"]["center"] = { pos.x(), pos.y(), pos.z() };
		
		auto rot = node.function()->transform().matrix().block<3,3>(0,0).eulerAngles(0, 1, 2);
		json["params"]["rotation"] = { rot.x(), rot.y(), rot.z() };
	
		break;
	}
	case CSGNodeType::Operation:
	{
		switch (node.operationType())
		{
		case CSGNodeOperationType::Union:
			json["op"] = "union";
			break;
		case CSGNodeOperationType::Intersection:
			json["op"] = "intersect";
			break;
		case CSGNodeOperationType::Difference:
			json["op"] = "subtract";
			break;
		case CSGNodeOperationType::Complement:
			json["op"] = "negate";
			break;
		}

		for (const auto& c : node.childsCRef())
			json["childs"].push_back(toJSON(c));

		break;
	}
	}

	return json;
}

lmu::AABB lmu::aabb_from_node(const lmu::CSGNode& n)
{
	return aabb_from_primitives(lmu::allDistinctFunctions(n));
}

lmu::AABB lmu::aabb_from_primitives(const std::vector<lmu::ImplicitFunctionPtr>& prims)
{
	lmu::AABB aabb;
	for (const auto& p : prims)
	{
		aabb = aabb.setunion(p->aabb());
	}
	return aabb;
}

bool lmu::_is_empty_set(const lmu::CSGNode& n, double sampling_grid_size,
	const Eigen::Vector3d& min, const Eigen::Vector3d& max)
{
	if ((max - min).cwiseAbs().minCoeff() <= sampling_grid_size)
		return true;

	//std::cout << max.transpose() << " " << min.transpose() << std::endl;
	
	const Eigen::Vector3d s = (max - min);
	const Eigen::Vector3d p = min + s * 0.5;
	const double d = n.signedDistance(p);
	if (d < 0.0)
		return false;

	//if (4.0 * d * d > s.squaredNorm())
	//	return true;

	const Eigen::Vector3d sh = s * 0.5;
	return
		_is_empty_set(n, sampling_grid_size, min + Eigen::Vector3d(0, 0, 0), min + sh) &&
		_is_empty_set(n, sampling_grid_size, min + Eigen::Vector3d(sh.x(), 0, 0), min + Eigen::Vector3d(s.x(), sh.y(), sh.z())) &&
		_is_empty_set(n, sampling_grid_size, min + Eigen::Vector3d(sh.x(), 0, sh.z()), max) &&
		_is_empty_set(n, sampling_grid_size, min + Eigen::Vector3d(0, 0, sh.z()), min + Eigen::Vector3d(sh.x(), sh.y(), s.z())) &&

		_is_empty_set(n, sampling_grid_size, min + Eigen::Vector3d(0, sh.y(), 0), min + Eigen::Vector3d(sh.x(), s.y(), s.z())) &&
		_is_empty_set(n, sampling_grid_size, min + Eigen::Vector3d(sh.x(), sh.y(), 0), min + Eigen::Vector3d(s.x(), s.y(), sh.z())) &&
		_is_empty_set(n, sampling_grid_size, min + Eigen::Vector3d(sh.x(), sh.y(), sh.z()), max) &&
		_is_empty_set(n, sampling_grid_size, min + Eigen::Vector3d(0, sh.y(), sh.z()), min + Eigen::Vector3d(sh.x(), s.y(), s.z()));
}

bool lmu::_is_empty_set(const lmu::CSGNode & n, const lmu::PointCloud& sp)
{
	for (int i = 0; i < sp.rows(); ++i)
	{
		Eigen::Vector3d p = sp.row(i).leftCols(3).transpose();
		if (n.signedDistance(p) < 0.0)
			return false;
	}
	return true;
}

bool lmu::_is_in(const lmu::ImplicitFunctionPtr& primitive, const lmu::PointCloud& in_out, const CSGNode& node)
{
	for (int i = 0; i < in_out.rows(); ++i)
	{
		Eigen::Vector3d p = in_out.row(i).leftCols(3).transpose();
		auto pd = primitive->signedDistance(p);
		auto nd = node.signedDistance(p);

		if (nd >= 0.0 && pd < 0.0)
			return false;
	}

	return true;
}

bool lmu::_is_out(const lmu::ImplicitFunctionPtr& primitive, const lmu::PointCloud& in_out, const CSGNode& node)
{
	for (int i = 0; i < in_out.rows(); ++i)
	{
		Eigen::Vector3d p = in_out.row(i).leftCols(3).transpose();
		auto pd = primitive->signedDistance(p);
		auto nd = node.signedDistance(p);

		if (nd < 0.0 && pd < 0.0)
			return false;
	}

	return true;
}

bool lmu::_is_in(const lmu::ImplicitFunctionPtr& primitive, const lmu::CSGNode& n, double sampling_grid_size, const Eigen::Vector3d& min, const Eigen::Vector3d& max)
{
	if ((max - min).cwiseAbs().minCoeff() <= sampling_grid_size)
		return true;

	const Eigen::Vector3d s = (max - min);
	const Eigen::Vector3d p = min + s * 0.5;
	
	if (primitive->signedDistance(p) <= 0.0 && n.signedDistance(p) > 0.0)
	{
		return false;
	}
	
	const Eigen::Vector3d sh = s * 0.5;
	return
		_is_in(primitive, n, sampling_grid_size, min + Eigen::Vector3d(0, 0, 0), min + sh) &&
		_is_in(primitive, n, sampling_grid_size, min + Eigen::Vector3d(sh.x(), 0, 0), min + Eigen::Vector3d(s.x(), sh.y(), sh.z())) &&
		_is_in(primitive, n, sampling_grid_size, min + Eigen::Vector3d(sh.x(), 0, sh.z()), max) &&
		_is_in(primitive, n, sampling_grid_size, min + Eigen::Vector3d(0, 0, sh.z()), min + Eigen::Vector3d(sh.x(), sh.y(), s.z())) &&

		_is_in(primitive, n, sampling_grid_size, min + Eigen::Vector3d(0, sh.y(), 0), min + Eigen::Vector3d(sh.x(), s.y(), s.z())) &&
		_is_in(primitive, n, sampling_grid_size, min + Eigen::Vector3d(sh.x(), sh.y(), 0), min + Eigen::Vector3d(s.x(), s.y(), sh.z())) &&
		_is_in(primitive, n, sampling_grid_size, min + Eigen::Vector3d(sh.x(), sh.y(), sh.z()), max) &&
		_is_in(primitive, n, sampling_grid_size, min + Eigen::Vector3d(0, sh.y(), sh.z()), min + Eigen::Vector3d(sh.x(), s.y(), s.z()));
}
