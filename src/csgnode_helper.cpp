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
		auto geoType = json.at("geo").get<std::string>();
		std::transform(geoType.begin(), geoType.end(), geoType.begin(), ::tolower);
		
		auto name = json.at("name").get<std::string>();

		auto params = json.at("params");

		auto center = readVec3(params.at("center"));

		auto rotIter = params.find("rotation");
		auto rotation = rotIter == params.end() ? Eigen::Vector3d(0,0,0) : readVec3(*rotIter);
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

lmu::AABB lmu::aabb_from_node(const lmu::CSGNode & n)
{
	lmu::AABB aabb;
	auto prims = lmu::allDistinctFunctions(n);
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
