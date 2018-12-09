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

		auto rotation = readVec3(params.at("rotation"));
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
