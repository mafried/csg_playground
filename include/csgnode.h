#ifndef CSGNODE_H
#define CSGNODE_H

#include <vector>
#include <memory>
#include <unordered_map>

#include "helper.h"

#include "mesh.h"

#include <Eigen/Core>

namespace lmu
{
	class ICSGNode;
	using CSGNodePtr = std::shared_ptr<ICSGNode>;
	using ImplicitFunctionPtr = std::shared_ptr<ImplicitFunction>;

	enum class CSGNodeType
	{
		Operation,
		Geometry
	};

	enum class CSGNodeOperationType
	{
		Unknown = 0,
		Intersection,
		Union,
		Difference,
		Complement,
		Invalid
	};

	std::string operationTypeToString(CSGNodeOperationType type);
	std::string nodeTypeToString(CSGNodeType type);
	
	class CSGNode;

	class ICSGNode
	{
	public: 

		virtual CSGNodePtr clone() const = 0;

		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p) const = 0;

		virtual std::string name() const = 0; 

		virtual CSGNodeType type() const = 0;

		virtual CSGNodeOperationType operationType() const = 0;

		virtual std::vector<CSGNode> childs() const = 0;
		virtual const std::vector<CSGNode>& childsCRef() const = 0;
		virtual std::vector<CSGNode>& childsRef() = 0;
		virtual bool addChild(const CSGNode& child) = 0;

		virtual std::tuple<int,int> numAllowedChilds() const = 0;

		virtual ImplicitFunctionPtr function() const = 0;

		virtual Mesh mesh() const = 0;
	};

	class CSGNodeBase : public ICSGNode 
	{
	public: 
	
		CSGNodeBase(const std::string& name, CSGNodeType type) : 
			_name(name),
			_type(type)
		{
		}

		virtual std::string name() const override
		{
			return _name;
		}

		virtual CSGNodeType type() const override
		{
			return _type;
		}
	protected: 
		std::string _name;
		CSGNodeType _type;

		//CSGNodeBase(CSGNodeBase const& other) :
		//	_name(other._name),
		//	_type(other._type)
		//{
		//}
		//void operator=(CSGNodeBase const &t) = delete;
		//CSGNodeBase(CSGNodeBase &&) = delete;

	};

	class CSGNodeOperation : public CSGNodeBase
	{
	public:
		CSGNodeOperation(const std::string& name, const std::vector<CSGNode>& childs = {}) :
			CSGNodeBase(name, CSGNodeType::Operation),
			_childs(childs)
		{	
		}

		virtual ImplicitFunctionPtr function() const override 
		{
			return nullptr;
		}

		virtual const std::vector<CSGNode>& childsCRef() const override
		{
			return _childs;
		}

		virtual std::vector<CSGNode>& childsRef() override
		{
			return _childs;
		}

		virtual std::vector<CSGNode> childs() const override
		{
			return _childs;
		}

		virtual bool addChild(const CSGNode& child) override
		{
			if (_childs.size() >= std::get<1>(numAllowedChilds()))
				return false; 

			_childs.push_back(child);

			return true;
		}

	protected: 
		std::vector<CSGNode> _childs;
	};

	class CSGNodeGeometry : public CSGNodeBase
	{
	public:
		explicit CSGNodeGeometry(ImplicitFunctionPtr function) :
			CSGNodeBase(function->name(), CSGNodeType::Geometry),
			_function(function)
		{
		}

		virtual ImplicitFunctionPtr function() const override
		{
			return _function;
		}

		virtual CSGNodePtr clone() const override 
		{
			return std::make_shared<CSGNodeGeometry>(_function/*->clone() <= We don't clone the function since its reference is later used as its id*/);
		}

		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p) const override
		{
			return _function->signedDistanceAndGradient(p);
		}

		virtual std::vector<CSGNode> childs() const override
		{
			return _childs;
		}

		virtual const std::vector<CSGNode>& childsCRef() const override
		{
			return _childs;
		}

		virtual std::vector<CSGNode>& childsRef() override
		{
			return _childs;
		}

		virtual bool addChild(const CSGNode& child) override
		{
			return false;
		}

		virtual std::tuple<int, int> numAllowedChilds() const override
		{
			return std::make_tuple(0, 0);
		}

		virtual CSGNodeOperationType operationType() const override
		{
			return CSGNodeOperationType::Invalid;
		}

		virtual Mesh mesh() const override
		{
			return _function->meshCRef();
		}

	protected:
		ImplicitFunctionPtr _function;
	private:
		std::vector<CSGNode> _childs; //always empty.
	};

	
	class CSGNode : public ICSGNode 
	{
	public:

		explicit CSGNode(CSGNodePtr node) :
			_node(node)
		{
		}

		CSGNode(const CSGNode& node) :
			_node(node.clone())
		{
			//std::cout << "Copy" << std::endl;
		}

		CSGNode& operator = (const CSGNode& other)
		{
			if (this != &other) 
				_node = other.clone();

			return *this;
		}

		virtual CSGNodePtr clone() const override
		{
			return _node->clone();
		}

		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p) const override
		{
			return _node->signedDistanceAndGradient(p);
		}

		virtual std::string name() const override
		{
			return _node->name(); 
		}

		virtual CSGNodeType type() const override
		{
			return _node->type();
		}

		virtual CSGNodeOperationType operationType() const override
		{
			return _node->operationType();
		}

		virtual std::vector<CSGNode> childs() const override
		{
			return _node->childs();
		}

		virtual bool addChild(const CSGNode& child) override
		{
			return _node->addChild(child);
		}

		virtual std::tuple<int,int> numAllowedChilds() const override
		{
			return _node->numAllowedChilds();
		}

		virtual ImplicitFunctionPtr function() const override
		{
			return _node->function();
		}

		CSGNodePtr nodePtr() const
		{
			return _node;
		}

		virtual Mesh mesh() const override
		{
			return _node->mesh();
		}

		virtual const std::vector<CSGNode>& childsCRef() const override
		{
			return _node->childsCRef();
		}

		virtual std::vector<CSGNode>& childsRef() override
		{
			return _node->childsRef();
		}

	private: 
		CSGNodePtr _node;
	};

	class UnionOperation : public CSGNodeOperation
	{
	public:
		UnionOperation(const std::string& name, const std::vector<CSGNode>& childs = {}) :
			CSGNodeOperation(name, childs)
		{
		}

		virtual CSGNodePtr clone() const override;
		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p) const override;
		virtual CSGNodeOperationType operationType() const override;
		virtual std::tuple<int, int> numAllowedChilds() const override;
		virtual Mesh mesh() const override;
	};

	class IntersectionOperation : public CSGNodeOperation
	{
	public:
		IntersectionOperation(const std::string& name, const std::vector<CSGNode>& childs = {}) :
			CSGNodeOperation(name, childs)
		{
		}

		virtual CSGNodePtr clone() const override;
		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p) const override;
		virtual CSGNodeOperationType operationType() const override;
		virtual std::tuple<int, int> numAllowedChilds() const override;
		virtual Mesh mesh() const override;
	};

	class DifferenceOperation : public CSGNodeOperation
	{
	public:
		DifferenceOperation(const std::string& name, const std::vector<CSGNode>& childs = {}) :
			CSGNodeOperation(name, childs)
		{
		}

		virtual CSGNodePtr clone() const override;
		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p) const override;
		virtual CSGNodeOperationType operationType() const override;
		virtual std::tuple<int, int> numAllowedChilds() const override;
		virtual Mesh mesh() const override;
	};

	/*class DifferenceRLOperation : public CSGNodeOperation
	{
	public:
		DifferenceRLOperation(const std::string& name, const std::vector<CSGNode>& childs = {}) :
			CSGNodeOperation(name, childs)
		{
		}

		virtual CSGNodePtr clone() const override;
		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p) const override;
		virtual CSGNodeOperationType operationType() const override;
		virtual std::tuple<int, int> numAllowedChilds() const override;
		virtual Mesh mesh() const override;
	};*/

	// =================================================================================== Methods ===================================================================================

	CSGNode createOperation(CSGNodeOperationType type, const std::string& name = std::string(), const std::vector<CSGNode>& childs = {});

	int depth(const CSGNode& node, int curDepth = 0);
	int numNodes(const CSGNode& node);
	int numPoints(const CSGNode& node);
	CSGNode* nodePtrAt(CSGNode& node, int idx);
	int depthAt(const CSGNode& node, int idx);	
	std::vector<CSGNodePtr> allGeometryNodePtrs(const CSGNode& node);

	double computeGeometryScore(const CSGNode& node, double epsilon, double alpha, const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& funcs);

	void writeNode(const CSGNode& node, const std::string& file);

	enum class NodePartType
	{
		LeftBracket, 
		RightBracket,
		Node
	};

	struct NodePart
	{
		NodePart(NodePartType type, CSGNode* node) :
			type(type),
			node(node)
		{
		}
		NodePartType type;
		CSGNode* node;	
		friend std::ostream& operator<<(std::ostream& os, const NodePart& np);
	};

	std::ostream& operator<<(std::ostream& os, const NodePart& np);

	using SerializedCSGNode = std::vector<NodePart>;

	std::ostream& operator<<(std::ostream& os, const SerializedCSGNode& v);
	bool operator==(const NodePart& lhs, const NodePart& rhs);
	bool operator!=(const NodePart& lhs, const NodePart& rhs);

	SerializedCSGNode serializeNode(CSGNode& node);

	struct LargestCommonSubgraph
	{
		LargestCommonSubgraph(CSGNode* n1Root, CSGNode* n2Root, const std::vector<CSGNode*>& n1Appearances, const std::vector<CSGNode*>& n2Appearances, int size) :
			n1Root(n1Root),
			n2Root(n2Root),
			n1Appearances(n1Appearances),
			n2Appearances(n2Appearances),
			size(size)
		{
		}

		bool isEmptyOrInvalid() const
		{
			return size == 0 || n1Appearances.empty() || n2Appearances.empty();
		}

		CSGNode* n1Root;
		CSGNode* n2Root;

		std::vector<CSGNode*> n1Appearances;
		std::vector<CSGNode*> n2Appearances;

		int size;
	};

	LargestCommonSubgraph findLargestCommonSubgraph(CSGNode& n1, CSGNode& n2);

	enum class MergeResult
	{
		First,
		Second,
		None
	};

	MergeResult mergeNodes(const LargestCommonSubgraph& lcs);



	Mesh computeMesh(const CSGNode& node, const Eigen::Vector3i& numSamples, const Eigen::Vector3d& min = Eigen::Vector3d(0.0, 0.0, 0.0), 
		const Eigen::Vector3d& max = Eigen::Vector3d(0.0, 0.0, 0.0));

	Eigen::MatrixXd computePointCloud(const CSGNode& node, const Eigen::Vector3i& numSamples, double maxDistance, double errorSigma,
		const Eigen::Vector3d& min = Eigen::Vector3d(0.0, 0.0, 0.0), const Eigen::Vector3d& max = Eigen::Vector3d(0.0, 0.0, 0.0));

	Eigen::VectorXd computeDistanceError(const Eigen::MatrixXd& samplePoints, const CSGNode& referenceNode, const CSGNode& node, bool normalize);

}

#endif