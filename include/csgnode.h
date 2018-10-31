#ifndef CSGNODE_H
#define CSGNODE_H

#include <vector>
#include <memory>
#include <unordered_map>

#include "helper.h"

#include "mesh.h"

#include <Eigen/Core>
#include <boost/any.hpp>

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
		Identity,
		Invalid
	};

	std::string operationTypeToString(CSGNodeOperationType type);
	std::string nodeTypeToString(CSGNodeType type);
	
	class CSGNode;

	class ICSGNode
	{
	public: 		
		using Attributes = std::unordered_map<std::string, boost::any>;
	
		virtual Attributes& attributesRef() = 0;
		virtual Attributes attributes() const = 0;

		virtual CSGNodePtr clone() const = 0;

		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p, double h = 0.001) const = 0;
		virtual double signedDistance(const Eigen::Vector3d& p) const = 0;

		virtual std::string name() const = 0; 

		virtual CSGNodeType type() const = 0;

		virtual CSGNodeOperationType operationType() const = 0;

		virtual std::vector<CSGNode> childs() const = 0;
		virtual const std::vector<CSGNode>& childsCRef() const = 0;
		virtual std::vector<CSGNode>& childsRef() = 0;
		virtual bool addChild(const CSGNode& child) = 0;

		virtual std::tuple<int,int> numAllowedChilds() const = 0;

		virtual ImplicitFunctionPtr function() const = 0;
		virtual void setFunction(const ImplicitFunctionPtr& f) = 0;

		virtual size_t hash(size_t seed) const = 0;

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

		virtual Attributes& attributesRef() override
		{
			return _attr;
		}

		virtual Attributes attributes() const override
		{
			return _attr;
		}

	protected: 
		std::string _name;
		CSGNodeType _type;
		ICSGNode::Attributes _attr;
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

		virtual void setFunction(const ImplicitFunctionPtr& f) override
		{
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

		virtual size_t hash(size_t seed) const override;

	protected: 
		std::vector<CSGNode> _childs;
	};

	class CSGNodeGeometry : public CSGNodeBase
	{
	public:
		explicit CSGNodeGeometry(ImplicitFunctionPtr function) :
			CSGNodeBase(function ? function->name() : "NullFunction" , CSGNodeType::Geometry),
			_function(function)
		{
		}

		virtual ImplicitFunctionPtr function() const override
		{
			return _function;
		}

		virtual void setFunction(const ImplicitFunctionPtr& f) override
		{
			_function = f;
			_name = f->name();
		}

		virtual CSGNodePtr clone() const override 
		{
			return std::make_shared<CSGNodeGeometry>(_function/*->clone() <= We don't clone the function since its reference is later used as its id*/);
		}

		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p, double h = 0.001) const override
		{
			return _function->signedDistanceAndGradient(p, h);
		}

		virtual double signedDistance(const Eigen::Vector3d& p) const override
		{
			return _function->signedDistance(p);
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

		virtual size_t hash(size_t seed) const override;

	protected:
		ImplicitFunctionPtr _function;
	private:
		std::vector<CSGNode> _childs; //always empty.
	};

	
	// =================================================================================== Standard Methods ===================================================================================

	class CSGNode;

	int depth(const CSGNode& node, int curDepth = 0);
	int numNodes(const CSGNode& node);
	int numPoints(const CSGNode& node);
	CSGNode* nodePtrAt(CSGNode& node, int idx);
	int depthAt(const CSGNode& node, int idx);
	std::vector<CSGNodePtr> allGeometryNodePtrs(const CSGNode& node);
	std::vector<ImplicitFunctionPtr> allDistinctFunctions(const CSGNode& node);

	void visit(const CSGNode& node, const std::function<void(const CSGNode& node)>& f);
	void visit(CSGNode& node, const std::function<void(CSGNode& node)>& f);

	class CSGNode : public ICSGNode 
	{
	public:

		template<typename T> 
		T attribute(const std::string& name) const 
		{
			auto it = _node->attributesRef().find(name);
			if (it == _node->attributesRef().end())
				return T();
			else
			{
				return boost::any_cast<T>(it->second);
			}
		}

		template<typename T>
		void setAttribute(const std::string& name, const T& value)
		{
			_node->attributesRef()[name] = value;
		}

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

		inline virtual CSGNodePtr clone() const override final
		{
			return _node ? _node->clone() : nullptr;
		}

		inline virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p, double h = 0.001) const override final
		{
			return _node->signedDistanceAndGradient(p, h);
		}

		inline virtual double signedDistance(const Eigen::Vector3d& p) const override final
		{
			return _node->signedDistance(p);
		}

		inline virtual std::string name() const override final
		{
			return _node->name(); 
		}

		inline virtual CSGNodeType type() const override final
		{
			return _node->type();
		}

		inline virtual CSGNodeOperationType operationType() const override final
		{
			return _node->operationType();
		}

		inline virtual std::vector<CSGNode> childs() const override final
		{
			return _node->childs();
		}

		inline virtual bool addChild(const CSGNode& child) override final
		{
			return _node->addChild(child);
		}

		inline virtual std::tuple<int,int> numAllowedChilds() const override final
		{
			return _node->numAllowedChilds();
		}

		inline virtual ImplicitFunctionPtr function() const override final
		{
			return _node->function();
		}

		inline virtual void setFunction(const ImplicitFunctionPtr& f) override final
		{
			_node->setFunction(f);
		}

		inline virtual Mesh mesh() const override final
		{
			return _node->mesh();
		}

		inline virtual const std::vector<CSGNode>& childsCRef() const override final
		{
			return _node->childsCRef();
		}

		inline virtual std::vector<CSGNode>& childsRef() override final
		{
			return _node->childsRef();
		}

		virtual Attributes& attributesRef() override
		{
			return _node->attributesRef();
		}

		virtual Attributes attributes() const override
		{
			return _node->attributes();
		}

		CSGNodePtr nodePtr() const
		{
			return _node;
		}

		std::string info() const 
		{
			std::stringstream ss;

			auto functions = allGeometryNodePtrs(*this);

			ss << "# Num nodes: " << numNodes(*this) << std::endl;
			ss << "# Depth: " << depth(*this) << std::endl;
			ss << "# Functions: " << functions.size() << std::endl;
			int totalNumPoints = 0;
			for (const auto& f : functions)
			{
				int numPoints = f->function()->pointsCRef().rows();
				totalNumPoints += numPoints;
				ss << "#    function '" << f->name() << "' type: " << iFTypeToString(f->function()->type()) << " #points: " << numPoints << std::endl;
			}
			ss << "# Num points: " << totalNumPoints << std::endl;

			return ss.str();
		}

		virtual size_t hash(size_t seed) const override
		{
			return _node->hash(seed);
		}

		bool isValid() const
		{
			return _node != nullptr;
		}

		static const CSGNode invalidNode;

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
		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p, double h = 0.001) const override;
		virtual double signedDistance(const Eigen::Vector3d& p) const override;
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
		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p, double h = 0.001) const override;
		virtual double signedDistance(const Eigen::Vector3d& p) const override;
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
		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p, double h = 0.001) const override;
		virtual double signedDistance(const Eigen::Vector3d& p) const override;
		virtual CSGNodeOperationType operationType() const override;
		virtual std::tuple<int, int> numAllowedChilds() const override;
		virtual Mesh mesh() const override;
	};

	class ComplementOperation : public CSGNodeOperation
	{
	public:
		ComplementOperation(const std::string& name, const std::vector<CSGNode>& childs = {}) :
			CSGNodeOperation(name, childs)
		{
		}

		virtual CSGNodePtr clone() const override;
		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p, double h = 0.001) const override;
		virtual double signedDistance(const Eigen::Vector3d& p) const override;
		virtual CSGNodeOperationType operationType() const override;
		virtual std::tuple<int, int> numAllowedChilds() const override;
		virtual Mesh mesh() const override;
	};

	class IdentityOperation : public CSGNodeOperation
	{
	public:
		IdentityOperation(const std::string& name, const std::vector<CSGNode>& childs = {}) :
			CSGNodeOperation(name, childs)
		{
		}

		virtual CSGNodePtr clone() const override;
		virtual Eigen::Vector4d signedDistanceAndGradient(const Eigen::Vector3d& p, double h = 0.001) const override;
		virtual double signedDistance(const Eigen::Vector3d& p) const override;
		virtual CSGNodeOperationType operationType() const override;
		virtual std::tuple<int, int> numAllowedChilds() const override;
		virtual Mesh mesh() const override;
	};
	
	CSGNode createOperation(CSGNodeOperationType type, const std::string& name = std::string(), const std::vector<CSGNode>& childs = {});

	double computeGeometryScore(const CSGNode& node, double epsilon, double alpha, double h, const std::vector<std::shared_ptr<lmu::ImplicitFunction>>& funcs);

	double computeRawDistanceScore(const CSGNode& node, const Eigen::MatrixXd& points);
	
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

	struct CommonSubgraph
	{
		CommonSubgraph(CSGNode* n1Root, CSGNode* n2Root, const std::vector<CSGNode*>& n1Appearances, const std::vector<CSGNode*>& n2Appearances, int size) :
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

	CommonSubgraph findLargestCommonSubgraph(CSGNode& n1, CSGNode& n2, const std::vector<std::string>& blackList = {});

	std::vector<CommonSubgraph> findCommonSubgraphs(CSGNode & n1, CSGNode & n2);

	enum class MergeResult
	{
		First,
		Second,
		None
	};

	MergeResult mergeNodes(const CommonSubgraph& lcs, bool allowIntersections);
	
	Mesh computeMesh(const CSGNode& node, const Eigen::Vector3i& numSamples, const Eigen::Vector3d& min = Eigen::Vector3d(0.0, 0.0, 0.0), 
		const Eigen::Vector3d& max = Eigen::Vector3d(0.0, 0.0, 0.0));
	
	int optimizeCSGNodeStructure(CSGNode& node);

	void optimizeCSGNode(CSGNode& node, double tolerance);

	void convertToTreeWithMaxNChilds(CSGNode& node, int n);

	struct CSGNodeSamplingParams
	{
		CSGNodeSamplingParams(double maxDistance, double maxAngleDistance, double errorSigma, double samplingStepSize = 0.0,
			const Eigen::Vector3d& min = Eigen::Vector3d(0.0, 0.0, 0.0), const Eigen::Vector3d& max = Eigen::Vector3d(0.0, 0.0, 0.0));

		double samplingStepSize; 
		double maxDistance; 
		double maxAngleDistance;
		double errorSigma;
		Eigen::Vector3d minDim;
		Eigen::Vector3d maxDim;
	};

	PointCloud computePointCloud(const CSGNode& node, const CSGNodeSamplingParams& params);

	Eigen::VectorXd computeDistanceError(const Eigen::MatrixXd& samplePoints, const CSGNode& referenceNode, const CSGNode& node, bool normalize);

	std::tuple<Eigen::Vector3d, Eigen::Vector3d> computeDimensions(const std::vector<std::shared_ptr<ImplicitFunction>>& geos);
	std::tuple<Eigen::Vector3d, Eigen::Vector3d> computeDimensions(const CSGNode& node);

	CSGNode* findSmallestSubgraphWithImplicitFunctions(CSGNode& node, const std::vector<ImplicitFunctionPtr>& funcs);

}

#endif