#ifndef TESTS_H
#define TESTS_H

#include "test.h"
#include "csgtree.h"
#include "csgnode.h"
#include "csgnode_evo.h"
#include "csgnode_helper.h"
#include "evolution.h"

using namespace lmu;


std::unordered_map<std::string, ImplicitFunctionPtr> geometries(const std::vector<std::string>& names)
{
	std::unordered_map<std::string, ImplicitFunctionPtr> map;
	for (const auto& name : names)
		map[name] = std::make_shared<IFNull>(name);

	return map;
}

//TESTS 
TEST(CSGTreeInvalidTest)
{
	lmu::CSGTreeRanker ranker(0.0, {});

	lmu::CSGTree tree; 

	ASSERT_TRUE(ranker.treeIsInvalid(tree));

}

TEST(CSGTreeTest)
{
	lmu::CSGTree tree;

	ASSERT_TRUE(tree.depth() == 0);
	ASSERT_TRUE(tree.numNodes() == 1);
	ASSERT_TRUE(tree.node(123) == nullptr);

	tree = lmu::CSGTree( 
	{ 
		lmu::CSGTree({ lmu::CSGTree() ,lmu::CSGTree() }),
		lmu::CSGTree()
	});

	ASSERT_EQ(tree.depth(), 2);
	ASSERT_EQ(tree.numNodes(), 5);
	ASSERT_EQ(tree.node(123), nullptr);
	ASSERT_EQ(tree.node(3), &tree.childs[0].childs[1]);

	ASSERT_EQ(tree.sizeWithFunctions(), 5);
	tree.functions.push_back(nullptr);
	ASSERT_EQ(tree.sizeWithFunctions(), 6);

	ASSERT_EQ(tree.nodeDepth(3), 2);

	ASSERT_EQ(tree.nodeDepth(123), -1);

	//Clique Test
	auto if1 = std::make_shared<lmu::IFNull>("if1");
	auto if2 = std::make_shared<lmu::IFNull>("if2");
	auto if3 = std::make_shared<lmu::IFNull>("if3");
	auto if4 = std::make_shared<lmu::IFNull>("if4");
	auto if5 = std::make_shared<lmu::IFNull>("if5");

	std::vector<lmu::Clique> cliques =
	{
		lmu::Clique({if1}),
		lmu::Clique({if1, if2 }),
		lmu::Clique({if2, if3, if4}),
		lmu::Clique({if3, if1 }),
		lmu::Clique({if4, if5}),
		lmu::Clique({ if5 }),


	};

	lmu::createCSGTreeTemplateFromCliques(cliques).write("testres.dot");
}

TEST(CSGNodeTest)
{
	using namespace lmu;

	auto g = geometries({ "A", "B", "C", "D", "E" });

	CSGNode n1 =		
		opUnion(
		{
			opUnion(
			{
				opDiff(
				{
					geometry(g["A"]),
					geometry(g["B"])					
				}),
				opDiff(
				{
					geometry(g["B"]),
					geometry(g["A"])
				})
			}),
			geometry(g["C"])			
		});

	CSGNode n2 =		
		opDiff(
		{
			opUnion(
			{
				opDiff(
				{
					geometry(g["A"]),
					geometry(g["B"])					
				}),
				opDiff(
				{
					geometry(g["B"]),
					geometry(g["A"])
				})
			}),
			geometry(g["E"])			
		});

	CSGNode n3 =		
		opUnion(
		{
			geometry(g["D"]),
			geometry(g["B"])			
		});

	CSGNodeClique clique =
	{
		std::make_tuple(Clique(), n1),
		std::make_tuple(Clique(), n2),
		std::make_tuple(Clique(), n3)
	};

	auto mergedNode = mergeCSGNodeCliqueSimple(clique);
	
	std::stringstream ss;
	ss << serializeNode(mergedNode);
	
	ASSERT_EQ(ss.str(), "((((A)Difference(B))Union(((D)Union(B))Difference(A)))Difference(E))Union(C)");

	n1 =		
		opUnion(
		{
			opDiff(
			{
				geometry(g["B"]),
				geometry(g["D"])
			}),
			opInter(
			{
				geometry(g["D"]),
				geometry(g["C"])
			})
		});

	n2 =		
		opUnion(
		{
			opDiff(
			{
				geometry(g["A"]),
				geometry(g["C"])
			}),
			geometry(g["B"])		
		});


	clique =
	{
		std::make_tuple(Clique(), n1),
		std::make_tuple(Clique(), n2)
	};

	mergedNode = mergeCSGNodeCliqueSimple(clique);

	n1 =
		opUnion(
		{
			geometry(g["A"]),
			geometry(g["B"])
		});

	n2 =
		opUnion(
	{
		geometry(g["C"]),
		geometry(g["B"])
	});

	n3 =
		opDiff(
	{
		geometry(g["B"]),
		geometry(g["D"])
	});

	clique =
	{
		std::make_tuple(Clique(), n1),
		std::make_tuple(Clique(), n2),
		std::make_tuple(Clique(), n3)
	};

	mergedNode = mergeCSGNodeCliqueSimple(clique);

	writeNode(mergedNode, "test.dot");
}

#endif