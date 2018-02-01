#ifndef TESTS_H
#define TESTS_H

#include "test.h"
#include "csgtree.h"

//TESTS 
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


#endif