#ifndef TESTS_H
#define TESTS_H

#include "test.h"
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
	writeNode(mergedNode, "test.dot");

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
}

#endif