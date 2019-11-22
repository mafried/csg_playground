#ifndef OPTIMIZER_TEST_H
#define OPTIMIZER_TEST_H

#include "test.h"

#include "csgnode_helper.h"
#include "optimizer_red.h"

using namespace lmu;

CSGNode sphere(double x, double y, double z, double r, const std::string& name = "")
{
	return geo<IFSphere>((Eigen::Affine3d)(Eigen::Translation3d(x, y, z)), r, name);
}

TEST(OptimizerRedundancyTest)
{
	const double sampling = 0.01;
	EmptySetLookup esl;

	//s1 does overlap with s2, s3 does neither overlap with s1 nor with s2.
	auto s1 = sphere(0, 0, 0, 1);
	auto s2 = sphere(1, 0, 0, 1, "s2");
	auto s3 = sphere(3, 0, 0, 1);

	ASSERT_TRUE(is_empty_set(opInter({ s1, s3 }), sampling, esl));
	ASSERT_TRUE(!is_empty_set(opInter({ s1, s2 }), sampling, esl));
	
	auto node_with_redun = opUnion({ s2, opInter({ s1, s3 }) });
	auto node_without_redun = remove_redundancies(node_with_redun, sampling);
	ASSERT_TRUE(
		numNodes(node_without_redun) == 1,
		node_without_redun.type() == CSGNodeType::Geometry,
		node_without_redun.name() == "s2"
	);
}

#endif
