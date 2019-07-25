#ifndef MERGE_HEADER
#define MERGE_HEADER

#ifndef DLL_LINKAGE
#define DLL_LINKAGE
#endif

#include <vector>
#include <utility>
#include <PrimitiveShape.h>

typedef std::vector< std::pair< MiscLib::RefCountPtr< PrimitiveShape >, size_t > > ShapeVector;
typedef MiscLib::RefCountPtr<PrimitiveShape> Primitive;


std::vector<Primitive>
MergeSimilarPrimitives(ShapeVector& primitives, float dist_thresh, float dot_thresh, float angle_thresh);

void
MergeSimilarPrimitives(std::vector<Primitive>& primitives, std::vector<PointCloud>& pointClouds,
	float dist_thresh, float dot_thresh, float angle_thresh,
	std::vector<Primitive>& mergedPrimitives, std::vector<PointCloud>& mergedPointClouds);

void SplitPointsPrimitives(const ShapeVector& shapes, const PointCloud& pc,
	std::vector<Primitive>& primitives, std::vector<PointCloud>& pointClouds);

#endif
