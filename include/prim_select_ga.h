#ifndef PRIM_SELECT_GA_H
#define PRIM_SELECT_GA_H

#include <vector>
#include <memory>

#include "csgnode.h"
#include "evolution.h"

#include <Eigen/Core>

#include "primitive_extraction.h"

namespace lmu
{
	struct SelectionValue
	{
		SelectionValue(DHType dh_type, bool active);

		DHType dh_type;
		bool active;

		size_t hash(size_t seed) const;
	};

	struct PrimitiveSelection
	{
		PrimitiveSelection(const PrimitiveSet* primitives);
		PrimitiveSelection(const PrimitiveSet* primitives, const std::vector<DHType>& dh_types);
		
		const PrimitiveSet* prims;
		std::vector<SelectionValue> selection;

		CSGNode to_node() const;

		int get_num_active() const;

		size_t hash(size_t seed) const;

	};

	std::ostream& operator<<(std::ostream& out, const PrimitiveSelection& r);


	struct SelectionRank
	{
		explicit SelectionRank(double v = 0.0);

		SelectionRank(double geo, double size, double combined);

		double geo;
		double size;
		double combined;

		friend inline bool operator< (const SelectionRank& lhs, const SelectionRank& rhs) { return lhs.combined < rhs.combined; }
		friend inline bool operator> (const SelectionRank& lhs, const SelectionRank& rhs) { return rhs < lhs; }
		friend inline bool operator<=(const SelectionRank& lhs, const SelectionRank& rhs) { return !(lhs > rhs); }
		friend inline bool operator>=(const SelectionRank& lhs, const SelectionRank& rhs) { return !(lhs < rhs); }
		friend inline bool operator==(const SelectionRank& lhs, const SelectionRank& rhs) { return lhs.combined == rhs.combined; }
		friend inline bool operator!=(const SelectionRank& lhs, const SelectionRank& rhs) { return !(lhs == rhs); }

		friend SelectionRank operator-(SelectionRank lhs, const SelectionRank& rhs)
		{
			lhs -= rhs;
			return lhs;
		}

		SelectionRank& operator-=(const SelectionRank& rhs)
		{
			geo -= rhs.geo;
			size -= rhs.size;
			combined -= rhs.combined;

			return *this;
		}
	};

	std::ostream& operator<<(std::ostream& out, const SelectionRank& r);

	struct CSGNodeGenerationParams
	{
		CSGNodeGenerationParams(double create_new_prob, double active_prob, bool use_prim_geo_scores_as_active_prob, double dh_type_prob, bool evolve_dh_type, bool use_all_prims_for_ga);

		double create_new_prob;
		double active_prob;
		double dh_type_prob;
		bool evolve_dh_type;
		bool use_prim_geo_scores_as_active_prob;
		bool use_all_prims_for_ga;
	};

	struct PrimitiveDecomposition
	{
		PrimitiveDecomposition(const CSGNode& node, const PrimitiveSet& rem_prims, const PrimitiveSet& in_prims, const PrimitiveSet& out_prims);

		CSGNode node;
		PrimitiveSet remaining_primitives;
		PrimitiveSet inside_primitives;
		PrimitiveSet outside_primitives;

		PrimitiveSet get_primitives(bool all) const;
		std::vector<DHType> get_dh_types(bool all) const;

	};

	PrimitiveDecomposition decompose_primitives(const PrimitiveSet& primitives, const ModelSDF& model_sdf, double inside_t, double outside_t, double voxel_size);

	CSGNode integrate_node(const CSGNode& into, const PrimitiveSelection& s);

	CSGNode generate_csg_node(const PrimitiveDecomposition& decomposition, const std::shared_ptr<PrimitiveSetRanker>& primitive_ranker, const CSGNodeGenerationParams& params);
}

#endif