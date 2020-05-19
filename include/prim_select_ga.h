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

		PrimitiveSelection(const PrimitiveSet* primitives, const ModelSDF& model_sdf, double t_inside, double t_outside);
		
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
		CSGNodeGenerationParams(double create_new_prob, double active_prob, double dh_type_prob, bool evolve_dh_type);

		double create_new_prob;
		double active_prob;
		double dh_type_prob;
		bool evolve_dh_type;
	};

	CSGNode generate_csg_node(const PrimitiveSet& primitives, const std::shared_ptr<ModelSDF>& model_sdf, const CSGNodeGenerationParams& params);
}

#endif