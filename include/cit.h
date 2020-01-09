#ifndef CIT_H
#define CIT_H

#include "dnf.h"

namespace lmu
{
	// Canonical Intersection Terms (also: fundamental products)
	struct CITS
	{
		DNF dnf;
		std::vector<Eigen::Vector3d> points;
		int size() const
		{
			return points.size();
		}
	};

	struct CITSets
	{
		CITS cits;
		DNF prime_implicants;
		std::vector<std::unordered_set<int>> pis_as_cit_indices;
	};

	CITS generate_cits(const lmu::CSGNode& n, double sampling_grid_size, const std::vector<ImplicitFunctionPtr>& primitives);

	DNF extract_prime_implicants(const CITS& cits, double sampling_grid_size);

	std::vector<std::unordered_set<int>> convert_pis_to_cit_indices(const DNF& prime_implicants, const CITS& cits);
	
	CITSets generate_cit_sets(const lmu::CSGNode& n, double sampling_grid_size, const std::vector<ImplicitFunctionPtr>& primitives = {});

	std::ostream& operator <<(std::ostream& stream, const CITSets& c);

	struct PythonInterpreter;

	CSGNode optimize_pi_set_cover(const CSGNode& node, double sampling_grid_size, 
		const PythonInterpreter& interpreter, const std::vector<ImplicitFunctionPtr>& primitives = {});


}

#endif