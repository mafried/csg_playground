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

	enum class CITSGenerationOptions
	{
		INSIDE,
		OUTSIDE
	};

	CITS generate_cits(const lmu::CSGNode& n, double sampling_grid_size, CITSGenerationOptions options, 
		const std::vector<ImplicitFunctionPtr>& primitives = {});

	struct ModelSDF;
	std::tuple<std::vector<Eigen::Vector3d>, std::vector<double>> generate_cits(const ModelSDF& m, const std::vector<ImplicitFunctionPtr>& primitives, double sampling_grid_size, const lmu::CSGNode& gt_node);

	DNF extract_prime_implicants(const CITS& cits, const lmu::PointCloud& outside_points, double sampling_grid_size);

	std::vector<std::unordered_set<int>> convert_pis_to_cit_indices(const DNF& prime_implicants, const CITS& cits);
	
	CITSets generate_cit_sets(const lmu::CSGNode& n, double sampling_grid_size,  
		bool use_cit_points_for_pi_extraction, const std::vector<ImplicitFunctionPtr>& primitives = {});

	PointCloud extract_points_from_cits(const CITS& cit_sets);

	std::ostream& operator <<(std::ostream& stream, const CITSets& c);

	struct PythonInterpreter;
	
	CSGNode optimize_pi_set_cover(const CSGNode& node, double sampling_grid_size, bool use_cit_points_for_pi_extraction,
		const PythonInterpreter& interpreter, const std::vector<ImplicitFunctionPtr>& primitives = {}, std::ostream& report_stream = std::cout);


}

#endif