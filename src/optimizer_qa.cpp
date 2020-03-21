#include "optimizer_qa.h"
#include "optimizer_py.h"

using namespace lmu;


std::vector<std::unordered_set<int>> dominant_halfspaces_index_sets(const CITS& cits)
{
	std::vector<std::unordered_set<int>> index_sets;

	for (int i = 0; i < cits.dnf.functions.size(); ++i)
	{
		std::unordered_set<int> fully_in, fully_out;
		for (const auto& cit : cits.dnf.clauses)
		{
			//TODO
		}
	}

	return index_sets;
}

CSGNode lmu::optimize_with_qa(const CSGNode& n, double sampling_grid_size, const std::vector<ImplicitFunctionPtr>& primitives,
	const PythonInterpreter& interpreter)
{
	CITS in_cits = generate_cits(n, sampling_grid_size, CITSGenerationOptions::INSIDE, primitives);
	CITS out_cits = generate_cits(n, sampling_grid_size, CITSGenerationOptions::OUTSIDE, primitives);

	CITS in_out_cits;
	in_out_cits.dnf.functions = in_cits.dnf.functions;
	in_out_cits.points.insert(in_out_cits.points.end(), in_cits.points.begin(), in_cits.points.end());
	in_out_cits.points.insert(in_out_cits.points.end(), out_cits.points.begin(), out_cits.points.end());
	in_out_cits.dnf.clauses.insert(in_out_cits.dnf.clauses.end(), in_cits.dnf.clauses.begin(), in_cits.dnf.clauses.end());
	in_out_cits.dnf.clauses.insert(in_out_cits.dnf.clauses.end(), out_cits.dnf.clauses.begin(), out_cits.dnf.clauses.end());

	auto index_sets = dominant_halfspaces_index_sets(in_out_cits);


	//Set to cover is {0,..., #in sits-1}
	std::unordered_set<int> indices_to_cover;
	for (int i = 0; i < in_cits.size(); ++i)
		indices_to_cover.insert(i);

	auto selected_cit_index_sets = interpreter.set_cover(index_sets, indices_to_cover);

	return CSGNode(nullptr);
}
