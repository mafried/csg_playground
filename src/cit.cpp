#include "cit.h"
#include "csgnode_helper.h"
#include "optimizer_red.h"

struct ClauseHash {
public:
	size_t operator()(const lmu::Clause& c) const {
		std::size_t seed = 0;

		for (int i = 0; i < c.literals.size(); ++i)
		{
			bool lit = c.literals[i];
			boost::hash_combine(seed, lit);
			if (lit)
				boost::hash_combine(seed, c.negated[i]);
		}
		//boost::hash_combine(seed, lit_hash);
		//boost::hash_combine(seed, negate_hash);

		return seed;
	}
};

struct ClauseEqual
{
	bool operator()(const lmu::Clause& lhs, const lmu::Clause& rhs) const
	{
		if (lhs.size() != rhs.size()) return false;

		for (int i = 0; i < lhs.size(); ++i)
			if (lhs.literals[i] != rhs.literals[i] ||
				(rhs.literals[i] && lhs.negated[i] != rhs.negated[i]))
			{
				return false;
			}

		return true;
	}
};

struct ClauseAndPoint
{
	ClauseAndPoint(const lmu::Clause& c, Eigen::Vector3d& p) : clause(c), p(p)
	{
	}

	lmu::Clause clause; 
	Eigen::Vector3d p;
};

bool operator==(const ClauseAndPoint& lhs, const ClauseAndPoint& rhs)
{
	return ClauseEqual()(lhs.clause, rhs.clause);
}

struct ClauseAndPointHash {
public:
	size_t operator()(const ClauseAndPoint& c) const 
	{
		return ClauseHash()(c.clause);
	}
};

struct ClauseAndPointEqual
{
	bool operator()(const ClauseAndPoint& lhs, const ClauseAndPoint& rhs) const
	{
		return lhs == rhs;
	}
};

lmu::CITS lmu::generate_cits(const lmu::CSGNode& n, double sgs)
{
	lmu::AABB aabb = aabb_from_node(n);
	Eigen::Vector3d min = aabb.c - aabb.s;
	Eigen::Vector3d max = aabb.c + aabb.s;

	auto prims = allDistinctFunctions(n);
	std::unordered_set<ClauseAndPoint, ClauseAndPointHash, ClauseAndPointEqual> clauses;

	//std::cout << "C: " << aabb.c.transpose() << " S: " << aabb.s.transpose() << std::endl;
	//std::cout << "min: " << min.transpose() << " max: " << max.transpose() << std::endl;

	Eigen::Vector3i s = ((max - min) / sgs).cast<int>() + Eigen::Vector3i(1,1,1); //TODO: round
	
	for(int x = 0; x < s.x(); x++)
		for (int y = 0; y < s.y(); y++)
			for (int z = 0; z < s.z(); z++)
			{
				Eigen::Vector3d p(x * sgs + min.x(), y * sgs + min.y(), z * sgs + min.z());
				
				double nd = n.signedDistance(p);				
				if (nd >= 0.0) continue;

				lmu::Clause clause(prims.size());
				for (int i = 0; i < prims.size(); ++i)
				{
					double pd = prims[i]->signedDistance(p);
					clause.literals[i] = true;
					clause.negated[i] = pd > 0.0;
				}
				clauses.insert(ClauseAndPoint(clause, p));
			}

	CITS cits;

	cits.dnf.functions = prims;
	std::transform(clauses.begin(), clauses.end(), std::back_inserter(cits.points), [](const ClauseAndPoint& cap) { return cap.p; });
	std::transform(clauses.begin(), clauses.end(), std::back_inserter(cits.dnf.clauses), [](const ClauseAndPoint& cap) { return cap.clause; });

	//std::cout << "CLAUSES: " << cits.dnf.clauses.size() << std::endl;
	//lmu::writeNode(lmu::DNFtoCSGNode(cits.dnf), "dnf2.dot");

	return cits;
}

bool is_outside(const lmu::Clause& c, const lmu::CITS& cits, double sampling_grid_size, lmu::EmptySetLookup& esLookup)
{
	auto clause_node = lmu::clauseToCSGNode(c, cits.dnf.functions);
	auto model_node = lmu::DNFtoCSGNode(cits.dnf);
	
	//lmu::writeNode(model_node, "self.dot");

	//std::stringstream ss;
	//ss << c << ".dot";
	//lmu::writeNode(clause_node, ss.str());
	
	bool empty_set = lmu::is_empty_set(lmu::opDiff({ clause_node, model_node }), sampling_grid_size, esLookup);

	//std::cout << c << " empty set: " << empty_set << std::endl;

	bool outside = !empty_set;

	
	return outside;
	//for (const auto& p : cits.points)
	//	if (clause_node.signedDistance(p) > 0.0)
	//	{
	//		std::cout << c << " is outside." << clause_node.signedDistance(p) << std::endl;
	//		return true;
	//	}

	//return false;
}

lmu::Clause create_prime_clause(const lmu::Clause& c, const lmu::CITS& cits, double sampling_grid_size, lmu::EmptySetLookup& esLookup)
{
	lmu::Clause prime_clause = c;
	int num_removed = 0;
	int available_literals = std::count_if(c.literals.begin(), c.literals.end(), [](bool available) {return available; });

	//std::cout << "pi clause: " << c << " " << available_literals << std::endl;
	for (int i = 0; i < c.size(); ++i)
	{
		if (!prime_clause.literals[i])
			continue;

		prime_clause.literals[i] = false;
		if (is_outside(prime_clause, cits, sampling_grid_size, esLookup) || available_literals == num_removed + 1)
		{
			prime_clause.literals[i] = true;
		}
		else
		{
			num_removed++;
		}
	}

	return prime_clause;
}

lmu::DNF lmu::extract_prime_implicants(const CITS& cits, double sampling_grid_size)
{
	DNF prime_implicants;

	std::unordered_set<Clause, ClauseHash, ClauseEqual> prime_clauses;
	lmu::EmptySetLookup esLookup;

	for (const auto& clause : cits.dnf.clauses)
	{
		prime_clauses.insert(create_prime_clause(clause, cits, sampling_grid_size, esLookup));
	}

	prime_implicants.clauses = std::vector<lmu::Clause>(prime_clauses.begin(), prime_clauses.end());
	prime_implicants.functions = cits.dnf.functions;

	return prime_implicants;
}

std::vector<std::unordered_set<int>> lmu::convert_pis_to_cit_indices(const DNF& prime_implicants, const CITS& cits)
{
	std::vector<std::unordered_set<int>> cit_indices_vec;

	for (const auto& clause : prime_implicants.clauses)
	{
		std::unordered_set<int> sit_indices;
		
		for (int i = 0; i < cits.points.size(); ++i)
		{
			if (clause.signedDistance(cits.points[i], cits.dnf.functions) <= 0.0)
			{
				//std::cout << " " << i << " ";
				sit_indices.insert(i);
			}
		}
		
		cit_indices_vec.push_back(sit_indices);
	}

	return cit_indices_vec;
}

lmu::CITSets lmu::generate_cit_sets(const lmu::CSGNode& n, double sampling_grid_size)
{
	CITSets sets;

	sets.cits = generate_cits(n, sampling_grid_size);
	sets.prime_implicants = extract_prime_implicants(sets.cits, sampling_grid_size);
	sets.pis_as_cit_indices = convert_pis_to_cit_indices(sets.prime_implicants, sets.cits);
	
	return sets;
}

std::ostream& lmu::operator <<(std::ostream& stream, const lmu::CITSets& c)
{
	stream << "# Primitives: ";
	for (const auto& p : c.cits.dnf.functions) stream << p->name() << " ";
	stream << std::endl;

	stream << "# Canonical Intersection Terms: ";
	for (const auto cit : c.cits.dnf.clauses)
	{
		print_clause(stream, cit, c.cits.dnf.functions, false);
		stream << " ";
	}
	stream << std::endl;
	stream << "  Espresso: " << espressoExpression(c.cits.dnf) << std::endl;
	stream << std::endl;
	
	stream << "# Prime Implicants: ";
	for (const auto& pi : c.prime_implicants.clauses)
	{
		print_clause(stream, pi, c.cits.dnf.functions, false);
		stream << " ";
	}
	stream << std::endl;
	stream << "  Espresso: " << espressoExpression(c.prime_implicants) << std::endl;
	stream << "  Sets: ";
	for (const auto indices : c.pis_as_cit_indices)
	{
		stream << "{ ";
		for (auto index : indices) stream << index << " ";
		stream << "} ";
	}
	stream << std::endl;
	stream << "  Set to cover: ";
	stream << "{ ";
	for (int i = 0; i < c.cits.dnf.clauses.size(); ++i) stream << i << " ";
	stream << "} ";

	stream << std::endl;

	return stream;
}

lmu::CSGNode lmu::optimize_pi_set_cover(CSGNode node, double sampling_grid_size)
{
	auto cit_sets = generate_cit_sets(node, sampling_grid_size);

	//TODO: do set cover on prime implicants.

	std::cout << "SET" << std::endl;
	std::cout << cit_sets << std::endl;

	return DNFtoCSGNode(cit_sets.prime_implicants);
}
