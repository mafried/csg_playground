#include "prim_select_ga.h"
#include "csgnode_helper.h"
#include "optimizer_red.h"

#include "mesh.h"
#include <igl/writeOBJ.h>


using namespace lmu;


lmu::CSGNode computeForTwoFunctions(const std::vector<ImplicitFunctionPtr>& functions, const SelectionRanker& ranker);

lmu::SelectionValue::SelectionValue(DHType dh_type, bool active) : 
	dh_type(dh_type),
	active(active)
{
}

size_t lmu::SelectionValue::hash(size_t seed) const
{
	boost::hash_combine(seed, dh_type);
	boost::hash_combine(seed, active);

	return seed;
}

lmu::PrimitiveSelection::PrimitiveSelection(const PrimitiveSet* primitives) : 
	prims(primitives),
	node(opNo())
{
	std::default_random_engine rnd_engine;
	std::random_device rnd_device;
	rnd_engine.seed(rnd_device());

	std::bernoulli_distribution d{};
	using parm_t = decltype(d)::param_type;

	if (primitives)
	{
		int i = 0;
		for (const auto& p : *primitives)
		{
			DHType dh_type = d(rnd_engine, parm_t{ 0.5 }) ? DHType::INSIDE : DHType::OUTSIDE;

			selection.push_back(SelectionValue(dh_type, true));

			ordering.push_back(i);
			i++;
		}
	}

	std::shuffle(ordering.begin(), ordering.end(), rnd_engine);
}

lmu::PrimitiveSelection::PrimitiveSelection(const PrimitiveSet* primitives, const std::vector<DHType>& dh_types, const std::vector<int>& ordering) :
	prims(primitives),
	ordering(ordering),
	node(opNo())
{
	if (primitives)
	{
		for (int i = 0; i < primitives->size(); ++i)
		{
			selection.push_back(SelectionValue(dh_types[i], true));
		}
	}
}

lmu::PrimitiveSelection::PrimitiveSelection(const CSGNode& node) : 
	node(node),
	prims(nullptr)
{
}

lmu::CSGNode lmu::PrimitiveSelection::to_node() const
{
	
	auto node = lmu::opUnion();

	for (int i = 0; i < selection.size(); ++i)
	{
		int idx = ordering[i];
		
		if (selection[idx].dh_type == DHType::OUTSIDE)
		{

			node = lmu::opDiff({ node, lmu::geometry(prims->at(idx).imFunc) });

		}
		else
		{
			node.addChild(lmu::geometry(prims->at(idx).imFunc));
		}
	}

	// Replace operations with a single operand with the operand.
	lmu::visit(node, [](lmu::CSGNode& n)
	{
		if (n.childsCRef().size() == 1)
		{
			n = n.childsCRef()[0];
		}
	});
		
	return node;
}

int lmu::PrimitiveSelection::get_num_active() const
{
	int n = 0;
	for (const auto& s : selection)
		n += s.active ? 1 : 0;
	return n;
}

bool lmu::PrimitiveSelection::is_selection() const
{
	return prims != nullptr; //node.operationType() == CSGNodeOperationType::Noop;
}

size_t lmu::PrimitiveSelection::hash(size_t seed) const
{	
	if (node.operationType() == CSGNodeOperationType::Noop)
	{
		int i = 0;
		for (const auto& s : selection)
		{
			boost::hash_combine(seed, s.dh_type);
			boost::hash_combine(seed, s.active);
			boost::hash_combine(seed, ordering[i]);
			i++;
		}
	}
	else
	{
		seed = node.hash(seed);
	}

	return seed;
}

std::ostream& lmu::operator<<(std::ostream& out, const PrimitiveSelection& ps)
{
	int i = 0;
	for (const auto& s : ps.selection)
	{
		out << (ps.prims ? ps.prims->at(i++).imFunc->name() : "No name") << ": " << s.active << " " << (s.dh_type == DHType::INSIDE ? "IN" : s.dh_type == DHType::OUTSIDE ? "OUT" : "NONE") << " | ";
	}
	out << std::endl;

	return out;
}

//////////////////////////// CREATOR ////////////////////////////

struct SelectionCreator
{
	SelectionCreator(const lmu::PrimitiveSelection& ps, const CSGNodeGenerationParams& params) :
		ps(ps),
		params(params)
	{
		_rndEngine.seed(_rndDevice());

	}

	lmu::PrimitiveSelection mutate(const lmu::PrimitiveSelection& s) const
	{
		static std::bernoulli_distribution d{};
		using parm_t = decltype(d)::param_type;

		static std::uniform_int_distribution<> uniform_d{};
		using p_ud = decltype(uniform_d)::param_type;
				
		if (d(_rndEngine, parm_t{ params.create_new_prob }))
		{
			return create();
		}
		else
		{
			int num_selections = ps.selection.size();
			auto new_s = s;

			for (int i = 0; i < num_selections; ++i)
			{
				int s_idx = uniform_d(_rndEngine, p_ud{ 0, num_selections - 1 });
							
				if (d(_rndEngine, parm_t{ 0.5 }))
				{
					new_s.selection[s_idx].dh_type = s.selection[s_idx].dh_type == DHType::INSIDE ? DHType::OUTSIDE : DHType::INSIDE; 
				}
			}

			// Shuffle indices sequence.
			if (d(_rndEngine, parm_t{ 0.5 }))
			{				
				int x1 = uniform_d(_rndEngine, p_ud{ 0, num_selections});
				int x2 = x1;
				while (x2 == x1)
					x2 = uniform_d(_rndEngine, p_ud{ 0, num_selections });

				int start = std::min(x1, x2);
				int stop = std::max(x1, x2);

				for (const auto& v : new_s.ordering)
					std::cout << (int)v << " ";
				std::cout << std::endl;
				std::shuffle(new_s.ordering.begin() + start, new_s.ordering.begin() + stop, _rndEngine);
			
				for (const auto& v : new_s.ordering)
					std::cout << (int)v << " ";
				std::cout << std::endl;
				std::cout << "----------------- " << start << " " << stop << std::endl;
			}
			
			return new_s;
		}
	}

	std::vector<PrimitiveSelection> crossover(const PrimitiveSelection& s1, const PrimitiveSelection& s2) const
	{
		int num_selections = ps.selection.size();

		auto new_s1 = s1;
		auto new_s2 = s2;

		static std::uniform_int_distribution<> uniform_d{};
		using p_ud = decltype(uniform_d)::param_type;
			
		int x1 = uniform_d(_rndEngine, p_ud{ 0, num_selections });
		int x2 = x1;
		while (x2 == x1)
			x2 = uniform_d(_rndEngine, p_ud{ 0, num_selections });

		int start = std::min(x1, x2);
		int stop = std::max(x1, x2);

		//Swap selection. 

		/*
		std::cout << "--------------------------" << std::endl;
		std::cout << "Crossover: Exchange dh type sequence. start: " << start << " end: " << stop << std::endl;

		for (const auto& v : new_s1.selection)
			std::cout << (int)v.dh_type << " ";
		std::cout << std::endl;
		for (const auto& v : new_s2.selection)
			std::cout << (int)v.dh_type << " ";
		std::cout << std::endl;
		*/
		
		std::swap_ranges(new_s1.selection.begin() + start, new_s1.selection.begin() + stop, new_s2.selection.begin() + start);

		/*
		for (const auto& v : new_s1.selection)
			std::cout << (int)v.dh_type << " ";
		std::cout << std::endl;
		for (const auto& v : new_s2.selection)
			std::cout << (int)v.dh_type << " ";
		std::cout << std::endl;
		

		std::cout << "--------------------------" << std::endl;
		*/
		
		/*		
		// Also exchange ordering sequnece if it contains the same indices. 
		if (std::is_permutation(new_s1.ordering.begin() + start, new_s1.ordering.begin() + stop, new_s2.ordering.begin() + start, new_s2.ordering.begin() + stop))
		{
			std::cout << "Crossover: Exchange ordering sequence. start: " << start << " end: " << stop << std::endl;
			for (const auto& v : new_s1.ordering)
				std::cout << v << " ";
			std::cout << std::endl;
			for (const auto& v : new_s2.ordering)
				std::cout << v << " ";
			std::cout << std::endl;
		*/
		
		std::swap_ranges(new_s1.ordering.begin() + start, new_s1.ordering.begin() + stop, new_s2.ordering.begin() + start);
		
		/*
			for (const auto& v : new_s1.ordering)
				std::cout << v << " ";
			std::cout << std::endl;
			for (const auto& v : new_s2.ordering)
				std::cout << v << " ";
			std::cout << std::endl;
			
		}
		*/

		return std::vector<lmu::PrimitiveSelection>
		{
			new_s1, new_s2
		};
	}

	PrimitiveSelection create() const
	{
		static std::bernoulli_distribution d{};
		using parm_t = decltype(d)::param_type;

		auto new_ps = ps;

		for (int i = 0; i < new_ps.selection.size(); ++i)
		{
			new_ps.selection[i].active = true;
			
			new_ps.selection[i].dh_type = d(_rndEngine, parm_t{ params.dh_type_prob }) ? DHType::INSIDE : DHType::OUTSIDE;

			new_ps.ordering[i] = i;
		}

		std::shuffle(new_ps.ordering.begin(), new_ps.ordering.end(), _rndEngine);
		
		return new_ps;
	}

	std::string info() const
	{
		return std::string();
	}

	private:

		lmu::PrimitiveSelection ps;
	
		CSGNodeGenerationParams params;

		mutable std::default_random_engine _rndEngine;
		mutable std::random_device _rndDevice;
};


struct CSGNodeCreator
{
	CSGNodeCreator(const lmu::PrimitiveSelection& ps, const CSGNodeGenerationParams& params) :
		params(params)
	{
		for (const auto& p : *ps.prims)
			primitives.push_back(p.imFunc);	

		_rndEngine.seed(_rndDevice());
	}

	PrimitiveSelection mutate(const PrimitiveSelection& ps) const
	{
		auto node = ps.node;

		static std::bernoulli_distribution d{};
		using parm_t = decltype(d)::param_type;

		static std::uniform_int_distribution<> du{};
		using parmu_t = decltype(du)::param_type;

		static std::uniform_real_distribution<double> dur(-0.1, 0.1);
		using parmur_t = decltype(dur)::param_type;

		if (d(_rndEngine, parm_t{ params.create_new_prob }))
		{
			return create(params.max_tree_depth);
		}
		else
		{
			int nodeIdx = du(_rndEngine, parmu_t{ 0, numNodes(node) - 1 });

			auto newNode = node;

			CSGNode* subNode = nodePtrAt(newNode, nodeIdx);

			create(*subNode, params.max_tree_depth, /*0*/ depth(*subNode));

			return PrimitiveSelection(newNode);
		}
	}

	std::vector<PrimitiveSelection> crossover(const PrimitiveSelection& ps1, const PrimitiveSelection& ps2) const
	{
		auto node1 = ps1.node;
		auto node2 = ps2.node;

		if (!node1.isValid() || !node2.isValid())
			return std::vector<PrimitiveSelection> {PrimitiveSelection(node1), PrimitiveSelection(node2)};

		int numNodes1 = numNodes(node1);
		int numNodes2 = numNodes(node2);

		auto newNode1 = node1;
		auto newNode2 = node2;

		static std::uniform_int_distribution<> du{};
		using parmu_t = decltype(du)::param_type;

		int nodeIdx1 = du(_rndEngine, parmu_t{ 0, numNodes1 - 1 });
		int nodeIdx2 = du(_rndEngine, parmu_t{ 0, numNodes2 - 1 });

		CSGNode* subNode1 = nodePtrAt(newNode1, nodeIdx1);
		CSGNode* subNode2 = nodePtrAt(newNode2, nodeIdx2);

		std::swap(*subNode1, *subNode2);

		//std::cout << "MAX: " << params.max_tree_depth << std::endl;
		//std::cout << "Before: 1: " << depth(newNode1) << " 2: " << depth(newNode2) << std::endl;
		shrink_large_nodes(newNode1, params.max_tree_depth);
		shrink_large_nodes(newNode2, params.max_tree_depth);
		//std::cout << "After: 1: " << depth(newNode1) << " 2: " << depth(newNode2) << std::endl;

		return std::vector<PrimitiveSelection>
		{
			PrimitiveSelection(newNode1), PrimitiveSelection(newNode2)
		};
	}

	void shrink_large_nodes(CSGNode& node, int max_depth) const
	{
		while (depth(node) > max_depth)
		{
			lmu::visit(node, [&node, &max_depth](CSGNode& n)
			{
				if (n.childsCRef().empty() || depth(node) <= max_depth)
					return;

				bool all_childs_are_leaves = true;
				for (const auto &c : n.childsCRef())
				{
					if (c.type() != CSGNodeType::Geometry && c.operationType() != CSGNodeOperationType::Noop)
					{
						all_childs_are_leaves = false;
						break;
					}
				}
				if (all_childs_are_leaves)
				{
					n = n.childsCRef()[0];
				}
			});
		}		
	}

	PrimitiveSelection create() const
	{
		return create(params.max_tree_depth);
	}

	PrimitiveSelection create(int max_depth) const
	{
		auto node = opNo();
		create(node, max_depth, 0);
		return PrimitiveSelection(node);
	}

	std::string info() const
	{
		return std::string();
	}

private:

	void create(CSGNode& node, int max_depth, int cur_depth) const
	{
		static std::bernoulli_distribution db{};
		using parmb_t = decltype(db)::param_type;

		if (cur_depth >= max_depth)
		{
			node = create_rnd_primitive_node();
		}
		else
		{
			if (db(_rndEngine, parmb_t{ params.subtree_prob }))
			{
				node = create_rnd_operation_node();

				auto numAllowedChilds = node.numAllowedChilds();
				int numChilds = clamp(std::get<1>(numAllowedChilds), std::get<0>(numAllowedChilds), 2); //2 is the maximum number of childs allowed for create

				for (int i = 0; i < numChilds; ++i)
				{
					auto child = CSGNode::invalidNode;
					create(child, max_depth, cur_depth + 1);
					node.addChild(child);
				}
			}
			else
			{
				node = create_rnd_primitive_node();
			}
		}
	}

	CSGNode create_rnd_primitive_node() const
	{
		static std::uniform_int_distribution<> du{};
		using parmu_t = decltype(du)::param_type;

		int funcIdx =  du(_rndEngine, parmu_t{ 0, static_cast<int>(primitives.size() - 1) });
		return geometry(primitives[funcIdx]);
	}

	CSGNode create_rnd_operation_node() const
	{
		std::discrete_distribution<> d({ 1, 1, 1, 1});
		int op = d(_rndEngine);

		CSGNodeOperationType op_type; 
		switch (op)
		{
		default:
		case 0: 
			op_type = CSGNodeOperationType::Union;
			break;
		case 1:
			op_type = CSGNodeOperationType::Difference;
			break;
		case 2:
			op_type = CSGNodeOperationType::Intersection;
			break;
		case 3:
			op_type = CSGNodeOperationType::Complement;
			break;
		}

		return createOperation(op_type);
	}

	CSGNodeGenerationParams params;
	std::vector<ImplicitFunctionPtr> primitives;

	mutable std::default_random_engine _rndEngine;
	mutable std::random_device _rndDevice;
};

struct CombinedCreator
{
	CombinedCreator(const lmu::PrimitiveSelection& ps, const CSGNodeGenerationParams& params) :
		params(params),
		node_creator(ps, params),
		selection_creator(ps, params)
	{
		_rndEngine.seed(_rndDevice());
	}

	lmu::PrimitiveSelection mutate(const lmu::PrimitiveSelection& s) const
	{
		if (s.is_selection())
		{
			return selection_creator.mutate(s);
		}
		else
		{
			return node_creator.mutate(s);
		}
	}

	std::vector<PrimitiveSelection> crossover(const PrimitiveSelection& s1, const PrimitiveSelection& s2) const
	{
		if (s1.is_selection() && s2.is_selection())
		{
			return selection_creator.crossover(s1, s2);
		}
		else if (!s1.is_selection() && !s2.is_selection())
		{
			return node_creator.crossover(s1, s2);
		}
		else
		{		
			
			if (s1.is_selection())
			{
				PrimitiveSelection s1_converted(s1.to_node());
				return node_creator.crossover(s1_converted, s2);
			}

			if (s2.is_selection())
			{
				PrimitiveSelection s2_converted(s2.to_node());
				return node_creator.crossover(s1, s2_converted);
			}
			
			return { s1,s2 };
		}
	}

	PrimitiveSelection create() const
	{
		static std::bernoulli_distribution db{};
		using parmb_t = decltype(db)::param_type;

		if (db(_rndEngine, parmb_t{ params.node_creator_prob }))
		{
			return node_creator.create();
		}
		else
		{
			return selection_creator.create();
		}
	}

	std::string info() const
	{
		return std::string();
	}

private:

	mutable std::default_random_engine _rndEngine;
	mutable std::random_device _rndDevice;
	CSGNodeGenerationParams params;

	CSGNodeCreator node_creator;
	SelectionCreator selection_creator;
};


//////////////////////////// RANKER ////////////////////////////


lmu::SelectionRanker::SelectionRanker(const std::shared_ptr<ModelSDF>& model_sdf) :
	model_sdf(model_sdf)
{
}

lmu::SelectionRanker::SelectionRanker(const std::vector<Eigen::Vector3d>& points, const std::vector<double>& distances) :
	points(points),
	distances(distances),
	model_sdf(nullptr)
{

}

lmu::SelectionRank lmu::SelectionRanker::rank(const PrimitiveSelection& s, bool debug) const
{
	auto node = s.is_selection() ? s.to_node() : s.node;

	double mean_abs_d = 0;

	for (int i = 0; i < points.size(); ++i)
	{
		auto d = node.signedDistance(points[i]);

		mean_abs_d += std::abs(distances[i] - d);
	}

	mean_abs_d /= (double)points.size();

	auto node_size = (double)numNodes(node);

	auto sr = SelectionRank(mean_abs_d, node_size, 0.0);

	sr.capture_unnormalized();

	return sr;
}


lmu::SelectionRank lmu::SelectionRanker::rank2(const PrimitiveSelection& s, bool debug) const
{
	auto node = s.is_selection() ? s.to_node() : s.node;

	if (lmu::numNodes(node) == 0)
	{
		return SelectionRank(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 0.0);
	}

	auto min = model_sdf->origin;
	Eigen::Vector3d grid_size = model_sdf->grid_size.cast<double>();
	auto max = model_sdf->origin + grid_size * model_sdf->voxel_size;

	int size = 32;

	Eigen::Vector3d cell_size = (max - min) / (double)size;


	//std::cout << "min " << min.transpose() << " max: " << max.transpose() << " cell_size: " << cell_size.transpose() << std::endl;

	double mean_abs_d = 0.0;
	double max_abs_d = 0.0;
	double counter = 0.0;
	for (int x = 0; x < size; ++x)
	{
		for (int y = 0; y < size; ++y)
		{
			for (int z = 0; z < size; ++z)
			{
				Eigen::Vector3d p(min.x() + (double)x * cell_size.x(), min.y() + (double)y * cell_size.y(), min.z() + (double)z * cell_size.z());

				auto sd_gr = node.signedDistanceAndGradient(p);

				auto d = (double)model_sdf->sdf_value(p).d;

				if (sd_gr.x() == std::numeric_limits<double>::max())
				{
					continue;
				}
				//std::cout << d << " " << sd_gr.x() << "|";

				double abs_dist = std::abs(sd_gr.x() - d);

				max_abs_d = std::max(max_abs_d, mean_abs_d);

				mean_abs_d += abs_dist;
				counter += 1.0;
			}
		}
	}

	mean_abs_d /= counter;

	if (!std::isfinite(mean_abs_d))
		return SelectionRank(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 0.0);

	//std::cout << "MEAN: " << mean_abs_d << " " << counter << std::endl;

	auto node_size = (double)numNodes(node);

	auto sr = SelectionRank(mean_abs_d, node_size, 0.0);

	sr.capture_unnormalized();

	return sr;
}

lmu::SelectionRank lmu::SelectionRanker::rank3(const PrimitiveSelection& s, bool debug) const
{

	auto node = s.is_selection() ? s.to_node() : s.node;
	auto d = 0.0;

	auto grid_size = model_sdf->grid_size;
	auto voxel_size = model_sdf->voxel_size;
	auto origin = model_sdf->origin;

	std::vector<Eigen::Matrix<double, 1, 6>> points;
	int step = 1;


	for (int x = 0; x < grid_size.x(); x += step)
	{
		for (int y = 0; y < grid_size.y(); y += step)
		{
			for (int z = 0; z < grid_size.z(); z += step)
			{
				Eigen::Vector3d p = Eigen::Vector3d(x, y, z) * voxel_size + origin; //+ Eigen::Vector3d(voxel_size * 0.5, voxel_size * 0.5, voxel_size * 0.5);

				int idx = x + grid_size.x() * y + grid_size.x() * grid_size.y() * z;

				auto v = model_sdf->data[idx];

				auto sd_gr = node.signedDistanceAndGradient(p);
				Eigen::Vector3f gr = sd_gr.bottomRows(3).cast<float>();

				//Eigen::Matrix<double, 1, 6> point;

				if (v.d > voxel_size && sd_gr.x() > voxel_size)
					continue;

				Eigen::Vector3d p1 = p - (double)v.d * v.n.normalized().cast<double>();
				Eigen::Vector3d p2 = p - sd_gr.x() * sd_gr.bottomRows(3).normalized();

				double abs_dist = (p2 - p1).norm();

				//d += std::abs(v.v - sd);
				if (abs_dist < voxel_size && gr.dot(v.n) >= 0.0)
				{
					d += 1.0;

					//point << p.transpose(), 0.0, 1.0, 0.0;

				}
				else
				{
					//point << p.transpose(),1.0, 0.0, gr.dot(v.n) >= 0.0 ? 0.0 : 1.0;
				}

				//if(debug)
				//	points.push_back(point);

			}
		}
	}

	//std::cout << s << " D: " << d << std::endl;

	auto size = (double)numNodes(node);

	auto sr = SelectionRank(d, size, 0.0);

	//sr.points = points;

	sr.capture_unnormalized();


	return sr;
}


std::string lmu::SelectionRanker::info() const
{
	return std::string();
}


//////////////////////////// SYNC POINT ////////////////////////////

struct SelectionSyncPoint
{
	SelectionSyncPoint(int max_budget, double node_ratio) :
		max_budget(max_budget),
		node_ratio(node_ratio),
		node_iter_budget(1),
		node_done(false),
		selection_done(false),
		transfer_counter(0)
	{
	}

	void synchronize(const std::string& id, std::vector<RankedCreature<PrimitiveSelection, SelectionRank>>& population) const 
	{
		if (node_ratio == 1.0)
			return;

		std::cout << "============================== SYNC POINT ID " << id << " " << (int)this << "Ratio: " << node_ratio << std::endl;


		if (id == "selection")
		{
			std::unique_lock<std::mutex> l(mutex);
			
			cv.wait(l, [this]{ return selection_iter_budget > 0 || node_done; });
			
			sorted_pop = population;
			std::cout << "Sorted Pop: " << sorted_pop.size() << std::endl;

			selection_iter_budget--;
			node_iter_budget = max_budget;

			l.unlock();
			cv.notify_one();

		}
		else
		{
			std::unique_lock<std::mutex> l(mutex);

			cv.wait(l, [this] { return node_iter_budget > 0 || selection_done; });

			TournamentSelector<RankedCreature<PrimitiveSelection, SelectionRank>> selector(2);

			if (!sorted_pop.empty() && transfer_counter >= max_budget)
			{
				transfer_counter = 0;

				std::cout << "TRANSFER" << std::endl;

				// replace last n trees of the population.
				int n = node_ratio * population.size();
				for (int i = n; i < population.size(); ++i)
				{
					auto selected = selector.selectFrom(sorted_pop);
					bool already_there = false;
					for (const auto& p : population)
					{
						if (p.creature.to_node().hash(0) == selected.creature.to_node().hash(0))
						{
							already_there = true;
							break;
						}
					}
					if (!already_there)
					{
						population[i] = selector.selectFrom(sorted_pop);
						std::cout << "TRANSFER DONE" << std::endl;
						//population[i].creature = PrimitiveSelection(population[i].creature/*.to_node()*/);
					}
				}

				//re-sort by rank.
				std::sort(population.begin(), population.end(),
					[](const auto& a, const auto& b)
				{
					return a.rank > b.rank;
				});

				sorted_pop.clear();
			}
			else
			{
				transfer_counter++;
			}

			node_iter_budget--;
			selection_iter_budget = max_budget;

			l.unlock();
			cv.notify_one();
		}
	}

	void mark_as_done(const std::string& id) const 
	{
		if (id == "selection")
			selection_done = true;
		else
			node_done = true;
	}		

	std::string info() const
	{
		return std::string();
	}

	mutable bool selection_done;
	mutable bool node_done; 

	mutable std::vector<RankedCreature<PrimitiveSelection, SelectionRank>> sorted_pop; 
	mutable std::mutex mutex;
	mutable std::condition_variable cv;
	mutable int selection_iter_budget;
	mutable int node_iter_budget;
	mutable int transfer_counter;

	int max_budget;
	double node_ratio = 0.5;
};

//////////////////////////// POP MAN ////////////////////////////

struct SelectionPopMan
{
	SelectionPopMan(double geo_weight, double size_weight) :
		geo_weight(geo_weight),
		size_weight(size_weight)
	{
		_rndEngine.seed(_rndDevice());
	}

	void manipulateBeforeRanking(std::vector<RankedCreature<PrimitiveSelection, SelectionRank>>& population) const
	{
		/*
		double selection_ratio = 0.5;

		int c = 0;
		for (auto& s : population)
		{
			if (s.creature.is_selection())
				c++;
		}
		int num_to_transform = c - (int)((double)population.size() * selection_ratio);

		std::cout << "C: " << c << " num to transform: " << num_to_transform << " " << ((int)((double)population.size() * selection_ratio)) << std::endl;

		while(num_to_transform > 0)
		{
			static std::uniform_int_distribution<> uniform_d{};
			using p_ud = decltype(uniform_d)::param_type;
			
			int i = uniform_d(_rndEngine, p_ud{ 0, (int)population.size() - 1 });
			if (population[i].creature.is_selection())
			{
				population[i].creature = PrimitiveSelection(population[i].creature.to_node());
				num_to_transform--;
			}
		}
		*/
	}

	void manipulateAfterRanking(std::vector<RankedCreature<PrimitiveSelection, SelectionRank>>& population) const
	{
		int c = 0; 
		for (auto& s : population)
		{
			if (s.creature.is_selection())
				c++;
		}
		std::cout << "SELECTION: " << c << " of " << population.size() << std::endl;
		double node_ratio = (double)c / (double)population.size(); 
		
		// Re-normalize scores and compute combined score. 
		SelectionRank max_r(-std::numeric_limits<double>::max()), min_r(std::numeric_limits<double>::max());
				
		for (auto& s : population)
		{
			if (s.rank.geo == std::numeric_limits<double>::max() || s.rank.size == std::numeric_limits<double>::max())
				continue;

			max_r.size = std::max(max_r.size, s.rank.size) ;
			min_r.size = std::min(min_r.size, s.rank.size);
			max_r.geo = std::max(max_r.geo, s.rank.geo);
			min_r.geo = std::min(min_r.geo, s.rank.geo);
		}
		auto diff_r = max_r - min_r;

		for (auto& ps : population)
		{
			ps.rank.combined_unnormalized = -ps.rank.geo * geo_weight - ps.rank.size * size_weight;

			// Normalized
			
			//std::cout << "GEO: " << min_r.geo << " " << max_r.geo << " " << diff_r.geo << std::endl;

			ps.rank.geo = ps.rank.geo < 0.0 || diff_r.geo == 0.0 ? 0.0 : (ps.rank.geo - min_r.geo) / diff_r.geo;
			ps.rank.size = ps.rank.size < 0.0 || diff_r.size == 0.0 ? 0.0 : (ps.rank.size - min_r.size) / diff_r.size;

			ps.rank.combined = -ps.rank.geo * geo_weight - ps.rank.size * size_weight;

			ps.rank.node_ratio = node_ratio;
			
			//std::cout << "Size: " << size_weight << " " << (ps.rank.size * size_weight) << std::endl;
			//std::cout << "Rank: " << ps.rank << std::endl;
			//std::cout << "diff: " << diff_r << std::endl;
		}
		//std::cout << "========================" << std::endl;
	}

	std::string info() const
	{
		return std::string();
	}

private:

	mutable std::default_random_engine _rndEngine;
	mutable std::random_device _rndDevice;

	double geo_weight;
	double size_weight;

};

//////////////////////////// Stop Criterion ////////////////////////////


struct SelectionIterationStopCriterion
{
	SelectionIterationStopCriterion(int maxCount, double delta, int maxIterations) :
		_maxCount(maxCount),
		_delta(delta),
		_maxIterations(maxIterations),
		_currentCount(0),
		_lastBestRank(0.0)
	{
	}

	bool shouldStop(const std::vector<RankedCreature<PrimitiveSelection, SelectionRank>>& population, int iterationCount)
	{
		std::cout << "Iteration " << iterationCount << std::endl;

		if (iterationCount >= _maxIterations)
			return true;

		if (population.empty())
			return true;

		SelectionRank currentBestRank = population[0].rank;

		if (std::abs(currentBestRank.geo_unnormalized - _lastBestRank.geo_unnormalized) < _delta)
		{
			//No change
			_currentCount++;
		}
		else
		{
			_currentCount = 0;
		}

		_lastBestRank = currentBestRank;

		return _currentCount >= _maxCount;
	}

	std::string info() const
	{
		std::stringstream ss;
		ss << "No Change Stop Criterion Selector (maxCount=" << _maxCount << ", delta=" << _delta << ", maxIterations=" << _maxIterations << ")";
		return ss.str();
	}

private:
	int _maxCount;
	int _currentCount;
	int _maxIterations;
	double _delta;
	SelectionRank _lastBestRank;
};

using SelectionTournamentSelector = TournamentSelector<RankedCreature<PrimitiveSelection, SelectionRank>>;
using SelectionGA = GeneticAlgorithm<PrimitiveSelection, SelectionCreator, lmu::SelectionRanker, SelectionRank,
	SelectionTournamentSelector, SelectionIterationStopCriterion, SelectionPopMan>;
using NodeGA = GeneticAlgorithm<PrimitiveSelection, CombinedCreator, lmu::SelectionRanker, SelectionRank,
	SelectionTournamentSelector, SelectionIterationStopCriterion, SelectionPopMan, SelectionSyncPoint>;


std::ostream& lmu::operator<<(std::ostream& out, const SelectionRank& r)
{
	out << "{ 'combined': " << r.combined << ", 'geo': " << r.geo << ", 'size': " << r.size << ", 'geo_unnormalized': " << 
		r.geo_unnormalized << ", 'size_unnormalized': " << r.size_unnormalized << ", 'combined_unnormalized': " << r.combined_unnormalized <<", 'node_ratio': " << r.node_ratio << "}";
	return out;
}

PrimitiveDecomposition lmu::decompose_primitives(const PrimitiveSet& primitives, const ModelSDF& model_sdf, double inside_t, double outside_t, double voxel_size)
{
	PrimitiveSet remaining_primitives, outside_primitives, inside_primitives;
	std::vector<CSGNode> outside_primitive_nodes;
	std::vector<CSGNode> inside_primitive_nodes;

	std::vector<Eigen::Matrix<double, 1, 6>> debug_points;

	
	for (const auto& p : primitives)
	{
		switch (model_sdf.get_dh_type(p, inside_t, outside_t, voxel_size, debug_points, false))
		{
		case DHType::INSIDE:
			inside_primitive_nodes.push_back(geometry(p.imFunc));
			inside_primitives.push_back(p);
			break;
		case DHType::OUTSIDE:
			outside_primitive_nodes.push_back(geometry(p.imFunc));
			outside_primitives.push_back(p);
			break;
		case DHType::NONE:
			remaining_primitives.push_back(p);
			break;
		}
	} 

	auto node = opNo();
	if (inside_primitive_nodes.empty() && outside_primitive_nodes.empty())
	{
		node = opNo();
	}
	else if (!inside_primitive_nodes.empty() && outside_primitive_nodes.empty())
	{
		node = inside_primitive_nodes.size() > 1 ? opUnion(inside_primitive_nodes) : inside_primitive_nodes[0];
	}
	else if (inside_primitive_nodes.empty() && !outside_primitive_nodes.empty())
	{
		node = opDiff({ opNo(),  outside_primitive_nodes.size() > 1 ? opUnion(outside_primitive_nodes) : outside_primitive_nodes[0] });
	}
	else
	{
		node = opDiff({ opUnion(inside_primitive_nodes), opUnion(outside_primitive_nodes) });
	}

	return PrimitiveDecomposition(node, remaining_primitives, inside_primitives, outside_primitives);
}

CSGNode lmu::integrate_node(const CSGNode& into, const PrimitiveSelection& s)
{
	auto res = into;
	std::vector<CSGNode> diff_prims;
	std::vector<CSGNode> union_prims;
	for (int i = 0; i < s.selection.size(); ++i)
	{
		if (s.selection[i].active)
		{
			if (s.selection[i].dh_type == DHType::OUTSIDE)
				diff_prims.push_back(geometry(s.prims->at(i).imFunc));
			else
				union_prims.push_back(geometry(s.prims->at(i).imFunc));
		}
	}

	switch (res.operationType())
	{
	case CSGNodeOperationType::Difference:

		if (res.childsCRef().size() == 2)
		{
			// Insert union nodes into left hand side. 
			if ((!union_prims.empty()))
			{
				if (res.childsRef()[0].operationType() == CSGNodeOperationType::Union)
				{
					res.childsRef()[0].childsRef().insert(res.childsRef()[0].childsRef().end(), union_prims.begin(), union_prims.end());
				}
				else if (res.childsRef()[0].operationType() == CSGNodeOperationType::Noop)
				{
					res.childsRef()[0] = union_prims.size() > 1 ? opUnion(union_prims) : union_prims[0];
				}
				else
				{
					auto n = opUnion({ res.childsRef()[0] });
					n.childsRef().insert(n.childsRef().end(), union_prims.begin(), union_prims.end());
					res.childsRef()[0] = n;
				}
			}

			// Insert difference nodes into right hand side.
			if ((!diff_prims.empty()))
			{
				if (res.childsRef()[1].operationType() == CSGNodeOperationType::Union)
				{
					res.childsRef()[1].childsRef().insert(res.childsRef()[1].childsRef().end(), diff_prims.begin(), diff_prims.end());
				}
				else
				{
					auto n = opUnion({ res.childsRef()[1] });
					n.childsRef().insert(n.childsRef().end(), diff_prims.begin(), diff_prims.end());
					res.childsRef()[1] = n;
				}
			}
		}
		else
		{
			std::cout << "Difference node is malformed." << std::endl;
		}

		break;

	case CSGNodeOperationType::Union:

		res.childsRef().insert(res.childsRef().end(), union_prims.begin(), union_prims.end());

		if (!diff_prims.empty())
		{
			res = opDiff({ res, opUnion(diff_prims) });
		}

		break;


	case CSGNodeOperationType::Noop:

		if (union_prims.empty() && !diff_prims.empty())
		{
			res = opDiff({ res, diff_prims.size() > 1 ? opUnion(diff_prims) : diff_prims[0] });
		}
		else if (!union_prims.empty() && diff_prims.empty())
		{		
			res = union_prims.size() > 1 ? opUnion(union_prims) : union_prims[0];
		}
		else if (!union_prims.empty() && !diff_prims.empty())
		{
			res = opDiff({ union_prims.size() > 1 ? opUnion(union_prims) : union_prims[0], diff_prims.size() > 1 ? opUnion(diff_prims) : diff_prims[0] });
		}

		break;

	default:
		std::cout << "Node configuration not supported." << std::endl;
		break;		
	}
	
	return res;
}

CSGNode lmu::integrate_node(const CSGNode& into, const CSGNode& node)
{
	if (into.operationType() == CSGNodeOperationType::Noop)
	{
		return node;
	} 
	else if (node.operationType() == CSGNodeOperationType::Noop)
	{
		return into;
	}
	else
	{
		if (into.operationType() == CSGNodeOperationType::Difference)
		{
			if (into.childsCRef()[0].operationType() == CSGNodeOperationType::Noop)
			{
				auto n = into;
				n.childsRef()[0] = node;
				return n;
			}
			else if (into.childsCRef()[0].operationType() == CSGNodeOperationType::Union)
			{
				auto n = into;
				n.childsRef()[0].childsRef().push_back(node);
				return n;
			}
			else 
			{
				auto n = into;
				n.childsRef()[0] = opUnion({node, n.childsRef()[0] });
				return n;
			}
		}
		else
		{
			return opUnion({ into, node });
		}
	}
}

#include "cit.h"

lmu::NodeGenerationResult lmu::generate_csg_node(const std::vector<lmu::ImplicitFunctionPtr>& primitives, const std::shared_ptr<ModelSDF>& model_sdf, const CSGNodeGenerationParams& params,
	std::ostream& s1, std::ostream& s2, const lmu::CSGNode& gt_node)
{	
	if (primitives.empty())
	{
		return lmu::opNo();
	}
	else if (primitives.size() == 1)
	{
		return lmu::geometry(primitives[0]);
	}

	NodeGenerationResult gen_res(opNo());

	auto cits = generate_cits(*model_sdf, primitives, model_sdf->voxel_size, false);

	lmu::PointCloud pc(std::get<0>(cits).size(), 6);
	for (int i = 0; i < pc.rows(); ++i)
	{
		pc.row(i) << std::get<0>(cits)[i].transpose(), 0.0, 0.0, 0.0;
	}
	lmu::writePointCloudXYZ("cit_points.xyz", pc);
	
	SelectionRanker ranker(std::get<0>(cits), std::get<1>(cits));

	//SelectionRanker ranker(model_sdf);

	if(primitives.size() == 2)
	{
		return computeForTwoFunctions(primitives, ranker);
	}

	int tournament_k = 2;
	int population_size = 150;
	double mut_prob = 0.4;
	double cross_prob = 0.4;
	double geo_weight = params.geo_weight;
	double size_weight = params.size_weight;

	lmu::PrimitiveSet primitive_set;
	for (const auto& p : primitives)
	{
		switch (p->type())
		{
		case ImplicitFunctionType::Cylinder:
			primitive_set.push_back(Primitive(p, ManifoldSet(), PrimitiveType::Cylinder));
			break;
		case ImplicitFunctionType::Sphere:
			primitive_set.push_back(Primitive(p, ManifoldSet(), PrimitiveType::Sphere));
			break;
		case ImplicitFunctionType::Box:
			primitive_set.push_back(Primitive(p, ManifoldSet(), PrimitiveType::Box));
			break;
		case ImplicitFunctionType::Polytope:
			primitive_set.push_back(Primitive(p, ManifoldSet(), PrimitiveType::Polytope));
			break;
		}
	}

	std::cout << "Groundtruth rank: " << ranker.rank(PrimitiveSelection(gt_node)) << std::endl;

	SelectionTournamentSelector selector_selection(2);
	SelectionTournamentSelector selector_node(2);

	SelectionIterationStopCriterion criterion_selection(params.max_count, 0.000001, params.max_iterations);
	SelectionIterationStopCriterion criterion_node(params.max_count, 0.000001, params.max_iterations);

	SelectionPopMan pop_man_selection(geo_weight, size_weight);
	SelectionPopMan pop_man_node(geo_weight, size_weight);

	SelectionSyncPoint sync_point(params.max_budget, params.node_ratio);


	auto creator_params_selection = params;
	creator_params_selection.node_creator_prob = 0.0;
	CombinedCreator creator_selection(lmu::PrimitiveSelection(&primitive_set), creator_params_selection);

	auto creator_params_node = params;
	creator_params_node.node_creator_prob = 1.0;
	CombinedCreator creator_node(lmu::PrimitiveSelection(&primitive_set), creator_params_node);

	NodeGA ga_selection("selection");
	NodeGA ga_node("node"); 

	NodeGA::Parameters ga_params(population_size, tournament_k, mut_prob, cross_prob, true, Schedule(), Schedule(), true);

	//We only use the selection ga if it is required in the config.
	NodeGA::Result res_node;
	if (params.node_ratio < 1.0)
	{
		auto res_selection_future = ga_selection.runAsync(ga_params, selector_selection, creator_selection, ranker, criterion_selection, pop_man_selection, sync_point);

		res_node = ga_node.run(ga_params, selector_node, creator_node, ranker, criterion_node, pop_man_node, sync_point);

		//ga_selection.stop();
		res_selection_future.wait();
		auto res_selection = res_selection_future.get();

		res_selection.statistics.save(s1, &res_selection.population[0].creature);
	}
	else
	{
		res_node = ga_node.run(ga_params, selector_node, creator_node, ranker, criterion_node, pop_man_node, sync_point);
	}


	gen_res.node = res_node.population[0].creature.is_selection() ? res_node.population[0].creature.to_node() : res_node.population[0].creature.node;
	res_node.statistics.save(s2, &res_node.population[0].creature);

	return gen_res;
}

Mesh lmu::refine(const Mesh& m, const PrimitiveSet& ps)
{
	auto res = m;

	for (int i = 0; i < res.vertices.rows(); ++i)
	{
		Eigen::Vector3d v = res.vertices.row(i).transpose();

		Eigen::Vector4d sd_gr(std::numeric_limits<double>::max(), 0.0, 0.0, 0.0);
		for (const auto& p : ps)
		{
			auto p_sd_gr = p.imFunc->signedDistanceAndGradient(v);

			if (std::abs(sd_gr.x()) > std::abs(p_sd_gr.x()))
			{
				sd_gr = p_sd_gr;
			}			
		}
				
		Eigen::Vector3d delta = sd_gr.bottomRows(3).normalized() * sd_gr.x() * -1.0;
		Eigen::Vector3d new_v = v + delta;

		res.vertices.row(i) << new_v.transpose();
	}

	return res;
}

lmu::SelectionRank::SelectionRank(double v) :
	geo(v),
	size(v),
	combined(v)
{
}

lmu::SelectionRank::SelectionRank(double geo, double size, double combined) :
	geo(geo),
	size(size),
	combined(combined)
{
}

void lmu::SelectionRank::capture_unnormalized()
{
	geo_unnormalized = geo;
	size_unnormalized = size;
	combined_unnormalized = combined;
}

lmu::CSGNodeGenerationParams::CSGNodeGenerationParams()
{
}

lmu::PrimitiveDecomposition::PrimitiveDecomposition(const CSGNode& node, const PrimitiveSet& rem_prims, const PrimitiveSet& in_prims, const PrimitiveSet& out_prims) :
	node(node),
	remaining_primitives(rem_prims),
	inside_primitives(in_prims),
	outside_primitives(out_prims)
{
}

PrimitiveSet lmu::PrimitiveDecomposition::get_primitives(bool all) const
{	
	auto all_prims = remaining_primitives;

	if(all)
	{
		all_prims.insert(all_prims.end(), inside_primitives.begin(), inside_primitives.end());
		all_prims.insert(all_prims.end(), outside_primitives.begin(), outside_primitives.end());
	}
	
	return all_prims;
}

std::vector<DHType> lmu::PrimitiveDecomposition::get_dh_types(bool all) const
{
	std::vector<DHType> dh_types;

	std::transform(remaining_primitives.begin(), remaining_primitives.end(), std::back_inserter(dh_types),
		[](const auto& p) -> DHType { return DHType::NONE; });

	if (all)
	{
		std::transform(inside_primitives.begin(), inside_primitives.end(), std::back_inserter(dh_types),
			[](const auto& p) -> DHType { return DHType::INSIDE; });

		std::transform(outside_primitives.begin(), outside_primitives.end(), std::back_inserter(dh_types),
			[](const auto& p) -> DHType { return DHType::OUTSIDE; });		
	}
	
	return dh_types;
}

lmu::NodeGenerationResult::NodeGenerationResult(const CSGNode & node, const std::vector<Eigen::Matrix<double, 1, 6>>& points) : 
	node(node),
	points(points)
{
}

CSGNode computeForTwoFunctions(const std::vector<ImplicitFunctionPtr>& functions, const lmu::SelectionRanker& ranker)
{
	std::vector<CSGNode> candidates;

	CSGNode un(std::make_shared<UnionOperation>("un"));
	un.addChild(CSGNode(std::make_shared<CSGNodeGeometry>(functions[0])));
	un.addChild(CSGNode(std::make_shared<CSGNodeGeometry>(functions[1])));
	candidates.push_back(un);

	CSGNode inter(std::make_shared<IntersectionOperation>("inter"));
	inter.addChild(CSGNode(std::make_shared<CSGNodeGeometry>(functions[0])));
	inter.addChild(CSGNode(std::make_shared<CSGNodeGeometry>(functions[1])));
	candidates.push_back(inter);

	CSGNode lr(std::make_shared<DifferenceOperation>("lr"));
	lr.addChild(CSGNode(std::make_shared<CSGNodeGeometry>(functions[0])));
	lr.addChild(CSGNode(std::make_shared<CSGNodeGeometry>(functions[1])));
	candidates.push_back(lr);

	CSGNode rl(std::make_shared<DifferenceOperation>("rl"));
	rl.addChild(CSGNode(std::make_shared<CSGNodeGeometry>(functions[1])));
	rl.addChild(CSGNode(std::make_shared<CSGNodeGeometry>(functions[0])));
	candidates.push_back(rl);

	double maxScore = -std::numeric_limits<double>::max();
	const CSGNode* bestCandidate = nullptr;
	for (const auto& candidate : candidates)
	{
		auto curScore = ranker.rank(PrimitiveSelection(candidate)).geo;
		//double curScore2 = ranker.rank(candidate);
		//std::cout << curScore << " " << curScore2 << std::endl;

		if (maxScore < curScore)
		{
			maxScore = curScore;
			bestCandidate = &candidate;
		}
	}

	return *bestCandidate;
}
