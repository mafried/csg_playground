#include <igl/opengl/glfw/Viewer.h>

#include "pointcloud.h"
#include "csgnode.h"
#include "csgnode_helper.h"
#include "ransac.h"
#include "mesh.h"
#include "curvature.h"
#include <Eigen/Core>

#include <boost/algorithm/string.hpp>

#include <pcl/surface/convex_hull.h>
#include <pcl/common/centroid.h>


#define WITH_VIEWER_GUI

void writePrimitive(const lmu::ImplicitFunctionPtr& prim, const std::string& directory, int iteration, std::unordered_map<lmu::ImplicitFunctionType, int>& primitiveIds)
{
	auto type = prim->type();
	auto fileName = directory + std::to_string(iteration) + "_" + lmu::iFTypeToString(type) + std::to_string(primitiveIds[type]) + ".xyz";
	
	std::cout << "Write primitive to " << fileName << std::endl;
	
	lmu::writePointCloudXYZ(fileName, prim->points());
	primitiveIds[type] = primitiveIds[type] + 1;
}

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/convex_hull_3.h>

typedef CGAL::Simple_cartesian<double>            K;
typedef CGAL::Polyhedron_3<K>                     Polyhedron_3;
typedef K::Point_3                                Point_3;
typedef K::Plane_3						          Plane_3;

struct Plane_equation {
	template <class Facet>
	typename Facet::Plane_3 operator()(Facet& f) {
		typename Facet::Halfedge_handle h = f.halfedge();
		typedef typename Facet::Plane_3  Plane;
		return Plane(h->vertex()->point(),
			h->next()->vertex()->point(),
			h->next()->next()->vertex()->point());
	}
};

lmu::Mesh g_mesh;
lmu::PointCloud g_pc;
Eigen::MatrixXd g_lines;

lmu::CSGNode create_rnd_polytope(const Eigen::Affine3d& trans, const Eigen::Vector3d& c, double r, int num_planes)
{

	// From  http://corysimon.github.io/articles/uniformdistn-on-sphere/
	//r = 1.0;

	//for (int i = 0; i < 100; i++)
	//{

	std::cout << "----------------" << std::endl;

	std::vector<Point_3> points;
	pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>);


	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 generator(seed);
	std::uniform_real_distribution<double> uniform01(0.0, 1.0);

	for (int i = 0; i < num_planes; i++)
	{
		double theta = 2 * M_PI * uniform01(generator);
		double phi = acos(1 - 2 * uniform01(generator));

		Point_3 p(
			r * sin(phi) * cos(theta),
			r * sin(phi) * sin(theta),
			r * cos(phi)
		);

		points.push_back(p);

		point_cloud->points.push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
	}

	pcl::PointXYZ centroid;
	pcl::computeCentroid(*point_cloud, centroid);

	std::cout << "=>" << points.size() << std::endl;

	/*points =
	{
		Point_3(0.113904, -0.0759709, 0.025244),
		Point_3(0.0449047, 0.0960923, -0.0901815),
		Point_3(0.0603977, -0.105431, -0.0679646),
		Point_3(-0.031272, -0.0929492, 0.0988199),
		Point_3(0.0761061, -0.0619027, 0.0987865),
		Point_3(0.00195516, -0.0964902, 0.100343),
		Point_3(-0.123282, 0.0265094, -0.0590055),
		Point_3(0.120172, -0.0540623, 0.0449309)
	};*/

	std::cout << "points: " << std::endl;
	std::cout << "{" << std::endl;
	for (const auto& p : points) std::cout << "Point_3(" << p.x() << ", " << p.y() << ", " << p.z() << ")," << std::endl;
	std::cout << "};" << std::endl;

	/*
	Polyhedron_3 ph;
	CGAL::convex_hull_3(points.begin(), points.end(), ph);

	std::cout << "Size of: " << ph.size_of_vertices() << std::endl;

	std::transform(ph.facets_begin(), ph.facets_end(), ph.planes_begin(), Plane_equation());
	*/
	pcl::ConvexHull<pcl::PointXYZ> chull;
	pcl::PointCloud<pcl::PointXYZ>::Ptr hull_points(new pcl::PointCloud<pcl::PointXYZ>);
	std::vector<pcl::Vertices> hull_polygon;

	chull.setInputCloud(point_cloud);
	chull.reconstruct(*hull_points, hull_polygon);
	std::cout << "DIMS: " << chull.getDimension() << std::endl;

	std::vector<Eigen::Vector3d> _p;
	std::vector<Eigen::Vector3d> _n;

	for (const auto& p : hull_polygon)
	{
		auto p1 = hull_points->points[p.vertices[0]];
		auto p2 = hull_points->points[p.vertices[1]];
		auto p3 = hull_points->points[p.vertices[2]];

		Eigen::Vector3d v1(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
		Eigen::Vector3d v2(p1.x - p3.x, p1.y - p3.y, p1.z - p3.z);
		auto n = v2.cross(v1).normalized();
		auto p = Eigen::Vector3d((p1.x + p2.x + p3.x) / 3.0, (p1.y + p2.y + p3.y) / 3.0, (p1.z + p2.z + p3.z) / 3.0);

		if (n.dot(Eigen::Vector3d(centroid.x, centroid.y, centroid.z) - p) > 0)
			n = -n;

		_p.push_back(p);
		_n.push_back(n);

		//std::cout << "P: " << _p.back().transpose() << " " << _n.back().transpose() << std::endl;
	}

	g_mesh.vertices = Eigen::MatrixXd(hull_points->points.size(), 3);
	for (int i = 0; i < hull_points->points.size(); ++i)
		g_mesh.vertices.row(i) << hull_points->points[i].x, hull_points->points[i].y, hull_points->points[i].z;

	g_mesh.indices = Eigen::MatrixXi(hull_polygon.size(), 3);
	for (int i = 0; i < hull_polygon.size(); ++i)
		g_mesh.indices.row(i) << hull_polygon[i].vertices[0], hull_polygon[i].vertices[1], hull_polygon[i].vertices[2];

	g_mesh.transform = trans;
	lmu::transform(g_mesh);

	g_pc = lmu::PointCloud(_p.size(), 6);
	for (int i = 0; i < _p.size(); ++i)
		g_pc.row(i) << _p[i].transpose(), 1.0, 0.0, 0.0;

	g_lines = Eigen::MatrixXd(_n.size(), 9);
	for (int i = 0; i < _n.size(); ++i)
		g_lines.row(i) << _p[i].transpose(), (_p[i] + _n[i]).transpose(), 1.0, 0.0, 0.0;


	/*
	for (auto plane = ph.planes_begin(); plane != ph.planes_end(); ++plane)
	{
		_p.push_back(Eigen::Vector3d(plane->point().x(), plane->point().y(), plane->point().z()));
		_n.push_back(Eigen::Vector3d(plane->orthogonal_vector().x(), plane->orthogonal_vector().y(), plane->orthogonal_vector().z()));
	}
	*/

	std::cout << "# planes: " << _p.size() << std::endl;

	return lmu::geo<lmu::IFPolytope>(trans, _p, _n, "");
		

	std::cout << "----------------" << std::endl;
	//}
	
	
}

lmu::CSGNode selectPrimitive(const Eigen::Affine3d& trans, const Eigen::Vector3d& dims, const std::unordered_set<int>& types)
{
	static std::random_device rd;     
	static std::mt19937 rng(rd());   
	std::uniform_int_distribution<int> uni(5, 5); 

	int type = uni(rng);
	while (types.count(type) == 0)
		type = uni(rng);

	auto t1 = Eigen::Affine3d::Identity();
	t1.rotate(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d(0.0, 0.0, 1.0)));
	t1.scale(-1.0);
	t1.translate(Eigen::Vector3d(-dims.maxCoeff() / 2.0,0,0));

	auto t2 = Eigen::Affine3d::Identity();
	t2.rotate(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d(0.0, 0.0, 1.0)));
	t2.scale(1.0);
	t2.translate(Eigen::Vector3d(-dims.maxCoeff() / 2.0, 0, 0));

	auto t3 = Eigen::Affine3d::Identity();
	t3.scale(-1.0);
	t3.translate(Eigen::Vector3d(-dims.maxCoeff(), 0, 0));

	//std::cout << "P1: " << lmu::geo<lmu::IFPlane>(t1, "").signedDistanceAndGradient(Eigen::Vector3d(0,0,0)).transpose() << std::endl;
	//std::cout << "P2: " << lmu::geo<lmu::IFPlane>(t2, "").signedDistanceAndGradient(Eigen::Vector3d(0,0,0)).transpose() << std::endl;
	
	std::cout << "Type: " << type << std::endl;

	switch (type)
	{
	case 0:
		return lmu::opPrim({ lmu::geo<lmu::IFBox>(trans, Eigen::Vector3d(dims.z(), dims.x(), dims.y()), 2, "") });
	case 1:
		return lmu::opPrim({ lmu::geo<lmu::IFSphere>(trans, dims.x(),"") });
	case 2:
		return lmu::opPrim({ lmu::geo<lmu::IFTorus>(trans, dims.minCoeff() / 2.0, dims.maxCoeff() / 2.0 , "") });
	case 3:
		return lmu::opPrim({ 
			lmu::geo<lmu::IFCylinder>(trans, dims.minCoeff() / 2.0, 100, ""),
			lmu::geo<lmu::IFPlane>(trans * t1, ""),
			lmu::geo<lmu::IFPlane>(trans * t2, "")
			});
	case 4: 
		return lmu::opPrim({ 
			lmu::geo<lmu::IFCone>(trans, 2.0 * std::atan(dims.minCoeff() / 2.0 / dims.maxCoeff()), 100, ""),
			lmu::geo<lmu::IFPlane>(trans * t3, "")
		});
	case 5:
		return lmu::opPrim({ create_rnd_polytope(trans, Eigen::Vector3d(0,0,0), dims.maxCoeff() / 2.0, 8)});		
	}
}

int main(int argc, char** argv)
{
	static std::random_device rd;
	static std::mt19937 rng(rd());
	static std::bernoulli_distribution db{};
	using parmb_t = decltype(db)::param_type;

	try
	{
		if (argc != 13)
		{
			std::cerr << "Must have 12 arguments." << std::endl;
			return -1;
		}

		double samplingRate = 0.02;
		double errorSigma = 0.0;
		double maxDistance = 0.01;
		double maxAngleDistance = 0.01;
		std::string modelType = "";
		std::string modelPath = "";
		std::string outputFolder = "";
		int k = 1;
		int pointCloudSize = 1024;
		double cutOutProb = 0.1;
		int maxIterations = 1;
		bool withLabels = false;
		std::unordered_set<int> types;

		modelType = std::string(argv[1]);
		modelPath = std::string(argv[2]);
		outputFolder = std::string(argv[3]);
		samplingRate = std::stod(argv[4]);
		maxDistance = std::stod(argv[5]);
		maxAngleDistance = std::stod(argv[6]);
		errorSigma = std::stod(argv[7]);
		k = std::stoi(argv[8]);
		pointCloudSize = std::stoi(argv[9]);
		cutOutProb = std::stod(argv[10]);
		maxIterations = std::stoi(argv[11]);

		std::unordered_set<std::string> strTypes;
		boost::split(strTypes, std::string(argv[12]), [](char c) {return c == ';'; });
		std::transform(strTypes.begin(), strTypes.end(), std::inserter(types, types.end()),
			[](const std::string& s) -> int { std::cout << "Primitive Type: " << s << std::endl; return  std::stoi(s); });


		lmu::CSGNodeSamplingParams params(maxDistance, maxAngleDistance, errorSigma);

		double preSamplingFactor = 5.0;

		std::cout << "CSG: " << modelType << std::endl;

		// Load object from node json or object mesh file.
		lmu::PointCloud pc;
		if (modelType == "csg")
		{
			lmu::CSGNode node = lmu::fromJSONFile(modelPath);
			pc = lmu::computePointCloud(node, params);
		}
		else if (modelType == "obj")
		{
			auto mesh = lmu::fromOBJFile(modelPath);
			lmu::scaleMesh(mesh, 1.0);
			pc = lmu::pointCloudFromMesh(mesh, maxDistance * preSamplingFactor, samplingRate * preSamplingFactor, errorSigma * preSamplingFactor);
		}
		else if (modelType == "off")
		{
			auto mesh = lmu::fromOFFFile(modelPath);
			lmu::scaleMesh(mesh, 1.0);
			pc = lmu::pointCloudFromMesh(mesh, maxDistance * preSamplingFactor, samplingRate * preSamplingFactor, errorSigma * preSamplingFactor);
		}
		else
		{
			std::cerr << "Wrong model type '" << modelType << "'.";
			return -1;
		}

		// Initialize viewer.
#ifdef WITH_VIEWER_GUI
		igl::opengl::glfw::Viewer viewer;
		viewer.mouse_mode = igl::opengl::glfw::Viewer::MouseMode::Rotation;
#endif 
		
		pc = lmu::normalize(pc);
		pc = lmu::farthestPointSampling(pc, pointCloudSize);
		std::cout << "Input point cloud: " << pc.rows() << std::endl;

		auto clusters = lmu::kMeansClustering(pc, k);

		std::cout << "Clustering done." << std::endl;

		for (int iter = 0; iter < maxIterations; iter++)
		{
			// Create primitives.

			auto model = lmu::opUnion();
			for (const auto& cluster : clusters) {

				auto clusterSize = std::get<1>(cluster).rows();
				if (clusterSize == 0)
					continue;
			
				auto dims = lmu::computeOBBDims(std::get<1>(cluster));
			
				auto trans = lmu::getOrientation(std::get<1>(cluster));

				auto geo = selectPrimitive(trans, dims, types);

				model.addChild(geo);				
			}

			auto model_pc = lmu::computePointCloud(model, 
				lmu::CSGNodeSamplingParams(0.01, 0.01, 0.0,0.0, Eigen::Vector3d(-1,-1,-1), Eigen::Vector3d(1,1,1)));
						
			model_pc = lmu::farthestPointSampling(model_pc, pointCloudSize);
			viewer.data().add_points(model_pc.leftCols(3), model_pc.rightCols(3));

			viewer.data().add_points(g_pc.leftCols(3), g_pc.rightCols(3));

			viewer.data().set_mesh(g_mesh.vertices, g_mesh.indices);
			
			viewer.data().lines = g_lines;
			viewer.data().show_lines = true;

			goto VIEWER;
					
			auto prim_ops = model.childs();
			
			// Assign points to primitive operations.

			std::unordered_map<int, std::vector<int>> point_indices_per_prim_op;
			for (int i = 0; i < model_pc.rows(); ++i)
			{
				int closest_prim_op_idx = -1;
				double closest_prim_op_dist = std::numeric_limits<double>::max();

				Eigen::Vector3d p = model_pc.row(i).leftCols(3).transpose();
				Eigen::Vector3d n = model_pc.row(i).rightCols(3).transpose();

				for (int j = 0; j < prim_ops.size(); ++j)
				{				
					double abs_d = std::abs(prim_ops[j].signedDistance(p));
					if (abs_d < closest_prim_op_dist)
					{
						closest_prim_op_idx = j;
						closest_prim_op_dist = abs_d;
					}
				}
				point_indices_per_prim_op[closest_prim_op_idx].push_back(i);
			}

			// Assign per-primitive operation points to closest primitive in primitive op. 
			
			std::unordered_map<lmu::ImplicitFunctionPtr, std::vector<int>> point_indices_per_prim;
			for (const auto& pi : point_indices_per_prim_op)
			{				
				for (int i = 0; i < pi.second.size(); ++i)
				{
					int point_index = pi.second[i];
					Eigen::Vector3d p = model_pc.row(point_index).leftCols(3).transpose();
					Eigen::Vector3d n = model_pc.row(point_index).rightCols(3).transpose();

					lmu::ImplicitFunctionPtr closest_prim = nullptr;
					double closest_prim_dist = std::numeric_limits<double>::max();

					for (const auto& prim : prim_ops[pi.first].childsCRef())
					{
						double abs_d = std::abs(prim.signedDistance(p));
						if (abs_d < closest_prim_dist)
						{
							closest_prim = prim.function();
							closest_prim_dist = abs_d;
						}
					}
					point_indices_per_prim[closest_prim].push_back(point_index);
				}
			}

			model_pc = lmu::normalize(model_pc);
			model_pc = lmu::add_gaussian_noise(model_pc, 0.001, 0.001);

			std::unordered_map<lmu::ImplicitFunctionPtr, std::vector<Eigen::Matrix<double, 1, 6>>> points_per_prim;
			for (const auto& pp : point_indices_per_prim)
			{
				std::transform(pp.second.begin(), pp.second.end(), std::back_inserter(points_per_prim[pp.first]),
					[&model_pc](int i) {return model_pc.row(i); });
			}
			
			for (const auto& pair : points_per_prim)
			{
				pair.first->setPoints(lmu::pointCloudFromVector(pair.second));
			}
			
			// Save primitives. 

			std::cout << "Output point cloud: " << model_pc.rows() << std::endl;

			auto primFile = outputFolder + std::to_string(iter) + "_prim.prim";
			std::ofstream ps(primFile);

			//Write primitive point clouds to file.
			int i = 0;
			int pointIdx = 0;
			std::unordered_map<lmu::ImplicitFunctionType, int> primitiveIds;
			std::vector<Eigen::Matrix<double, 1, 8>> points;
			auto prims = lmu::allDistinctFunctions(model);
			for (const auto& prim : prims)
			{
				//writePrimitive(prim, outputFolder, iter, primitiveIds);

				for (int j = 0; j < prim->pointsCRef().rows(); ++j)
				{
					auto point = Eigen::Matrix<double, 1, 8>();
					point << prim->pointsCRef().row(j), (double)prim->type(), (double)i;
					points.push_back(point);
				}

				ps << (int)prim->type() << std::endl;
				ps << pointIdx << " " << prim->pointsCRef().rows() << std::endl;
				ps << prim->serializeTransform() << std::endl;
				ps << prim->serializeParameters() << std::endl;
				
				pointIdx += prim->pointsCRef().rows();
				i++;

#ifdef WITH_VIEWER_GUI
				double c = (double)i / (double)prims.size();
				viewer.data().add_points(prim->pointsCRef().leftCols(3),
				Eigen::RowVector3d(c, 0, 0).replicate(prim->pointsCRef().rows(), 1));
#endif
			}
			

			//auto mesh2 = lmu::computeMesh(model, Eigen::Vector3i(50, 50, 50));
			//viewer.data().set_mesh(mesh2.vertices, mesh2.indices);

			//viewer.data().set_points(modelPC.leftCols(3), modelPC.rightCols(3));
			auto pcFile = outputFolder + std::to_string(iter) +"_pc.xyz";
			std::cout << "Write point cloud to " << pcFile << "." << std::endl;
			std::ofstream s(pcFile);
			s << points.size() << " " << 8 << std::endl;
			for (int i = 0; i < points.size(); i++)
			{
				for (int j = 0; j < points[i].cols(); j++)
					s << points[i].col(j) << " ";
				s << std::endl;
			}
			
		} 

#ifdef WITH_VIEWER_GUI
		VIEWER:
		viewer.data().point_size = 5.0;
		viewer.core.background_color = Eigen::Vector4f(1, 1, 1, 1);
		viewer.launch();
#endif 
	}
	catch (const std::exception& ex)
	{
		std::cerr << "An exception occured: " << std::string(ex.what()) << std::endl;
		return -1;
	}

	return 0;
}
