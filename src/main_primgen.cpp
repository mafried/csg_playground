#include <igl/opengl/glfw/Viewer.h>

#include "pointcloud.h"
#include "csgnode.h"
#include "csgnode_helper.h"
#include "ransac.h"
#include "mesh.h"
#include "curvature.h"
#include "params.h"
#include <Eigen/Core>

#include <boost/algorithm/string.hpp>

#include <pcl/surface/convex_hull.h>
#include <pcl/common/centroid.h>

// For debugging.
lmu::Mesh g_mesh;
lmu::PointCloud g_pc;
Eigen::MatrixXd g_lines;

#define WITH_VIEWER_GUI

void writePrimitive(const lmu::ImplicitFunctionPtr& prim, const std::string& directory, int iteration, std::unordered_map<lmu::ImplicitFunctionType, int>& primitiveIds)
{
	auto type = prim->type();
	auto fileName = directory + std::to_string(iteration) + "_" + lmu::iFTypeToString(type) + std::to_string(primitiveIds[type]) + ".xyz";
	
	std::cout << "Write primitive to " << fileName << std::endl;
	
	lmu::writePointCloudXYZ(fileName, prim->points());
	primitiveIds[type] = primitiveIds[type] + 1;
}

lmu::CSGNode create_rnd_polytope(const Eigen::Affine3d& trans, const Eigen::Vector3d& c, double r, int num_planes)
{
	// From  http://corysimon.github.io/articles/uniformdistn-on-sphere/
	
	pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 generator(seed);
	std::uniform_real_distribution<double> uniform01(0.0, 1.0);

	for (int i = 0; i < num_planes; i++)
	{
		double theta = 2 * M_PI * uniform01(generator);
		double phi = acos(1 - 2 * uniform01(generator));


		Eigen::Vector3d p(
			r * sin(phi) * cos(theta),
			r * sin(phi) * sin(theta),
			r * cos(phi)
		);

		point_cloud->points.push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
	}

	pcl::PointXYZ centroid;
	pcl::computeCentroid(*point_cloud, centroid);
	pcl::ConvexHull<pcl::PointXYZ> chull;
	pcl::PointCloud<pcl::PointXYZ>::Ptr hull_points(new pcl::PointCloud<pcl::PointXYZ>);
	std::vector<pcl::Vertices> hull_polygon;

	chull.setInputCloud(point_cloud);
	chull.reconstruct(*hull_points, hull_polygon);

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

		// Not all normals have the correct orientation.
		if (n.dot(Eigen::Vector3d(centroid.x, centroid.y, centroid.z) - p) > 0)
			n = -n;

		_p.push_back(p);
		_n.push_back(n);
	}

#ifdef _DEBUG 
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
		g_lines.row(i) << (trans *_p[i]).transpose(), (trans * (_p[i] + _n[i])).transpose(), 1.0, 0.0, 0.0;
#endif

	return lmu::geo<lmu::IFPolytope>(trans, _p, _n, "");
}

lmu::CSGNode selectPrimitive(const Eigen::Affine3d& trans, const Eigen::Vector3d& dims, 
	const std::vector<double>& prim_probs, int min_polytope_planes, int max_polytope_planes, double primitive_scaling)
{
	static std::random_device rd;     
	static std::mt19937 rng(rd());   

	std::discrete_distribution<int> uni_type(prim_probs.begin(), prim_probs.end());
	std::uniform_int_distribution<int> uni_planes(min_polytope_planes, max_polytope_planes);

	int num_planes = uni_planes(rng);
	int type = uni_type(rng);
	
	auto t1 = Eigen::Affine3d::Identity();
	t1.rotate(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d(0.0, 0.0, 1.0)));
	t1.scale(-1.0 * primitive_scaling);
	t1.translate(Eigen::Vector3d(-dims.maxCoeff() / 2.0,0,0));

	auto t2 = Eigen::Affine3d::Identity();
	t2.rotate(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d(0.0, 0.0, 1.0)));
	t2.scale(1.0 * primitive_scaling);
	t2.translate(Eigen::Vector3d(-dims.maxCoeff() / 2.0, 0, 0));

	auto t3 = Eigen::Affine3d::Identity();
	t3.scale(-1.0 * primitive_scaling);
	t3.translate(Eigen::Vector3d(-dims.maxCoeff(), 0, 0));

	switch (type)
	{
	case 0:
		return lmu::opPrim({ lmu::geo<lmu::IFBox>(trans, Eigen::Vector3d(dims.z(), dims.x(), dims.y()) * primitive_scaling, 2, "") });
	case 1:
		return lmu::opPrim({ lmu::geo<lmu::IFSphere>(trans, dims.x() * primitive_scaling,"") });
	case 2:
		return lmu::opPrim({ lmu::geo<lmu::IFTorus>(trans, dims.minCoeff() / 2.0 * primitive_scaling, dims.maxCoeff() / 2.0 * primitive_scaling, "") });
	case 3:
		return lmu::opPrim({ 
			lmu::geo<lmu::IFCylinder>(trans, dims.minCoeff() / 2.0 * primitive_scaling, 100, ""),
			lmu::geo<lmu::IFPlane>(trans * t1, ""),
			lmu::geo<lmu::IFPlane>(trans * t2, "")
			});
	case 4: 
		return lmu::opPrim({ 
			lmu::geo<lmu::IFCone>(trans, 2.0 * std::atan(dims.minCoeff() / 2.0 / dims.maxCoeff()), 100.0, ""),
			lmu::geo<lmu::IFPlane>(trans * t3, "")
		});
	case 5:
		return lmu::opPrim({ create_rnd_polytope(trans, Eigen::Vector3d(0,0,0), dims.maxCoeff() / 2.0, num_planes)});
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
		if (argc != 2)
		{
			std::cerr << "Must have a config ini file path as argument." << std::endl;
			return -1;
		}


		// Load config.

		auto config_file = std::string(argv[1]);

		lmu::ParameterSet s(config_file);

		auto input_file = s.getStr("Input", "File","");
		auto input_file_type = s.getStr("Input", "Type", "csg");

		auto input_sampling_rate = s.getDouble("Input", "Sampling.Rate", 0.01);
		auto input_sampling_max_d = s.getDouble("Input", "Sampling.MaxPosDistance", 0.01);
		auto input_sampling_max_angle_d = s.getDouble("Input", "Sampling.MaxAngleDistance", 0.01);
		auto input_sampling_point_cloud_size = s.getInt("Input", "Sampling.PointCloudSize", 1024);
		
		auto output_folder = s.getStr("Output", "Folder", "");

		auto output_sampling_rate = s.getDouble("Output", "Sampling.Rate", 0.01);
		auto output_sampling_max_d = s.getDouble("Output", "Sampling.MaxPosDistance", 0.01);
		auto output_sampling_max_angle_d = s.getDouble("Output", "Sampling.MaxAngleDistance", 0.01);
		auto output_sampling_point_cloud_size = s.getInt("Output", "Sampling.PointCloudSize", 1024);
		auto output_sampling_pos_noise = s.getDouble("Output", "Sampling.PosNoise", 0.0);
		auto output_sampling_angle_noise = s.getDouble("Output", "Sampling.AngleNoise", 0.0);

		auto k = s.getInt("Generator", "K", 1);
		auto iterations = s.getInt("Generator", "Iterations", 1);
		auto primitive_scaling = s.getDouble("Generator", "PrimitiveScaling", 1.0);

		auto sphere_p = s.getDouble("Generator", "Sphere", 1.0);
		auto box_p = s.getDouble("Generator", "Box", 1.0);
		auto cone_p = s.getDouble("Generator", "Cone", 1.0);
		auto cylinder_p = s.getDouble("Generator", "Cylinder", 1.0);
		auto torus_p = s.getDouble("Generator", "Torus", 1.0);
		auto polytope_p = s.getDouble("Generator", "Polytope", 1.0);
		auto polytope_min_planes = s.getInt("Generator", "Polytope.MinPlanes", 6);
		auto polytope_max_planes = s.getInt("Generator", "Polytope.MaxPlanes", 16);

		std::vector<double> prim_probs = { box_p, sphere_p, torus_p, cylinder_p, cone_p, polytope_p };

	
		// Load object from node json or object mesh file.

		lmu::PointCloud pc;
		if (input_file_type == "csg")
		{
			lmu::CSGNode node = lmu::fromJSONFile(input_file);
			lmu::CSGNodeSamplingParams params(input_sampling_max_d, input_sampling_max_angle_d, 0.0, input_sampling_rate);
			pc = lmu::computePointCloud(node, params);
		}
		else if (input_file_type == "obj")
		{			
			auto mesh = to_canonical_frame(lmu::fromOBJFile(input_file));
			pc = lmu::pointCloudFromMesh(mesh, input_sampling_max_d, input_sampling_rate, 0.0);
		}
		else if (input_file_type == "off")
		{
			auto mesh = to_canonical_frame(lmu::fromOFFFile(input_file));
			pc = lmu::pointCloudFromMesh(mesh, input_sampling_max_d, input_sampling_rate, 0.0);
		}
		else
		{
			std::cerr << "Wrong model type '" << input_file_type << "'.";
			return -1;
		}

		// Initialize viewer.
#ifdef WITH_VIEWER_GUI
		igl::opengl::glfw::Viewer viewer;
		viewer.mouse_mode = igl::opengl::glfw::Viewer::MouseMode::Rotation;
#endif 
		
		pc = lmu::to_canonical_frame(pc);
		pc = lmu::farthestPointSampling(pc, input_sampling_point_cloud_size);
		std::cout << "Input point cloud: " << pc.rows() << " Dims: " << pc.leftCols(3).colwise().maxCoeff() << " " << pc.leftCols(3).colwise().minCoeff() << std::endl;

		auto clusters = lmu::kMeansClustering(pc, k);

		std::cout << "Clustering done." << std::endl;

		for (int iter = 0; iter < iterations; iter++)
		{

			// Create primitives.

			auto model = lmu::opUnion();
			for (const auto& cluster : clusters) {

				auto clusterSize = std::get<1>(cluster).rows();
				if (clusterSize == 0)
					continue;
			
				auto dims = lmu::computeOBBDims(std::get<1>(cluster));
			
				auto trans = lmu::getOrientation(std::get<1>(cluster));

				auto geo = selectPrimitive(trans, dims, prim_probs, polytope_min_planes, polytope_max_planes, primitive_scaling);

				model.addChild(geo);				
			}

			auto model_pc = lmu::computePointCloud(model, 
				lmu::CSGNodeSamplingParams(output_sampling_max_d, output_sampling_max_angle_d, 0.0, output_sampling_rate, Eigen::Vector3d(-0.5,-0.5,-0.5), Eigen::Vector3d(1.5,1.5,1.5)));
						
			model_pc = lmu::farthestPointSampling(model_pc, output_sampling_point_cloud_size);

			#ifdef _DEBUG 
			viewer.data().add_points(model_pc.leftCols(3), model_pc.rightCols(3));
			viewer.data().add_points(g_pc.leftCols(3), g_pc.rightCols(3));
			viewer.data().set_mesh(g_mesh.vertices, g_mesh.indices);			
			viewer.data().lines = g_lines;
			viewer.data().show_lines = true;
			#endif

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

			model_pc = lmu::to_canonical_frame(model_pc);
			model_pc = lmu::add_gaussian_noise(model_pc, output_sampling_pos_noise, output_sampling_angle_noise);

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

			std::cout << "Output point cloud: " << model_pc.rows() << " Dims: " << (model_pc.leftCols(3).colwise().maxCoeff() - model_pc.leftCols(3).colwise().minCoeff()) << std::endl;

			auto primFile = output_folder + std::to_string(iter) + "_prim.prim";
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
						
			auto pcFile = output_folder + std::to_string(iter) +"_pc.xyz";
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
