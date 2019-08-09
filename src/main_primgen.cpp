#include <igl/opengl/glfw/Viewer.h>

#include "pointcloud.h"
#include "csgnode.h"
#include "csgnode_helper.h"
#include "ransac.h"
#include "mesh.h"
#include "curvature.h"
#include <Eigen/Core>

#include <boost/algorithm/string.hpp>


#define WITH_VIEWER_GUI

std::vector<lmu::ImplicitFunctionPtr> splitCylinder(const lmu::ImplicitFunctionPtr& cyl)
{
	const double h = 0.01;

	auto cylNode = lmu::geometry(cyl);
	std::vector<lmu::ImplicitFunctionPtr> splitted;

	std::vector<Eigen::Matrix<double, 1, 6>> cylPoints;
	std::vector<Eigen::Matrix<double, 1, 6>> box1Points;
	std::vector<Eigen::Matrix<double, 1, 6>> box2Points;

	Eigen::Vector3d refNormal(0,0,0);

	for (int i = 0; i < cyl->pointsCRef().rows(); ++i)
	{
		auto p = cyl->pointsCRef().row(i);

		lmu::Curvature c = lmu::curvature(p.leftCols(3).transpose(), cylNode, h);
		//std::cout << c.k1 << " " << c.k2 << std::endl;
		if (std::abs(c.k1) < h && std::abs(c.k2) < h)
		{
			if (refNormal == Eigen::Vector3d(0, 0, 0))
				refNormal = p.rightCols(3).transpose();

			if (refNormal.dot(p.rightCols(3).transpose()) <= 0.0)
				box1Points.push_back(p);
			else
				box2Points.push_back(p);
		}
		else
		{
			cylPoints.push_back(p);
		}
	}

	auto box1 = std::make_shared<lmu::IFBox>(Eigen::Affine3d::Identity(), Eigen::Vector3d(0, 0, 0), 2, ""); // Parameters do not matter;
	auto box2 = std::make_shared<lmu::IFBox>(Eigen::Affine3d::Identity(), Eigen::Vector3d(0, 0, 0), 2, ""); // Parameters do not matter;

	cyl->points() = lmu::pointCloudFromVector(cylPoints);
	box1->points() = lmu::pointCloudFromVector(box1Points);
	box2->points() = lmu::pointCloudFromVector(box2Points);

	splitted.push_back(cyl);
	splitted.push_back(box1);
	splitted.push_back(box2);

	return splitted;
}

void writePrimitive(const lmu::ImplicitFunctionPtr& prim, const std::string& directory, int iteration, std::unordered_map<lmu::ImplicitFunctionType, int>& primitiveIds)
{
	auto type = prim->type();
	auto fileName = directory + std::to_string(iteration) + "_" + lmu::iFTypeToString(type) + std::to_string(primitiveIds[type]) + ".xyz";
	
	std::cout << "Write primitive to " << fileName << std::endl;
	
	lmu::writePointCloudXYZ(fileName, prim->points());
	primitiveIds[type] = primitiveIds[type] + 1;
}

lmu::CSGNode selectPrimitive(const Eigen::Affine3d& trans, const Eigen::Vector3d& dims, const std::unordered_set<int>& types)
{
	static std::random_device rd;     
	static std::mt19937 rng(rd());   
	std::uniform_int_distribution<int> uni(0, 2); 

	int type = uni(rng);
	while (types.count(type) == 0)
		type = uni(rng);

	switch (type)
	{
	case 0:
		return lmu::geo<lmu::IFBox>(trans, Eigen::Vector3d(dims.z(), dims.x(), dims.y()), 2, "");
	case 1: 
		return lmu::geo<lmu::IFSphere>(trans, dims.x() / 2.0,"");
	default:
	case 2: 		
		return lmu::geo<lmu::IFCylinder>(trans, dims.minCoeff() / 2.0, dims.maxCoeff(), "");
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
		//auto mesh = lmu::fromOFFFile(modelPath);
		//lmu::scaleMesh(mesh, 1.0);
		//viewer.data().set_mesh(mesh.vertices, mesh.indices);

		std::cout << "Input point cloud size: " << pc.rows() << std::endl;
		//pc = lmu::farthestPointSampling(pc, pointCloudSize);
		//std::cout << "Reduced to " << pc.rows() << " Should be: " << (pointCloudSize) << std::endl;

		std::cout << "Clustering" << std::endl;
		
		auto clusters = lmu::kMeansClustering(pc, k);
	
		for (int iter = 0; iter < maxIterations; iter++)
		{
			auto model = lmu::opUnion();
			int i = 0;
			for (const auto& cluster : clusters) {

				auto clusterSize = std::get<1>(cluster).rows();
				std::cout << "Cluster Size: " << clusterSize << std::endl;
				if (clusterSize == 0)
					continue;

				auto dims = lmu::computeOBBDims(std::get<1>(cluster));

				//i++;
				//double c = (double)i / (double)clusters.size();
				//viewer.data().add_points(std::get<1>(cluster).leftCols(3), 
				//	Eigen::RowVector3d(c, 0, 0).replicate(std::get<1>(cluster).rows(),1));

				//std::cout << "Dims: " << dims << std::endl;

				auto trans = lmu::getOrientation(std::get<1>(cluster));

				auto geo = selectPrimitive(trans, dims, types);//lmu::geo<lmu::IFCylinder>(trans, dims.minCoeff() / 2.0, dims.maxCoeff(), "");
				if (db(rng, parmb_t{ cutOutProb }))
				{
					std::cout << "As cutout." << std::endl;
					model.addChild(lmu::opComp({ geo }));
				}
				else
				{
					model.addChild(geo);
				}
			}

			auto modelPC = lmu::computePointCloud(model, params);

			// Apply pseudo RANSAC to segment point cloud per primitive.
			double ratio = lmu::ransacWithSim(modelPC, params, model);
			std::cout << "RANSAC Sim Ratio: " << ratio << std::endl;
			
			// Split cylinders.
			auto prims = lmu::allDistinctFunctions(model);
			std::vector<lmu::ImplicitFunctionPtr> splittedPrims;
			for (const auto& prim : prims)
			{
				if (prim->type() == lmu::ImplicitFunctionType::Cylinder)
				{
					auto res = splitCylinder(prim);
					splittedPrims.insert(splittedPrims.end(), res.begin(), res.end());
				}
				else
				{
					splittedPrims.push_back(prim);
				}
			}
			prims = splittedPrims;

			auto primFile = outputFolder + std::to_string(iter) + "_prim.prim";
			std::ofstream ps(primFile);

			//Write primitive point clouds to file.
			i = 0;
			int pointIdx = 0;
			std::unordered_map<lmu::ImplicitFunctionType, int> primitiveIds;
			std::vector<Eigen::Matrix<double, 1, 8>> points;
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
