/*

#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <sstream>
#include <algorithm>
#include <limits>
#include <cmath>

#include <PointCloud.h>

// For RANSAC
#include <RansacShapeDetector.h>
#include <PlanePrimitiveShapeConstructor.h>
#include <CylinderPrimitiveShapeConstructor.h>
#include <SpherePrimitiveShapeConstructor.h>
#include <ConePrimitiveShapeConstructor.h>
#include <TorusPrimitiveShapeConstructor.h>

#include <PlanePrimitiveShape.h> // for PlanePrimitiveShape

#include <basic.h> // for Vec3f


typedef std::vector< std::pair< MiscLib::RefCountPtr< PrimitiveShape >, size_t > > ShapeVector;


void usage(const std::string& name) {
  std::cout << "Usage: " << std::endl;
  std::cout << name << " input.pc segmented_output.pcs" << std::endl;
}


bool read_xyzn(const std::string& infn, PointCloud& pc) {
  std::cout << "Reading points from file (format: xyzn)" << std::endl;
  std::ifstream ifile(infn.c_str());
  if (!ifile) {
    std::cerr << "Cannot open file " << infn << std::endl;
    return false;
  }
  std::string line;
  std::vector< Point > points;
  while (std::getline(ifile, line)) {
    float x, y, z;
    float nx, ny, nz;
    if (std::istringstream(line) >> x >> y >> z >> nx >> ny >> nz) {
      points.push_back(Point(Vec3f(x,y,z), Vec3f(nx,ny,nz)));
    } else {
      std::cerr << "Error while reading xyzn file" << std::endl;
      ifile.close();
      return false;
    }
  }

  unsigned int num_points = points.size();
  Point* points_ptr = new Point[num_points];
  for (unsigned i = 0; i < num_points; ++i) {
    points_ptr[i] = points[i];
  }

  PointCloud temp_pc(points_ptr, num_points);
  pc += temp_pc;

  std::cout << num_points << " points read" << std::endl;

  delete[] points_ptr;
  ifile.close();
  return true;
}


bool read_pwn(const std::string& input_filename, PointCloud& pc)
{
  std::cout << "Reading points from file (format: pwn)" << std::endl;

  std::ifstream input(input_filename.c_str());
  if (!input) {
    std::cerr << "error while reading pwn file" << std::endl;
    return false;
  }

  unsigned int num_points;
  input >> num_points;
  
  // Read num_points points (point coordinates)
  std::vector< Vec3f > coordinates;
  for (unsigned int i = 0; i < num_points; ++i) {
    float x, y, z;
    input >> x >> y >> z;
    coordinates.push_back(Vec3f(x,y,z));
  }

  // Read num_points normals (vector coordinates)
  std::vector< Vec3f > normals;
  for (unsigned int i = 0; i < num_points; ++i) {
    float nx, ny, nz;
    input >> nx >> ny >> nz;
    normals.push_back(Vec3f(nx,ny,nz));
  }

  Point* points_ptr = new Point[num_points];
  for (unsigned int i = 0; i < num_points; ++i) {
    points_ptr[i] = Point(coordinates[i], normals[i]);
  }

  PointCloud temp_pc(points_ptr, num_points);
  pc += temp_pc;

  std::cout << num_points << " points read" << std::endl;

  delete[] points_ptr;
  return true;
}


struct Triangle {
  unsigned int v0;
  unsigned int v1;
  unsigned int v2;
  explicit Triangle(unsigned int u, unsigned int v, unsigned int w)
  : v0(u), v1(v), v2(w) {}
};


struct TriangleMesh {
  std::vector<Triangle> triangles;
  std::vector<Vec3f> vertices;
  TriangleMesh() {}
  explicit TriangleMesh(const std::vector<Triangle>& tri, 
                        const std::vector<Vec3f>& vert)
      : triangles(tri), vertices(vert) { }
};


inline void
compute_centroid(const Vec3f& v0, const Vec3f& v1, 
                 const Vec3f& v2, Vec3f& centroid) {
  centroid[0] = (v0[0] + v1[0] + v2[0]) * 1.0 / 3.0;
  centroid[1] = (v0[1] + v1[1] + v2[1]) * 1.0 / 3.0;
  centroid[2] = (v0[2] + v1[2] + v2[2]) * 1.0 / 3.0;
}


inline void
cross_product(const Vec3f& u, const Vec3f& v, Vec3f& result) {
  result[0] = u[1]*v[2] - u[2]*v[1];
  result[1] = u[2]*v[0] - u[0]*v[2];
  result[2] = u[0]*v[1] - u[1]*v[0];
}


inline void
compute_normal(
    const Vec3f& v0, const Vec3f& v1, 
    const Vec3f& v2, Vec3f& normal)
{
  const float epsilon = 1e-8f;
  
  Vec3f v0v1(v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]);
  Vec3f v0v2(v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]);
  cross_product(v0v1, v0v2, normal);
  float length = sqrt(
      normal[0]*normal[0] + normal[1]*normal[1] + 
      normal[2]*normal[2]);
  if (fabs(length) > epsilon)
    normal /= length;
}


void
compute_points_from_triangles(
    const std::vector<Vec3f>& vertices, 
    const std::vector<Triangle>& triangles,
    Point* points_ptr)
{
  unsigned int num_triangles = triangles.size();
  for (unsigned int i = 0; i < num_triangles; ++i) {
    Triangle tri = triangles[i];
    Vec3f v0 = vertices[tri.v0];
    Vec3f v1 = vertices[tri.v1];
    Vec3f v2 = vertices[tri.v2];

    compute_centroid(v0, v1, v2, points_ptr[i].pos);

    Vec3f normal;
    compute_normal(v0, v1, v2, points_ptr[i].normal);
                        
    points_ptr[i].meshFaceIndex = i;
  }
}


bool
read_ply2(const std::string& infn, PointCloud& pc, TriangleMesh& trimesh) {
  std::cout << "Reading points from a file (format: ply2)" << std::endl;
  std::ifstream ifile(infn.c_str());
  if (!ifile) {
    std::cerr << "Cannot open file " << infn << std::endl;
    return false;
  }

  std::vector<Vec3f> vertices;
  std::vector<Triangle> triangles;
  unsigned int num_vertices = 0;
  unsigned int num_triangles = 0;
  ifile >> num_vertices;
  ifile >> num_triangles;

  // read vertex coordinates
  for (unsigned int i = 0; i < num_vertices; ++i) {
    float x, y, z;
    ifile >> x >> y >> z;
    vertices.push_back(Vec3f(x,y,z));
  }

  // read vertex indices corresponding to each triangle
  for (unsigned int i = 0; i < num_triangles; ++i) {
    unsigned int dummy, v0, v1, v2;
    ifile >> dummy >> v0 >> v1 >> v2;
    triangles.push_back(Triangle(v0, v1, v2));
  }

  // setup the triangle mesh
  trimesh.vertices = vertices;
  trimesh.triangles = triangles;

  Point* points_ptr = new Point[num_triangles];
  compute_points_from_triangles(vertices, triangles, points_ptr);

  PointCloud temp_pc(points_ptr, num_triangles);
  pc += temp_pc;

  std::cout << num_triangles << " points read" << std::endl;

  delete[] points_ptr;
  ifile.close();
  return true;
}


bool
read_off(const std::string& infn, PointCloud& pc, TriangleMesh& trimesh) {
  std::cout << "Reading points from a file (format: off)" << std::endl;
  std::ifstream ifile(infn.c_str());
  if (!ifile) {
    std::cerr << "Cannot open file " << infn << std::endl;
    return false;
  }

  std::string s;
  ifile >> s;
  if (s != "OFF") {
    std::cerr << "Invalid OFF file" << std::endl;
    return false;
  }

  std::vector<Vec3f> vertices;
  std::vector<Triangle> triangles;
  unsigned int num_vertices = 0;
  unsigned int num_triangles = 0;
  unsigned int dummy;  
  ifile >> num_vertices >> num_triangles >> dummy;


  // read vertex coordinates
  for (unsigned int i = 0; i < num_vertices; ++i) {
    float x, y, z;
    ifile >> x >> y >> z;
    vertices.push_back(Vec3f(x,y,z));
  }

  // read vertex indices corresponding to each triangle
  for (unsigned int i = 0; i < num_triangles; ++i) {
    unsigned int dummy, v0, v1, v2;
    ifile >> dummy >> v0 >> v1 >> v2;
    triangles.push_back(Triangle(v0, v1, v2));
  }

  // setup the triangle mesh
  trimesh.vertices = vertices;
  trimesh.triangles = triangles;

  Point* points_ptr = new Point[num_triangles];
  compute_points_from_triangles(vertices, triangles, points_ptr);

  PointCloud temp_pc(points_ptr, num_triangles);
  pc += temp_pc;

  std::cout << num_triangles << " points read" << std::endl;

  delete[] points_ptr;
  ifile.close();
  return true;
}


bool
read_pc(const std::string& infn, 
        PointCloud& pc, bool& is_trimesh, TriangleMesh& trimesh)
{

  if (infn.rfind(".xyzn", infn.size()) != std::string::npos) {
    is_trimesh = false;
    bool success = read_xyzn(infn, pc);
    if (!success) {
      std::cerr << "Error reading: " << infn << std::endl;
      return false;
    }
    return true;
  }

  if (infn.rfind(".ply2", infn.size()) != std::string::npos) {
    is_trimesh = true;
    bool success = read_ply2(infn, pc, trimesh);
    if (!success) {
      std::cerr << "Error reading: " << infn << std::endl;
      return false;
    }
    return true;
  }

  if (infn.rfind(".off", infn.size()) != std::string::npos) {
    is_trimesh = true;
    bool success = read_off(infn, pc, trimesh);
    if (!success) {
      std::cerr << "Error reading: " << infn << std::endl;
      return false;
    }
    return true;
  }

  std::cerr << "Error reading: " << infn << std::endl;
  std::cerr << "File format not recognized" << std::endl;
  return false;
}


// Write the original point-set to a file using the xyzn file format
bool
write_xyzn(const std::string& outfn, const PointCloud& pc)
{
  std::cout << "Saving point-set" << std::endl;
  std::ofstream ofile(outfn.c_str());

  if (!ofile) {
    std::cerr << "Cannot open file in write mode" << std::endl;
    return false;
  }

  for (unsigned int i = 0; i < pc.size(); ++i) {
    Point p = pc[i];
    ofile << p.pos[0] << " "
          << p.pos[1] << " "
          << p.pos[2] << " "
          << p.normal[0] << " "
          << p.normal[1] << " "
          << p.normal[2]
          << std::endl;
  }

  std::cout << "Data written" << std::endl;
  ofile.close();
  return true;
}


// Write the segmented point-set into a file using the xyzn file format.
// Only the points from the original point-set that were identified in a
// cluster will be written.
bool
write_segmented_pc_xyzn(
    const std::string& outfn, const PointCloud&pc,
    const ShapeVector& shapes)
{
  std::cout << "Saving segmented point-set" << std::endl;

  std::ofstream ofile(outfn.c_str());
  if (!ofile) {
    std::cerr << "Cannot open file " << outfn
              << " in write mode" << std::endl;
    return false;
  }

  if (shapes.size() == 0) {
    std::cerr << "Vector of shapes is empty" << std::endl;
    return false;
  }

  size_t sum = 0;

  for (unsigned int i = 0; i < shapes.size(); ++i) {
    for (unsigned int j = pc.size() - (sum + shapes[i].second);
         j < pc.size() - sum; ++j) {

      Point p = pc[j];
      ofile << p.pos[0] << " " << p.pos[1] << " "
            << p.pos[2] << " "
            << p.normal[0] << " " << p.normal[1] << " "
            << p.normal[2] << std::endl;
    }
    sum += shapes[i].second;
  }

  std::cout << "Data written" << std::endl;
  ofile.close();
  return true;
  
  return true;
}


void compute_bbox(const PointCloud& pc, Vec3f& min_pt, Vec3f& max_pt)
{
  float fmax = std::numeric_limits<float>::max();
  min_pt[0] = fmax; 
  min_pt[1] = fmax;
  min_pt[2] = fmax;

  max_pt[0] = -fmax;
  max_pt[1] = -fmax;
  max_pt[2] = -fmax;

  for (unsigned int i = 0; i < pc.size(); ++i) {
    Point p = pc[i];
    min_pt[0] = std::min(min_pt[0], p[0]);
    min_pt[1] = std::min(min_pt[1], p[1]);
    min_pt[2] = std::min(min_pt[2], p[2]);
    max_pt[0] = std::max(max_pt[0], p[0]);
    max_pt[1] = std::max(max_pt[1], p[1]);
    max_pt[2] = std::max(max_pt[2], p[2]);
  }
}


typedef std::vector< std::pair< MiscLib::RefCountPtr< PrimitiveShape >, size_t > > ShapeVector;


// Save the segmented point-set (coordinates only) in .pctl format.
// pc: point-cloud
// tl: type/label
//
// File format:
// num_points
// x y z type label
// ...
// x_n y_n z_n type label
bool
write_segmented_pc(
    const std::string& outfn, const PointCloud& pc, 
    const ShapeVector& shapes)
{
  std::cout << "Saving segmented point-set" << std::endl;
  
  std::ofstream ofile(outfn.c_str());
  if (!ofile) {
    std::cerr << "Cannot open file " << outfn 
              << " in write mode" << std::endl;
    return false;
  }

  if (shapes.size() == 0) {
    std::cerr << "Vector of shapes is empty" << std::endl;
    return false; 
  }

  size_t num_points = 0;
  for (unsigned int i = 0; i < shapes.size(); ++i) 
    num_points += shapes[i].second;
  ofile << num_points << std::endl;

  size_t sum = 0;

  for (unsigned int i = 0; i < shapes.size(); ++i) {
    for (unsigned int j = pc.size() - (sum + shapes[i].second);
         j < pc.size() - sum; ++j) {

      Point p = pc[j];
      ofile << p[0] << " " << p[1] << " "
            << p[2] << " " << shapes[i].first->Identifier()
            << " " << i << std::endl;

    }
    sum += shapes[i].second;
  }

  std::cout << "Data written" << std::endl;
  ofile.close();
  return true;
}


bool
write_segmented_xyzntl(
    const std::string& out_filename, const PointCloud& pc,
    const ShapeVector& shapes)
{
  std::cout << "Saving segmented point-set" << std::endl;
  
  std::ofstream ofile(out_filename.c_str());
  if (!ofile) {
    std::cerr << "Cannot open file " << out_filename 
              << " in write mode" << std::endl;
    return false;
  }

  if (shapes.size() == 0) {
    std::cerr << "Vector of shapes is empty" << std::endl;
    return false; 
  }

  size_t sum = 0;

  for (unsigned int i = 0; i < shapes.size(); ++i) {
    for (unsigned int j = pc.size() - (sum + shapes[i].second);
         j < pc.size() - sum; ++j) {

      Point p = pc[j];
      ofile << p.pos[0] << " " << p.pos[1] << " "
            << p.pos[2] << " "
            << p.normal[0] << " " << p.normal[1] << " "
            << p.normal[2] << " "
            << shapes[i].first->Identifier()
            << " " << i << std::endl;

    }
    sum += shapes[i].second;
  }

  std::cout << "Data written" << std::endl;
  ofile.close();
  return true;
}


// save segmented mesh using the PLYTL file format.
// PLYTL: PLY stands for .ply and TL stands for Type, Label.
// Type is the primitive type (an int)
// Label is the segment label (an int)
bool
write_segmented_mesh(
    const std::string& outfn, const PointCloud& pc,
    const ShapeVector& shapes, const TriangleMesh& trimesh)
{
  std::cout << "Saving segmented triangle mesh" << std::endl;
  std::ofstream ofile(outfn.c_str());
  if (!ofile) {
    std::cerr << "Cannot open file in write mode" << std::endl;
    return false;
  }

  if (shapes.size() == 0) {
    std::cerr << "Vector of shapes is empty" << std::endl;
    return false;
  }

  ofile << trimesh.vertices.size() << std::endl;
  ofile << trimesh.triangles.size() << std::endl;

  // save the vertex coordinates
  for (unsigned int i = 0; i < trimesh.vertices.size(); ++i) {
    Vec3f v = trimesh.vertices[i];
    ofile << v[0] << " " << v[1] << " " << v[2] << std::endl;
  }

  // save the triangles; each triangle correspond to a point in pc
  size_t sum = 0;
  for (unsigned int i = 0; i < shapes.size(); ++i) {
    for (unsigned int j = pc.size() - (sum + shapes[i].second);
         j < pc.size() - sum; ++j) {

      Point p = pc[j];
      size_t idx = p.meshFaceIndex;
      Triangle tri = trimesh.triangles[idx];
      ofile << "3 " << tri.v0 << " " << tri.v1 << " " << tri.v2 << " "
            << shapes[i].first->Identifier() << " "
            << i << std::endl;

    }
    sum += shapes[i].second;
  }

  // save also the remaining triangles ?? 
  // (i.e. triangles with no assigned primitives)
  // currently these are not saved, so the final triangle mesh has holes

  std::cout << "Data written" << std::endl;
  ofile.close();  
  return true;
}


// Compute additional halfplanes corresponding to the planes of the bounding
// boxes of the non-planar shapes:
void
compute_additional_halfplanes(
    const ShapeVector& shapes, const PointCloud& pc, 
    std::vector< MiscLib::RefCountPtr< PlanePrimitiveShape> >& halfplanes)
{
  size_t sum = 0;
  unsigned int num_primitives = shapes.size();
  
  for (unsigned int i = 0; i < num_primitives; ++i)
  {
    size_t id = shapes[i].first->Identifier();
    
    if (id == 0) {
      // a plane; we don't have to do anything
    } else {
      // Extract the corresponding point-cloud
      std::vector< Point > points;
      for (unsigned int j = pc.size() - (sum + shapes[i].second);
           j < pc.size() - sum; ++j)
      {
        Point p = pc[j];
        points.push_back(p);
      }

      // NOTE: I can probably use std::copy instead
      unsigned int num_points = points.size();      
      Point* points_ptr = new Point[num_points];
      for (unsigned int k = 0; k < num_points; ++k) {
        points_ptr[k] = points[k];
      }
      PointCloud temp_pc(points_ptr, num_points);
      delete[] points_ptr;

      
      // Compute its bounding box
      Vec3f min_pt;
      Vec3f max_pt;
      compute_bbox(temp_pc, min_pt, max_pt);

      
      // Find the 6 corresponding planes

      // construct 6 planes using the plane constructor:
      // Plane(Vec3f p1, Vec3f p2, Vec3f p3);
      // Then pass a plane object to PlanePrimitiveShape constructor;
      // Then add the shape to the list;
      
      // Plane 1:
      Vec3f p1(min_pt[0], min_pt[1], min_pt[2]);
      Vec3f p2(max_pt[0], min_pt[1], min_pt[2]);
      Vec3f p3(max_pt[0], min_pt[1], max_pt[2]);
      PlanePrimitiveShape* pps1 = new PlanePrimitiveShape(p1, p2, p3);
      halfplanes.push_back(pps1);

      // Plane 2:
      p1 = Vec3f(max_pt[0], max_pt[1], max_pt[2]); // Can I do that??
      p2 = Vec3f(max_pt[0], min_pt[1], max_pt[2]);
      p3 = Vec3f(max_pt[0], min_pt[1], min_pt[2]);
      PlanePrimitiveShape* pps2 = new PlanePrimitiveShape(p1, p2, p3);
      halfplanes.push_back(pps2);
      
      // Plane 3:
      p1 = Vec3f(max_pt[0], max_pt[1], max_pt[2]);
      p2 = Vec3f(max_pt[0], max_pt[1], min_pt[2]);
      p3 = Vec3f(min_pt[0], max_pt[1], min_pt[2]);
      PlanePrimitiveShape* pps3 = new PlanePrimitiveShape(p1, p2, p3);
      halfplanes.push_back(pps3);

      // Plane 4:
      p1 = Vec3f(min_pt[0], max_pt[1], max_pt[2]);
      p2 = Vec3f(min_pt[0], max_pt[1], min_pt[2]);
      p3 = Vec3f(min_pt[0], min_pt[1], min_pt[2]);
      PlanePrimitiveShape* pps4 = new PlanePrimitiveShape(p1, p2, p3);
      halfplanes.push_back(pps4);
      
      // Plane 5:
      p1 = Vec3f(max_pt[0], max_pt[1], max_pt[2]);
      p2 = Vec3f(min_pt[0], max_pt[1], max_pt[2]);
      p3 = Vec3f(min_pt[0], min_pt[1], max_pt[2]);
      PlanePrimitiveShape* pps5 = new PlanePrimitiveShape(p1, p2, p3);
      halfplanes.push_back(pps5);

      // Plane 6:
      p1 = Vec3f(min_pt[0], min_pt[1], min_pt[2]);
      p2 = Vec3f(min_pt[0], max_pt[1], min_pt[2]);
      p3 = Vec3f(max_pt[0], max_pt[1], min_pt[2]);
      PlanePrimitiveShape* pps6 = new PlanePrimitiveShape(p1, p2, p3);
      halfplanes.push_back(pps6);
      
    } // if plane

    sum += shapes[i].second;
    
  } // for each primitive
}


// Is the plane defined by the three points p1, p2 and p3 degenerate?
bool plane_is_degenerate(const Vec3f& p1, const Vec3f& p2, const Vec3f& p3)
{
  Vec3f p1p2 = p2 - p1;
  Vec3f p1p3 = p3 - p1;

  Vec3f normal = p1p2.cross(p1p3);
  float sqr_len = normal.sqrLength();

  return sqr_len < 1e-7;
}


// Decide if two planes are equivalent.
// The method Plane::equals() is insufficient because it returns false
// if two planes have different orientation (even if they are the same).
//
// Note: I can't pass the planes as const because of Plane::equals()
bool
planes_are_equivalent(Plane& p1, Plane& p2)
{
  // return true if they are equal obviously
  if (p1.equals(p2))
    return true;

  // otherwise check if they have opposite normal direction
  Vec3f position_p2 = p2.getPosition();
  Vec3f normal_p2 = p2.getNormal();
  Vec3f normal_p1 = p1.getNormal();

  // 0.2 comes from Plane::equals(); I am keeping the same value for
  // consistency
  if (p1.getDistance(position_p2) > 0.2)
    return false;

  float dot = normal_p1.dot(normal_p2);
  float abs_dot = fabsf(dot);

  // 0.9 comes from Plane::equals(); I am keeping the same value for
  // consistency
  if (abs_dot > 0.9)
    return true;
  else
    return false;
}


// Does the plane corresponding to the plane primitive shape pps exist?
bool
plane_exists(
    const PlanePrimitiveShape* pps,
    const std::vector< MiscLib::RefCountPtr< PlanePrimitiveShape > >& hp)
{
  Plane plane_argument = pps->Internal();
  
  unsigned int num_halfplanes = hp.size();
  for (unsigned int i = 0; i < num_halfplanes; ++i) {
    Plane curr_plane = hp[i]->Internal();
    //if (plane_argument.equals(curr_plane)) {
    if (planes_are_equivalent(plane_argument, curr_plane)) {
      return true;
    }
  }

  return false;
}


// TODO:
// 1) Find and remove degenerate planes; a plane is degenerate if its normal
// vector is null
// 2) Find and remove duplicates in the list of additional half-planes
//
// These can be done in compute_additional_halfplanes()
// to check if two planes are equal, use: Plane::equals()
// to get the underlying Plane, use PlanePrimitiveShape::Internal()
//
// Compute additional halfplanes corresponding to the planes of the bounding
// boxes of the non-planar shapes:
void
compute_additional_halfplanes_FIXED(
    const ShapeVector& shapes, const PointCloud& pc, 
    std::vector< MiscLib::RefCountPtr< PlanePrimitiveShape> >& halfplanes)
{
  size_t sum = 0;
  unsigned int num_primitives = shapes.size();
  
  for (unsigned int i = 0; i < num_primitives; ++i)
  {
    size_t id = shapes[i].first->Identifier();
    
    if (id == 0) {
      // a plane; we don't have to do anything
    } else {
      // Extract the corresponding point-cloud
      std::vector< Point > points;
      for (unsigned int j = pc.size() - (sum + shapes[i].second);
           j < pc.size() - sum; ++j)
      {
        Point p = pc[j];
        points.push_back(p);
      }

      // NOTE: I can probably use std::copy instead
      unsigned int num_points = points.size();      
      Point* points_ptr = new Point[num_points];
      for (unsigned int k = 0; k < num_points; ++k) {
        points_ptr[k] = points[k];
      }
      PointCloud temp_pc(points_ptr, num_points);
      delete[] points_ptr;

      
      // Compute its bounding box
      Vec3f min_pt;
      Vec3f max_pt;
      compute_bbox(temp_pc, min_pt, max_pt);

      
      // Find the 6 corresponding planes

      // construct 6 planes using the plane constructor:
      // Plane(Vec3f p1, Vec3f p2, Vec3f p3);
      // Then pass a plane object to PlanePrimitiveShape constructor;
      // Then add the shape to the list;
      
      // Plane 1:
      Vec3f p1(min_pt[0], min_pt[1], min_pt[2]);
      Vec3f p2(max_pt[0], min_pt[1], min_pt[2]);
      Vec3f p3(max_pt[0], min_pt[1], max_pt[2]);
      PlanePrimitiveShape* pps1 = new PlanePrimitiveShape(p1, p2, p3);
      if (!plane_is_degenerate(p1, p2, p3) &&
          !plane_exists(pps1, halfplanes))
      {
          halfplanes.push_back(pps1);
      }
      
      // Plane 2:
      p1 = Vec3f(max_pt[0], max_pt[1], max_pt[2]);
      p2 = Vec3f(max_pt[0], min_pt[1], max_pt[2]);
      p3 = Vec3f(max_pt[0], min_pt[1], min_pt[2]);
      PlanePrimitiveShape* pps2 = new PlanePrimitiveShape(p1, p2, p3);
      if (!plane_is_degenerate(p1, p2, p3) &&
          !plane_exists(pps2, halfplanes))
      {
          halfplanes.push_back(pps2);
      }

      // Plane 3:
      p1 = Vec3f(max_pt[0], max_pt[1], max_pt[2]);
      p2 = Vec3f(max_pt[0], max_pt[1], min_pt[2]);
      p3 = Vec3f(min_pt[0], max_pt[1], min_pt[2]);
      PlanePrimitiveShape* pps3 = new PlanePrimitiveShape(p1, p2, p3);
      if (!plane_is_degenerate(p1, p2, p3) &&
          !plane_exists(pps3, halfplanes))
      {
          halfplanes.push_back(pps3);
      }

      // Plane 4:
      p1 = Vec3f(min_pt[0], max_pt[1], max_pt[2]);
      p2 = Vec3f(min_pt[0], max_pt[1], min_pt[2]);
      p3 = Vec3f(min_pt[0], min_pt[1], min_pt[2]);
      PlanePrimitiveShape* pps4 = new PlanePrimitiveShape(p1, p2, p3);
      if (!plane_is_degenerate(p1, p2, p3) &&
          !plane_exists(pps4, halfplanes))
      {
          halfplanes.push_back(pps4);
      }
      
      // Plane 5:
      p1 = Vec3f(max_pt[0], max_pt[1], max_pt[2]);
      p2 = Vec3f(min_pt[0], max_pt[1], max_pt[2]);
      p3 = Vec3f(min_pt[0], min_pt[1], max_pt[2]);
      PlanePrimitiveShape* pps5 = new PlanePrimitiveShape(p1, p2, p3);
      if (!plane_is_degenerate(p1, p2, p3) &&
          !plane_exists(pps5, halfplanes))
      {
          halfplanes.push_back(pps5);
      }

      // Plane 6:
      p1 = Vec3f(min_pt[0], min_pt[1], min_pt[2]);
      p2 = Vec3f(min_pt[0], max_pt[1], min_pt[2]);
      p3 = Vec3f(max_pt[0], max_pt[1], min_pt[2]);
      PlanePrimitiveShape* pps6 = new PlanePrimitiveShape(p1, p2, p3);
      if (!plane_is_degenerate(p1, p2, p3) &&
          !plane_exists(pps6, halfplanes))
      {
          halfplanes.push_back(pps6);
      }
      
    } // if plane

    sum += shapes[i].second;
    
  } // for each primitive
}


// Write the list of identified primitives as:
// primitive_name_1 parameters_1
// ...
// primitive_name_n parameters_n
bool
write_primitives_list(
    const std::string& out_filename,
    const ShapeVector& shapes, const PointCloud& pc)
{
  std::cout << "Saving list of fitted primitives" << std::endl;

  std::ofstream output(out_filename.c_str());
  if (!output) {
    std::cerr << "Can not open file " << out_filename
              << " in write mode." << std::endl;
    return false;
  }

  if (shapes.size() == 0) {
    std::cerr << "No primitive shapes found." << std::endl;
    return false;
  }

  output.precision(16); // number of significant digits

  unsigned int num_primitives = shapes.size();
  std::cout << "Number of primitives: " << num_primitives << std::endl;
  for (unsigned int i = 0; i < num_primitives; ++i) {
    // current primitive name
    std::string name;
    shapes[i].first->Description(&name);
    output << name << " ";

    // current primitive parameters
    shapes[i].first->Serialize(&output);

    // Serialize() is already adding a line return
    //output << std::endl;
  }

  
  // compute additional halfplanes corresponding to the planes of the
  // bounding boxes of the non-planar shapes
  std::vector< MiscLib::RefCountPtr< PlanePrimitiveShape> > extra_halfplanes;
  compute_additional_halfplanes_FIXED(shapes, pc, extra_halfplanes);

  // save these additional halfplanes to file output;
  unsigned int num_extra_halfplanes = extra_halfplanes.size();
  std::cout << "Number of extra halfplanes: " << num_extra_halfplanes
            << std::endl;
  for (unsigned int i = 0; i < num_extra_halfplanes; ++i) {
    // current extra halfplane's name
    std::string name;
    extra_halfplanes[i]->Description(&name);
    output << name << " ";

    // current extra halfplane's parameters
    extra_halfplanes[i]->Serialize(&output, false); // by default it serializes in binary
  }
  
  return true;
}


bool
write_globfit(
    const std::string& outfn, const PointCloud& pc,
    const ShapeVector& shapes)
{
  std::cout << "Saving segmented point-set" << std::endl;

  std::ofstream ofile(outfn.c_str());
  if (!ofile) {
    std::cerr << "Cannot open file " << outfn
              << " in write mode" << std::endl;
    return false;
  }

  if (shapes.size() == 0) {
    std::cerr << "Vector of shapes is empty" << std::endl;
    return false; 
  }

  // Use 16 significant digits
  ofile.precision(16);

  size_t num_points = 0;
  for (unsigned int i = 0; i < shapes.size(); ++i) 
    num_points += shapes[i].second;

  ofile << "# Number of Points" << std::endl;
  ofile << num_points << std::endl;
  ofile << "# Points" << std::endl;
  ofile << "# px py pz nx ny nz confidence" << std::endl;


  size_t sum = 0;

  for (unsigned int i = 0; i < shapes.size(); ++i) {
    for (unsigned int j = pc.size() - (sum + shapes[i].second);
         j < pc.size() - sum; ++j) {

      Point p = pc[j];
      ofile << p[0] << " " << p[1] << " "
            << p[2] << " " << p.normal[0] << " "
            << p.normal[1] << " " << p.normal[2] << " 1.0" << std::endl;
      
    }
    sum += shapes[i].second;
  }

  ofile << "# End of points" << std::endl;

  ofile << std::endl;
  ofile << "# Number of Primitives" << std::endl;
  unsigned int num_primitives = shapes.size();
  ofile << num_primitives << std::endl;
  ofile << "# Primitives" << std::endl;
  sum = 0;
  for (unsigned int i = 0; i < num_primitives; ++i) {
    ofile << "# Primitive " << i << std::endl;
    // name
    std::string name;
    shapes[i].first->Description(&name);
    ofile << name << " ";
    // parameters
    shapes[i].first->Serialize(&ofile);
    //ofile << std::endl;

    // indices of points belonging to this primitive
    ofile << "# points idx_1 idx_2 ..." << std::endl;
    ofile<< "points";

    for (unsigned int j = 0; j < shapes[i].second; ++j) {
      ofile << " " << sum + j;
    }
    ofile << std::endl;
    sum += shapes[i].second;
  }
  ofile << "# End of Primitives" << std::endl;


  std::cout << "Data written" << std::endl;
  ofile.close();
  return true;
}


struct Options {
  float epsilon;
  float bitmap_epsilon;
  float normal_threshold;
  int min_support;
  float probability;
  bool use_plane;
  bool use_sphere;
  bool use_cylinder;
  bool use_cone;
  bool use_torus;

  Options() {
    epsilon = 0.01f;
    bitmap_epsilon = 0.02f;
    normal_threshold = 0.9f;
    min_support = 500;
    probability = 0.001f;
    use_plane = true;
    use_sphere = true;
    use_cylinder = true;
    use_cone = true;
    use_torus = true;
  }

  void printOptions() {
    std::cout << bitmap_epsilon << std::endl;
    std::cout << epsilon << std::endl;
    std::cout << min_support << std::endl;
    std::cout << normal_threshold << std::endl;
    std::cout << probability << std::endl;
    std::cout << use_cone << std::endl;
    std::cout << use_cylinder << std::endl;
    std::cout << use_plane << std::endl;
    std::cout << use_sphere << std::endl;
    std::cout << use_torus << std::endl;
  }
};


void
compute_segmentation(
    const std::string& infn, const std::string& outfn, 
    const Options& options)
{
  
  PointCloud pc;
  TriangleMesh trimesh;
  bool is_trimesh = false;

  bool success = read_pc(infn, pc, is_trimesh, trimesh);
  if (!success) {
    exit(1);
  }

  
  Vec3f min_pt, max_pt;
  compute_bbox(pc, min_pt, max_pt);
  pc.setBBox(min_pt, max_pt);

  // do the segmentation
  RansacShapeDetector::Options ransacOptions;

  ransacOptions.m_epsilon = options.epsilon * pc.getScale();
  ransacOptions.m_bitmapEpsilon = options.bitmap_epsilon * pc.getScale();
  ransacOptions.m_normalThresh = options.normal_threshold;
  ransacOptions.m_minSupport = options.min_support;
  ransacOptions.m_probability = options.probability;

  RansacShapeDetector detector(ransacOptions);

  if (options.use_plane) {
    PrimitiveShapeConstructor* plane =
        new PlanePrimitiveShapeConstructor();
    detector.Add(plane);
  }

  if (options.use_sphere) {
    PrimitiveShapeConstructor* sphere =
        new SpherePrimitiveShapeConstructor();
    detector.Add(sphere);
  }

  if (options.use_cylinder) {
    PrimitiveShapeConstructor* cylinder =
        new CylinderPrimitiveShapeConstructor();
    detector.Add(cylinder);
  }

  if (options.use_cone) {
    PrimitiveShapeConstructor* cone =
        new ConePrimitiveShapeConstructor();
    detector.Add(cone);
  }

  if (options.use_torus) {
    PrimitiveShapeConstructor* torus =
        new TorusPrimitiveShapeConstructor();
    detector.Add(torus);
  }

  
  // store the detected shapes
  std::vector< std::pair< MiscLib::RefCountPtr< PrimitiveShape >, size_t > > shapes;
  // run detection and returns the number of unassigned points
  size_t remaining = detector.Detect(pc, 0, pc.size(), &shapes);
  std::cerr << "detection finished " << remaining << std::endl;


  if (is_trimesh) {
    success = write_segmented_mesh(outfn, pc, shapes, trimesh);
  } else {
    success = write_segmented_pc(outfn, pc, shapes);
  }

  if (!success) exit(1);
  
  
  // Write the list of primitives (names and parameters)
  size_t dotpos = outfn.find_last_of('.');
  std::string base = outfn.substr(0, dotpos);
  std::string primitives_list_filename = base + ".fit";
  success = write_primitives_list(
      primitives_list_filename, shapes, pc);

  if (!success) exit(1);

  
  // Write the segmented point-set using the globfit file format
  //std::string out_globfit = base + ".globfit";
  //success = write_globfit(out_globfit, pc, shapes);


  // Write the points from the original point-set that were identified on
  // one shape
  std::string segmented_xyzn_filename = base + "-segmented" + ".xyzn";
  success = write_segmented_pc_xyzn(
      segmented_xyzn_filename, pc, shapes);

  if (!success) exit(1);
}


bool
readConf(const std::string& fname, Options& options)
{
  std::ifstream in(fname.c_str());
  if (!in)
  {
    std::cerr << "Could not open " << fname << std::endl;
    return false;
  }

  std::string dummy, type, name;

  in >> dummy >> type >> name;
  while (dummy.find("#") != std::string::npos)
  {
    if (name == "epsilon") {
      in >> options.epsilon;
    } else if (name == "bitmap_epsilon")
    {
      in >> options.bitmap_epsilon;
    } else if (name == "min_support")
    {
      in >> options.min_support;
    } else if (name == "normal_threshold")
    {
      in >> options.normal_threshold;
    } else if (name == "probability")
    {
      in >> options.probability;
    } else if (name == "use_cone")
    {
      in >> options.use_cone;
    } else if (name == "use_cylinder")
    {
      in >> options.use_cylinder;
    } else if (name == "use_plane")
    {
      in >> options.use_plane;
    } else if (name == "use_sphere")
    {
      in >> options.use_sphere;
    } else if (name == "use_torus")
    {
      in >> options.use_torus;
    } else 
    {
      std::cout << "Error in reading conf parameters." << std::endl;
      return false;
    }

    dummy = "end";

    in >> dummy >> type >> name;
  }

  in.close();

  return true;
}


int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    usage(argv[0]);
    return (-1);
  }

  std::string in_filename = argv[1];
  std::string out_filename = argv[2];

  Options options;
  if (argc == 4) {
    std::string conf = argv[3];
    bool success = readConf(conf, options);
    if (!success) {
      exit(-1);
    }
  }
  options.printOptions();

  compute_segmentation(in_filename, out_filename, options);

  return 0;
}

*/
