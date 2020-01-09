cmake_minimum_required(VERSION 3.1)
project(csg_playground)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" OFF)
option(LIBIGL_WITH_ANTTWEAKBAR      "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL             "Use CGAL"           ON)
option(LIBIGL_WITH_COMISO           "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK             "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE           "Use Embree"         OFF)
option(LIBIGL_WITH_LIM              "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB           "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK            "Use MOSEK"          OFF)
option(LIBIGL_WITH_NANOGUI          "Use Nanogui menu"   OFF) 
option(LIBIGL_WITH_PNG              "Use PNG"            OFF)
option(LIBIGL_WITH_PYTHON           "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN           "Use Tetgen"         OFF)
option(LIBIGL_WITH_TRIANGLE         "Use Triangle"       OFF)
option(LIBIGL_WITH_XML              "Use XML"            OFF)
option(LIBIGL_WITH_VIEWER           "Use OpenGL viewer"  OFF)
option(LIBIGL_WITH_OPENGL           "Use OpenGL"         OFF)
option(LIBIGL_WITH_OPENGL_GLFW      "Use GLFW"           OFF)

find_package(LIBIGL REQUIRED QUIET)

#CDD Lib
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/cddlib)
include_directories(${PROJECT_SOURCE_DIR}/third_party/cddlib/lib-src) 

#RANSAC Lib
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/ransac)
include_directories(${PROJECT_SOURCE_DIR}/third_party/ransac/src) 

#JSON
include_directories(${PROJECT_SOURCE_DIR}/third_party/json/src) 
include_directories(${PROJECT_SOURCE_DIR}/include/) 

#INIH
include_directories(${PROJECT_SOURCE_DIR}/third_party/inih) 

#CONCAVEMAN
include_directories(${PROJECT_SOURCE_DIR}/third_party/concaveman/src/main/cpp) 

#PCL
find_package(PCL 1.8 COMPONENTS common io features search kdtree keypoints surface ml)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#Python
find_package(PythonLibs 3 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
link_directories(${PYTHON_LIBRARIES})
message("Python includes: ", ${PYTHON_INCLUDE_DIRS})
message("Python libs: ", ${PYTHON_LIBRARIES})

# Add your project files
#file(GLOB SRCFILES *.cpp)

FILE(GLOB_RECURSE CSG_LIB_HEADERS "include/*.h")
message("Lib Headers: " ${CSG_LIB_HEADERS})

FILE(GLOB CSG_LIB_SOURCES "src/collision.cpp" "src/congraph.cpp" "src/csgnode.cpp" "src/csgnode_evo.cpp" "src/csgnode_evo_v2.cpp" "src/csgnode_helper.cpp" "src/curvature.cpp" "src/dnf.cpp" "src/evolution.cpp" "src/mesh.cpp" "src/pointcloud.cpp" "src/ransac.cpp" "src/statistics.cpp" "src/test.cpp" "src/helper.cpp" "src/params.cpp" "src/primitives.cpp" "src/primitive_extraction.cpp" "src/primitive_helper.cpp" "src/cluster.cpp" "src/optimizer_red.cpp" "src/optimizer_clustering.cpp" "src/optimizer_ga.cpp" "src/cit.cpp" "src/optimizer_py.cpp" "src/red_inserter.cpp")
message("Lib Sources: " ${CSG_LIB_SOURCES})

if(MSVC)
	add_definitions(-D_USE_MATH_DEFINES)
endif()


# Otherwise g++ was failing on cygwin 
if(CYGWIN)
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
endif()


# Compile the lib
add_library(csg_playground_lib STATIC ${CSG_LIB_HEADERS} ${CSG_LIB_SOURCES})
target_link_libraries(csg_playground_lib igl::core igl::cgal igl::opengl_glfw ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_FEATURES_LIBRARIES} ${PCL_SEARCH_LIBRARIES} ${PCL_KDTREE_LIBRARIES}  ${PCL_SURFACE_LIBRARIES}  ${PCL_ML_LIBRARIES} ${PYTHON_LIBRARIES} cddlib orig_ransac)

add_library(csg_c SHARED "include/c.h" "src/c.cpp")
target_link_libraries(csg_c csg_playground_lib igl::core igl::cgal)

# Program for sampling models
add_executable(main_sampling "src/main_sampling.cpp")
target_link_libraries(main_sampling csg_playground_lib igl::core igl::cgal)

# Program for generating primitives 
add_executable(primgen "src/main_primgen.cpp")
target_link_libraries(primgen csg_playground_lib  igl::core igl::cgal igl::core igl::opengl_glfw ${PCL_COMMON_LIBRARIES} ${PCL_SEARCH_LIBRARIES} ${PCL_KDTREE_LIBRARIES} ${PCL_SURFACE_LIBRARIES})

# Program for csg reconstruction
add_executable(main_csg "src/main_csg.cpp")
target_link_libraries(main_csg csg_playground_lib igl::core igl::cgal)

# Program for csg reconstruction with gui
if(LIBIGL_WITH_VIEWER AND LIBIGL_WITH_OPENGL AND LIBIGL_WITH_OPENGL_GLFW)
	add_executable(main "src/main.cpp")
	target_link_libraries(main csg_playground_lib  igl::core igl::cgal igl::core igl::opengl_glfw ${PCL_COMMON_LIBRARIES} ${PCL_SEARCH_LIBRARIES} ${PCL_KDTREE_LIBRARIES} ${PCL_SURFACE_LIBRARIES})
endif()
