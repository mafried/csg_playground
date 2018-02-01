cmake_minimum_required(VERSION 3.1)
project(csg_playground)

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
option(LIBIGL_WITH_OPENGL           "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW      "Use GLFW"           ON)
option(LIBIGL_WITH_PNG              "Use PNG"            OFF)
option(LIBIGL_WITH_PYTHON           "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN           "Use Tetgen"         OFF)
option(LIBIGL_WITH_TRIANGLE         "Use Triangle"       OFF)
option(LIBIGL_WITH_VIEWER           "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_XML              "Use XML"            OFF)

find_package(LIBIGL REQUIRED QUIET)

#JSON
include_directories(${PROJECT_SOURCE_DIR}/third_party/json/src) 
include_directories(${PROJECT_SOURCE_DIR}/include/) 

#PCL
find_package(PCL 1.8 REQUIRED COMPONENTS common io sample_consensus)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add your project files
file(GLOB SRCFILES *.cpp)

FILE(GLOB_RECURSE APP_HEADERS "include/*.h")
message("APP Headers: " ${APP_HEADERS})

FILE(GLOB_RECURSE APP_SOURCES "src/*.cpp")
message("APP Sources: " ${APP_SOURCES})


add_executable(${PROJECT_NAME}_bin ${APP_HEADERS} ${APP_SOURCES})

target_link_libraries(${PROJECT_NAME}_bin ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_SAMPLE_CONSENSUS_LIBRARIES} igl::core igl::viewer igl::cgal)