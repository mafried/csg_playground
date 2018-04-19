cmake_minimum_required(VERSION 3.1)
project(csg_playground)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# libigl
option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" OFF)
option(LIBIGL_WITH_ANTTWEAKBAR      "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL             "Use CGAL"           OFF)
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
option(LIBIGL_WITH_VIEWER           "Use OpenGL viewer"  OFF)
option(LIBIGL_WITH_XML              "Use XML"            OFF)

find_package(LIBIGL REQUIRED QUIET)

#Boost
find_package(Boost COMPONENTS graph REQUIRED)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

#JSON
include_directories(${PROJECT_SOURCE_DIR}/third_party/json/src) 
include_directories(${PROJECT_SOURCE_DIR}/include/) 

# Add your project files
file(GLOB SRCFILES *.cpp)

FILE(GLOB_RECURSE APP_HEADERS "include/*.h")
message("APP Headers: " ${APP_HEADERS})

FILE(GLOB_RECURSE APP_SOURCES "src/*.cpp")
message("APP Sources: " ${APP_SOURCES})

if(MSVC)
	add_definitions(-D_USE_MATH_DEFINES)
endif()

add_executable(${PROJECT_NAME}_bin ${APP_HEADERS} ${APP_SOURCES})

target_link_libraries(${PROJECT_NAME}_bin igl::core)