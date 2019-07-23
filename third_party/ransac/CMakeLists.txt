cmake_minimum_required(VERSION 3.1)

project(orig_ransac)

#set(CMAKE_BUILD_TYPE Release)

if(UNIX)
  add_definitions("-std=c++11 -Wall")
elseif(MSVC)
  add_definitions("-D_CRT_SECURE_NO_WARNINGS")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()

set(SRC_DIR "src/")
include_directories(${SRC_DIR})
file(GLOB_RECURSE CPP_HEADERS ${SRC_DIR}*.h)
file(GLOB_RECURSE CPP_FILES ${SRC_DIR}*.cpp)

#set(Boost_USE_STATIC_LIBS OFF) 
#set(Boost_USE_MULTITHREADED ON)  
#set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.45.0 REQUIRED) 

include_directories(${Boost_INCLUDE_DIRS}) 
add_library(orig_ransac ${CPP_FILES} ${CPP_HEADERS})
#target_link_libraries(orig_ransac_command ${Boost_LIBRARIES})
