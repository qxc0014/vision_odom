cmake_minimum_required( VERSION 2.8 )
project( keypoint )
set(CMAKE_BUILD_TYPE "Release")
SET( CMAKE_CXX_STANDARD 14)
SET( CMAKE_CXX_STANDARD_REQUIRED ON)
#add_compile_options()
#add_definitions("-DENABLE_SSE")
#set(CMAKE_CXX_FLAGS "-std=c++14 -O2 ${SSE_FLAGS} -g -march=native")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
SET( G2O_ROOT /usr/local/include/g2o )
include_directories("/usr/local/include/eigen3")
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
find_package(PCL 1.9 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_libraries(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
find_package(g2o REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})
find_package(CSparse REQUIRED)
include_directories(${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})
#add_library(slamBase slamBase.cpp)
add_executable( keypoint keypoint.cpp slamBase.cpp calTwist.cpp optimize.cpp showimage.cpp)
target_link_libraries( keypoint  
                        g2o_core 
                        g2o_stuff
                        g2o_solver_csparse 
                        g2o_csparse_extension 
                        g2o_types_slam3d
                        ${G2O_LIBRARIES}
                        ${CSPARSE_LIBRARIES}
                        ${OpenCV_LIBRARIES} 
                        ${Sophus_LIBRARIES} 
                        ${PCL_LIBRARIES}
                        ${Pangolin_LIBRARIES}    
                        ${CERES_LIBRARIES}
                        )
