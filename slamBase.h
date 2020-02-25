# pragma once
#include<iostream>
#include<fstream>
#include<cmath>
#include<Eigen/Core>
#include<Eigen/Dense>
#include<Eigen/Geometry>
#include<opencv2/core/core.hpp>
#include<opencv2/core/eigen.hpp>
#include<opencv2/features2d/features2d.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/calib3d/calib3d.hpp>
#include<pcl-1.9/pcl/io/pcd_io.h>
#include<pcl-1.9/pcl/filters/voxel_grid.h>
#include<pcl-1.9/pcl/point_types.h>
#include<pcl-1.9/pcl/visualization/cloud_viewer.h>
#include<pcl-1.9/pcl/common/transforms.h>
#include<boost/format.hpp>
#include<sophus/se3.hpp>
#include<pangolin/pangolin.h>
#include<ceres/ceres.h>
#include<g2o/types/slam3d/types_slam3d.h>
#include<g2o/core/sparse_optimizer.h>
#include<g2o/core/g2o_core_api.h>
#include<g2o/core/base_vertex.h>
#include<g2o/core/base_unary_edge.h>
#include<g2o/core/block_solver.h>
#include<g2o/core/factory.h>
#include<g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include<g2o/core/optimization_algorithm_levenberg.h>
#include<g2o/core/optimization_algorithm_gauss_newton.h>
#include<g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include<g2o/solvers/dense/linear_solver_dense.h>
#include<g2o/solvers/csparse/linear_solver_csparse.h>
#include <sophus/se3.hpp>


using namespace std;
using namespace cv;
using namespace Sophus;
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud; 
void feature2Dcal(const cv::Mat Image0,const cv::Mat Image1,vector<KeyPoint> &keypoint1,vector<KeyPoint> &keypoint2,vector<cv::DMatch> &matches,vector<cv::DMatch> &better_matches);
void calC2C(vector<Point2d> point1,vector<Point2d> point2, Mat cameraMatrix,Mat &R,Mat &t);
void tranRt(Mat R,Mat t,Eigen::Matrix3d &R_M,Eigen::Vector3d &t_M);
void bundleAdjustmentG2O(
const VecVector3d &points_3d,
const VecVector2d &points_2d,
const Mat &K,
Sophus::SE3d &pose
);
void bundleAdjustmentGaussNewton(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose
);
void calC2W(vector<cv::DMatch> &better_matches,Mat depth_image,Mat cameraMatrix,vector<Point2d> point1,vector<Point2d> point2,Mat &RR,Mat &tt,Sophus::SE3d &pose_gn,Sophus::SE3d &pose_g2o);

class COST_FUNCTION;
class VertexPose;
class EdgeProjection;
PointCloud::Ptr image2PointCloud(Mat rgb,Mat depth,Mat cameraMatrix);

