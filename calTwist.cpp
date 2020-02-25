#include"slamBase.h"

void calC2C(vector<Point2d> point1,vector<Point2d> point2, Mat cameraMatrix,Mat &R,Mat &t){
    Mat fundamental_matrix;
    Mat essential_matrix;  

    fundamental_matrix = findFundamentalMat(point1,point2,CV_FM_8POINT); 
    essential_matrix = findEssentialMat(point1,point2,cameraMatrix);
    recoverPose(essential_matrix,point1,point2,cameraMatrix,R,t);
}
void tranRt(Mat R,Mat t,Eigen::Matrix3d &R_M,Eigen::Vector3d &t_M){
    cv2eigen(R,R_M);
    cv2eigen(t,t_M);
}

void calC2W(vector<cv::DMatch> &better_matches,Mat depth_image,Mat cameraMatrix,vector<Point2d> point1,vector<Point2d> point2,Mat &RR,Mat &tt,Sophus::SE3d& pose_gn,Sophus::SE3d &pose_g2o){
    vector<Point3d> point1_w_3d;//世界坐标系下的点坐标
    vector<Point2d> point2_c_2d;//摄像头坐标系下的归一化坐标
    vector<Point2f> point2_match;

    for(int i =0;i< better_matches.size();i++){
        ushort dd;
        dd = depth_image.ptr<unsigned short>(int(point1[i].y))[int(point1[i].x)];
        double d = dd / 1000.00;
        Point2d point1_c_2d;
        if (dd == 0){ 
            continue;
          }   // bad depth
       
        point1_c_2d.x = (point1[i].x - cameraMatrix.at<double>(0,2))/cameraMatrix.at<double>(0,0);
        point1_c_2d.y = (point1[i].y - cameraMatrix.at<double>(1,2))/cameraMatrix.at<double>(1,1);
        point1_w_3d.push_back(Point3d(point1_c_2d.x*d,point1_c_2d.y*d,d));  
        point2_match.push_back(Point2d(point2[i].x,point2[i].y));//将筛选出来的关键点存入新的容器
    }   
    Mat r;
    /*solvePnP(point1_w_3d,point2_match,cameraMatrix,Mat(),r,tt,false);
    cv:Rodrigues(r,RR);*/

    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;
    Sophus::Vector6d se3;
    ceres::Problem problem;
      for (size_t i = 0; i < point1_w_3d.size(); ++i) {
        pts_3d_eigen.push_back(Eigen::Vector3d(point1_w_3d[i].x, point1_w_3d[i].y, point1_w_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(point2_match[i].x, point2_match[i].y));
        /*ceres::CostFunction *cost_function;
        cost_function = new COST_FUNCTION(pts_3d_eigen[i],pts_2d_eigen[i]);
        problem.AddResidualBlock(cost_function, NULL, se3.data());*/

  }
     /* ceres::Solver::Options options;
      options.dynamic_sparsity = true;
      options.max_num_iterations = 100;
      options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
      options.minimizer_type = ceres::TRUST_REGION;
      options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
      options.trust_region_strategy_type = ceres::DOGLEG;
      options.minimizer_progress_to_stdout = true;
      options.dogleg_type = ceres::SUBSPACE_DOGLEG;

      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      std::cout << summary.BriefReport() << "\n";
      std::cout << "estimated pose: \n" << Sophus::SE3d::exp(se3).matrix() << std::endl;*/
  
  //bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, cameraMatrix, pose_gn);
  //cout << "高斯牛顿：\n"<< pose_gn.matrix3x4() << endl; 
  bundleAdjustmentG2O(pts_3d_eigen,pts_2d_eigen,cameraMatrix,pose_g2o);
  //cout << "G2O:\n"<< pose_g2o.matrix3x4() << endl; 
}