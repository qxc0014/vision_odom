#include"slamBase.h"
class COST_FUNCTION:public ceres::SizedCostFunction<2,6>{//残差维数为2*6
  public:
      COST_FUNCTION(Vector3d a,Vector2d b):point3d(a),point2d(b){}//ceres的构造函数，输入为参数
      virtual ~COST_FUNCTION(){}
      virtual 
      bool Evaluate(
      double const* const* parameters,//待优化参数
      double *residuals,//参差
      double **jacobians)//雅克比矩阵
      const{
         Eigen::Map<const Eigen::Matrix<double,6,1>> T_se3(*parameters);
         Sophus::SE3d T_SE3 = Sophus::SE3d::exp(T_se3);
         Eigen::Vector3d Pc = T_SE3 * point3d;
         Eigen::Matrix3d K;
         double fx = 723.5,fy = 729.6,cx = 388.6,cy = 256.46;

         K << fx,0.0,fy,0.0,fy,cy,0.0,0.0,1.0;
         Eigen::Vector2d residual = point2d - (K*Pc).hnormalized();
         residuals[0] = residual[0];
         residuals[1] = residual[1];
         if(jacobians != NULL){
           Eigen::Matrix<double,2,6> J;
           double x = Pc[0];
           double y = Pc[1];
           double z = Pc[2];
           
           double x2 = x*x;
           double y2 = y*y;
           double z2 = z*z;

            J(0,0) = -fx/z;
            J(0,1) =  0;
            J(0,2) =  fx*x/z2;
            J(0,3) =  fx*x*y/z2;
            J(0,4) = -fx-fx*x2/z2;
            J(0,5) =  fx*y/z;
            J(1,0) =  0;
            J(1,1) = -fy/z;
            J(1,2) =  fy*y/z2;
            J(1,3) =  fy+fy*y2/z2;
            J(1,4) = -fy*x*y/z2;
            J(1,5) = -fy*x/z;
            int k=0;
            for(int i=0; i<2; ++i) {
                for(int j=0; j<6; ++j) {
                    jacobians[0][k++] = J(i,j);
                }
            }

         }
      }
  private:
     const Vector3d point3d;
     const Vector2d point2d;
};
/*-------------------------------g2o--------------------------------*/
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() override {
    _estimate = Sophus::SE3d();
  }

  /// left multiplication on SE3
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }

  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}
};
class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}

  virtual void computeError() override {
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
    pos_pixel /= pos_pixel[2];
    _error = _measurement - pos_pixel.head<2>();
  }

  virtual void linearizeOplus() override {
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_cam = T * _pos3d;
    double fx = _K(0, 0);
    double fy = _K(1, 1);
    double cx = _K(0, 2);
    double cy = _K(1, 2);
    double X = pos_cam[0];
    double Y = pos_cam[1];
    double Z = pos_cam[2];
    double Z2 = Z * Z;
    _jacobianOplusXi
      << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
      0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
  }

  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}

private:
  Eigen::Vector3d _pos3d;
  Eigen::Matrix3d _K;
};
void bundleAdjustmentGaussNewton(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose) {
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  const int iterations = 10;//高斯牛顿迭代次数
  double cost = 0, lastCost = 0;
  double fx = K.at<double>(0, 0);
  double fy = K.at<double>(1, 1);
  double cx = K.at<double>(0, 2);
  double cy = K.at<double>(1, 2);

  for (int iter = 0; iter < iterations; iter++) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Vector6d b = Vector6d::Zero();

    cost = 0;
    // compute cost
    for (int i = 0; i < points_3d.size(); i++) {
      Eigen::Vector3d pc = pose * points_3d[i];
      double inv_z = 1.0 / pc[2];
      double inv_z2 = inv_z * inv_z;
      Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, 
      fy * pc[1] / pc[2] + cy);//像素坐标预测值

      Eigen::Vector2d e = points_2d[i] - proj;//像素坐标观测值-预测值

      cost += e.squaredNorm();//向量与本身的点积
      Eigen::Matrix<double, 2, 6> J;
      /*计算雅克比矩阵*/
      J << -fx * inv_z,
        0,
        fx * pc[0] * inv_z2,
        fx * pc[0] * pc[1] * inv_z2,
        -fx - fx * pc[0] * pc[0] * inv_z2,
        fx * pc[1] * inv_z,
        0,
        -fy * inv_z,
        fy * pc[1] * inv_z,
        fy + fy * pc[1] * pc[1] * inv_z2,
        -fy * pc[0] * pc[1] * inv_z2,
        -fy * pc[0] * inv_z;
      H += J.transpose() * J;//求海塞
      b += -J.transpose() * e;
    }
    Vector6d dx;
    dx = H.ldlt().solve(b);
    if (isnan(dx[0])) {
      cout << "result is nan!" << endl;
      break;
    }
    if (iter > 0 && cost >= lastCost) {
      cout << "cost: " << cost << ", last cost: " << lastCost << endl;
      break;
    }
    pose = Sophus::SE3d::exp(dx) * pose;
    lastCost = cost;

    cout << "迭代第 " << iter << " cost=" << cout.precision(12) << cost << endl;
    if (dx.norm() < 1e-6) {
      break;
    }
  }
 /* cout << "高斯牛顿下降计算的SE3=\n" << pose.matrix() << endl;
  cout << "高斯牛顿下降计算的平移矩阵=\n"<< pose.translation() << endl;
  cout << "高斯牛顿下降计算的旋转矩阵=\n"<< pose.rotationMatrix()<< endl;
  Eigen::Quaterniond q(pose.rotationMatrix());
  cout << q.coeffs().w() <<" " << q.coeffs().x() << " "<< q.coeffs().y()<< " "<<q.coeffs().z() <<endl;*/
}
/*------------------------------G2O实现---------------------------------------------*/
void bundleAdjustmentG2O(
const VecVector3d &points_3d,
const VecVector2d &points_2d,
const Mat &K,
Sophus::SE3d &pose){
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;  // pose is 6, landmark is 3
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;     // 图模型
  optimizer.setAlgorithm(solver);   // 设置求解器
  optimizer.setVerbose(true);       // 打开调试输出

  // vertex
  VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
  vertex_pose->setId(0);
  vertex_pose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(vertex_pose);

  // K
  Eigen::Matrix3d K_eigen;
  K_eigen <<
          K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
    K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
    K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

  // edges
  int index = 1;
  for (size_t i = 0; i < points_2d.size(); ++i) {
    auto p2d = points_2d[i];
    auto p3d = points_3d[i];
    EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
    edge->setId(index);
    edge->setVertex(0, vertex_pose);
    edge->setMeasurement(p2d);
    edge->setInformation(Eigen::Matrix2d::Identity());
    optimizer.addEdge(edge);
    index++;
  }

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.setVerbose(true);
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
  cout << "pose estimated by g2o =\n" << vertex_pose->estimate().matrix() << endl;
  pose = vertex_pose->estimate();
}