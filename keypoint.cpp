#include"slamBase.h"
#include<chrono>
int PIC_NUM = 300;//输入图像
cv::FileStorage fin;
static pcl::VoxelGrid<PointT> voxel;

int main(int argc, char const *argv[])
{
    /*读取图片*/
    vector<cv::Mat> Image;
    vector<Mat> depth_image; 
    Mat cameraMatrix;
    /*使用astra相机*/
    for(int i = 0;i < PIC_NUM;i++){
        boost::format png("../rgb_png/%d.%s");
        boost::format depth("../depth_png/%d.%s");
        Image.push_back(imread((png%(i+1)%"png").str()));
        depth_image.push_back(imread((depth%(i+1)%"png").str(),-1));
    }
    fin.open("../calibrationdata/ost.yaml",cv::FileStorage::READ);
    fin["camera_matrix"] >> cameraMatrix;
    fin.release();
    cout << cameraMatrix << endl;
    vector<Sophus::SE3d> T;
    vector<Eigen::Quaterniond> q;
    PointCloud::Ptr output2 ( new PointCloud() );
    PointCloud::Ptr output1 ( new PointCloud() );
    PointCloud::Ptr cloud_filtered ( new PointCloud() );
    output2->reserve(1000000);
    cloud_filtered->reserve(1000000);

    /*初始化G2O*/
    typedef g2o::BlockSolver_6_3 SlamBlockSolver;
    typedef g2o::LinearSolverCSparse< SlamBlockSolver::PoseMatrixType > SlamLinearSolver;
    
    SlamLinearSolver* linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering( false );
    SlamBlockSolver* blockSolver = new SlamBlockSolver( unique_ptr<SlamLinearSolver>(linearSolver) );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( unique_ptr<SlamBlockSolver>(blockSolver) );
    g2o::SparseOptimizer globalOptimizer;
    globalOptimizer.setAlgorithm( solver ); 
    globalOptimizer.setVerbose( false );
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId( 0 );
    v->setEstimate( Eigen::Isometry3d::Identity() );
    v->setFixed( true );
    globalOptimizer.addVertex( v );

    for(int ii = 0 ;ii < PIC_NUM -1;ii++ ){
        PointCloud::Ptr cloud1 ( new PointCloud() );
        PointCloud::Ptr cloud2 ( new PointCloud() );
        PointCloud::Ptr output ( new PointCloud() );
        Sophus::SE3d posec2c,pose_g2o;
        vector<KeyPoint> keypoint1,keypoint2;
        std::vector<cv::DMatch> matches,better_matches;
        /*输入图像输出关键点容器与匹配器*/
        feature2Dcal(Image[ii],Image[ii+1],keypoint1,keypoint2,matches,better_matches);
        vector<Point2d> point1,point2;
        for(size_t i = 0;i < better_matches.size();i++){
            point1.push_back(keypoint1[better_matches[i].queryIdx].pt);
            point2.push_back(keypoint2[better_matches[i].trainIdx].pt);
        }
        /*对极几何计算相机间的位姿变换*/
       // Mat R,t;
       // calC2C(point1,point2,cameraMatrix,R,t);
        /*转换R,t的数据格式*/
       //Eigen::Matrix3d R_M;
       // Eigen::Vector3d t_M;
        //tranRt(R,t,R_M,t_M);
       // Sophus::SE3d pose1(R_M,t_M);
       // cout << "对极约束求出来的SE3=\n"<< pose1.matrix() << endl;
        /*pnp计算相机外参*/
        Mat RR,tt;
        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        calC2W(better_matches,depth_image[ii],cameraMatrix,point1,point2,RR,tt,posec2c,pose_g2o); 
       /* Eigen::Isometry3d T;
        T = g2o::Isometry3(pose_g2o.matrix());
        cout << T.matrix() << endl;
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId( ii+1 );
        v->setEstimate( Eigen::Isometry3d::Identity());
        globalOptimizer.addVertex(v);
        g2o::EdgeSE3* edge = new g2o::EdgeSE3();
        edge->vertices() [0] = globalOptimizer.vertex( ii );
        edge->vertices() [1] = globalOptimizer.vertex( ii+1 );
        Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();
        information(0,0) = information(1,1) = information(2,2) = 100;
        information(3,3) = information(4,4) = information(5,5) = 100;
        edge->setInformation( information );
        edge->setMeasurement(  T );
        globalOptimizer.addEdge(edge);*/

        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
        cout << "time used:" << time_used.count() << "seconds" << endl; 
        cloud1 = image2PointCloud(Image[ii],depth_image[ii],cameraMatrix);
        cloud2 = image2PointCloud(Image[ii+1],depth_image[ii+1],cameraMatrix); 
          if(ii == 0){
          pcl::transformPointCloud(*cloud1, *output, pose_g2o.matrix());
          *output1 =*output + *cloud2;
          }else{
            pcl::transformPointCloud(*output1, *output2, pose_g2o.matrix());
            *output2 += *cloud2;
            *output1 = *output2;
          }
        ii++;
    }


    /*三维点云降采样*/
    voxel.setInputCloud( output2 );
    voxel.setLeafSize( 0.02, 0.02, 0.02 );
    voxel.filter(*cloud_filtered);
    pcl::visualization::CloudViewer viewer("kk");
    
    viewer.showCloud(cloud_filtered);
    while (!viewer.wasStopped())
    {       
    }
    //showPointCloud(pointcloud);
    return 0;
}


