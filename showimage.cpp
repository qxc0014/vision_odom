#include"slamBase.h"

PointCloud::Ptr image2PointCloud(Mat rgb,Mat depth,Mat cameraMatrix){
  PointCloud::Ptr cloud(new PointCloud() );
   for(int m = 0;m < depth.rows;m++){
     for(int n =0;n < depth.cols;n++){
       PointT p;
       p.z = double(depth.ptr<unsigned short>(m)[n]) / 1000.00;
       p.x = (n - cameraMatrix.at<double>(0,2)) * p.z/cameraMatrix.at<double>(0,0);
       p.y = (m - cameraMatrix.at<double>(1,2)) * p.z/cameraMatrix.at<double>(1,1);
       p.b = rgb.ptr<unsigned char>(m)[n*rgb.channels()];
       p.g = rgb.ptr<unsigned char>(m)[n*rgb.channels()+1];
       p.r = rgb.ptr<unsigned char>(m)[n*rgb.channels()+2];
      cloud->push_back(p);
     }
   }
   return cloud;
 }