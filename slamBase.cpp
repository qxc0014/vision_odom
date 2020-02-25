#include"slamBase.h"
void feature2Dcal(const cv::Mat Image0,const cv::Mat Image1,vector<KeyPoint> &keypoint1,vector<KeyPoint> &keypoint2,vector<cv::DMatch> &matches,vector<cv::DMatch> &better_matches)
{
    cv::Mat descriptor1,descriptor2;
    Ptr<FeatureDetector> detect = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> match = DescriptorMatcher::create("BruteForce-Hamming");

    detect->detect(Image0,keypoint1);
    detect->detect(Image1,keypoint2);

    descriptor->compute(Image0,keypoint1,descriptor1);
    descriptor->compute(Image1,keypoint2,descriptor2);

    Mat outimg1,outimg2;
    //drawKeypoints(Image[0],keypoint1,outimg1);
    //imshow("image1",outimg1);

    match->match(descriptor1,descriptor2,matches);
    /* 返回两对键值对给minmax，q1传入的是最小匹配对，q2传入的是最大匹配对*/
   // auto minmax = minmax_element(matches.begin(),matches.end(),[](const DMatch &q1,const DMatch &q2) {return q1.distance < q2.distance;});
    auto minmax = minmax_element(matches.begin(),matches.end());
    double min_dist = minmax.first->distance;
    double max_dist = minmax.second->distance;

    //cout << "min distance =" << min_dist << endl;
    for(int i = 0;i < descriptor1.rows;i++){//描述子矩阵的一行代表一个关键点的描述子
        if(matches[i].distance <= max(2 * min_dist,20.0)){
            better_matches.push_back(matches[i]);
        }
    }
    //Mat outimg;
    
    /*drawMatches(Image0,keypoint1,Image1,keypoint2,matches,outimg1);
    imshow("1 to 2",outimg1);
    drawMatches(Image0,keypoint1,Image1,keypoint2,better_matches,outimg2);
    imshow("1 to 2 better",outimg2);*/
}