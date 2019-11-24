//
//  main.cpp
//  inclass
//
//  Created by Behnam Maneshgar on 2016-01-29.
//  Copyright © 2016 Behnam Maneshgar. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

void demoSIFTKeypoints(){
    cv::Mat img = cv::imread("img1.ppm");
    
    //Detect keypoint
    cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(img, keypoints);
    
    //Add results to image and save
    cv::Mat output;
    cv::drawKeypoints(img, keypoints, output);
    cv::imwrite("demoSIFTKeypoints.jpg", output);
    
    cv::imshow("demoSIFTkeypoints", output);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return;
}

void demoSURFKeypoints(){
    cv::Mat img = cv::imread("dog.bmp");
    
    //Detect keypoints
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();
    //    cv::SurfFeatureDetector detector;
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(img,keypoints);
    
    //Add results to image and save
    cv::Mat output;
    cv::drawKeypoints(img, keypoints, output);
    cv::imwrite("demoSURF_result.jpg", output);
    
    cv::imshow("demoSURFkeypoints", output);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return;
}

void demoSURFMatching(){
    cv::Mat img_1 = cv::imread("pano1_0008.png");
    cv::Mat img_2 = cv::imread("/pano1_0009.png");
    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;
    
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );
    //    SurfFeatureDetector detector( minHessian );
    std::vector<cv::KeyPoint> keypoints_1,keypoints_2;
    
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);
    
    //--Step 2: Calculate descriptors ( feature vectors)
    cv::Ptr<cv::xfeatures2d::SURF> extractor = cv::xfeatures2d::SURF::create();
    cv::Mat descriptors_1,descriptors_2;
    
    extractor->compute(img_1, keypoints_1, descriptors_1);
    extractor->compute(img_2, keypoints_2, descriptors_2);
    
    //--Step 3: Matching descriptor vectors using FLANN matcher
    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);
    
    double max_dist = 0;
    double min_dist = 100;
    
    //--Quick calculation of max and min distances between keypoints
    for (int i=0; i<descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) {
            min_dist = dist;
        }
        if (dist > max_dist) {
            max_dist = dist;
        }
    }
    
    std::cout << "--Max dist: " << max_dist << std::endl;
    std::cout << "--Min dist: " << min_dist << std::endl;
    
    //--Draw only "good" matches(i.e. whose distanc is less than 2*min_dist, or a small arbitrary value ( 0.02 ) in the event that min_dist is very small)
    //PS - rediusMatch can also be used here.
    
    std::vector<cv::DMatch> good_matches;
    
    for (int i=0; i <descriptors_1.rows; i++) {
        if (matches[i].distance <= std::max(2*min_dist,0.02)) {
            good_matches.push_back(matches[i]);
        }
    }
    
    //--Draw only good matches
    cv::Mat img_matches;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    //--Show detected matches
    cv::imshow("Good Matches", img_matches);
    cv::imwrite("demoSURF_result.jpg", img_matches);

    for (int i=0; i< (int)good_matches.size(); i++) {
        std::cout << "--Good MAtch [" << i << "] Keypoint 1:" << good_matches[i].queryIdx << "--Key" << std::endl;
    }
    cv::waitKey(0);
    cv::destroyAllWindows();
}


int main(int argc, const char * argv[]) {
    int demo_id = -1;
    while (demo_id != 0) {
        std::cout << "Enter number: \n1- demoSIFTKeypoints\n2- demoSURFMatching\n3- demoSURFKeypoints" << std::endl;
        std::cin >> demo_id;
        switch (demo_id) {
            case 1:
                demoSIFTKeypoints();
                break;
            case 2:
                demoSURFMatching();
                break;
            case 3:
                demoSURFKeypoints();
        }
    }
    return 0;
}
