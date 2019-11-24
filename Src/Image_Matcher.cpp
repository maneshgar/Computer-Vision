//
//  main.cpp
//  Midterm
//
//  Created by Behnam Maneshgar on 2016-02-12.
//  Copyright © 2016 Behnam Maneshgar. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

cv::Mat Gradient(cv::Mat inputImage, bool dx, bool dy){
    cv::Mat kernelX = (cv::Mat_<double>(3,3) <<  -0.125, 0,  0.125,
                       -0.25,  0, 0.25,
                       -0.125, 0,  0.125);
    
    cv::Mat kernelY = (cv::Mat_<double>(3,3) <<  -0.125, -0.25,  -0.125,
                       0,  0, 0,
                       0.125, 0.25,  0.125);
    cv::Mat gradient;
    
    if(dx && !dy){
        cv::filter2D(inputImage, gradient, -1, kernelX);
    }else if(!dx && dy){
        cv::filter2D(inputImage, gradient, -1, kernelY);
    }
    return gradient;
    
}

cv::Mat devideByScalar(cv::Mat input, float scalar){
    for (int u = 0; u <= input.rows; u++) {
        for (int v = 0; v <= input.cols; v++) {
            input.at<float>(u,v) = input.at<float>(u,v) / scalar;
        }
    }
    return input;
}

void printFloatMatrix(cv::Mat input){
    for (int i = 0 ; i < input.rows; i++) {
        for (int j = 0; j< input.cols; j++) {
            std::cout << input.at<float>(i,j) << ", ";
        }
        std::cout <<  ";" << std::endl;
    }
    std::cout << std::endl;
    return;
}

cv::Mat GaussianKernel(int sizex, int sizey, double sigma){
    cv::Mat Gauss(sizex,sizey,CV_32F);
    if (sizex % 2 != 1 || sizey % 2 != 1) {
        std::cout << std::endl << "Gaussian error: incorrect kernel dimension! " << std::endl;
        return Gauss;
    }
    int xrange = int(sizex-1)/2;
    int yrange = int(sizey-1)/2;
    float summation = 0;
    for (int u = -xrange; u <= xrange; u++) {
        for (int v = -yrange; v <= yrange; v++) {
            float test = (1/(2*M_PI*pow(sigma, 2))) * exp(-( (pow(u, 2) + pow(v, 2)) / (2*pow(sigma, 2))));
            summation += test;
            Gauss.at<float>((u+xrange), (v+yrange)) = test;
        }
    }
    //    devideByScalar(Gauss, summation);
    return Gauss;
}

std::vector<cv::KeyPoint> HarrisKeypointDetector(cv::Mat GrayScale, int sigma, float tresh){
    GrayScale.convertTo(GrayScale, CV_32F);
    int GKSize = (2*sigma) + 1;
    cv::Mat Gkernel;
    Gkernel = GaussianKernel(GKSize, GKSize, sigma);
    cv::Mat Ix = Gradient(GrayScale, true, false);
    cv::Mat Iy = Gradient(GrayScale, false, true);
    cv::Mat PIx, PIy;
    copyMakeBorder(Ix, PIx, sigma, sigma, sigma, sigma, cv::BORDER_REFLECT);
    copyMakeBorder(Iy, PIy, sigma, sigma, sigma, sigma, cv::BORDER_REFLECT);
    cv::Mat C(cv::Size(GrayScale.cols,GrayScale.rows),CV_32FC1);
    for (int i = sigma; i < GrayScale.rows + sigma; i++) {
        for (int j = sigma; j < GrayScale.cols + sigma; j++) {
            cv::Mat H(cv::Size(2,2),CV_32FC1, cv::Scalar(0,0,0));
            
            for (int x = -sigma; x <= sigma ; x++) {
                for (int y = -sigma; y <= sigma; y++) {
                    
                    H.at<float>(0,0) += PIx.at<float>(i+x,j+y) * PIx.at<float>(i+x,j+y) * Gkernel.at<float>(x+sigma , y+sigma);
                    
                    H.at<float>(0,1) += PIx.at<float>(i+x,j+y) * PIy.at<float>(i+x,j+y) * Gkernel.at<float>(x+sigma , y+sigma);
                    
                    H.at<float>(1,0) += PIy.at<float>(i+x,j+y) * PIx.at<float>(i+x,j+y) * Gkernel.at<float>(x+sigma , y+sigma);
                    
                    H.at<float>(1,1) += PIy.at<float>(i+x,j+y) * PIy.at<float>(i+x,j+y) * Gkernel.at<float>(x+sigma , y+sigma);
                }
            }
            
            float determinant = cv::determinant(H);
            cv::Scalar traceValue = cv::trace(H);
            if(traceValue[0] == 0){
                C.at<float>(i-sigma, j-sigma) = 0;
            }else{
                C.at<float>(i-sigma, j-sigma) = determinant / traceValue[0];
            }
        }
    }
    std::vector<cv::KeyPoint> Keys;
    int num = 0;
    for (int i = 0 ; i<C.rows; i++) {
        for (int j =0 ; j < C.cols; j++) {
            if (C.at<float>(i,j) > tresh) {
                bool isMax = true;
                for (int u = i-sigma ;  u <= i+sigma ; u++) {
                    for (int v = j-sigma ; v <= j+sigma ; v++) {
                        if( u>=0 && v >=0 && u<C.rows && v<C.cols && isMax){
                            if(!(i == u && j == v)){
                                if (C.at<float>(i,j) <= C.at<float>(u,v)) {
                                    isMax = false;
                                }
                            }
                        }
                    }
                }
                if (isMax) {
                    num ++;
                    Keys.push_back(cv::KeyPoint(j, i, 1000000));
                }
            }
        }
    }
    return Keys;
}

cv::Mat SIFTDescriptor(cv::Mat Image, std::vector<cv::KeyPoint> Keypoints){
    int radios = 8;
    cv::Mat descriptors(cv::Size(128,(int)Keypoints.size()), CV_32FC1, cv::Scalar(0));
    cv::Mat Ix, Iy;
    Image.convertTo(Image, CV_32F);
    
    Ix = Gradient(Image, true, false);
    Iy = Gradient(Image, false, true);
    for (int keyN = 0 ; keyN < Keypoints.size() ; keyN++) {
        int r=0;
        for (int i = Keypoints[keyN].pt.y - radios ; i < Keypoints[keyN].pt.y + radios; i+=4) {
            int c=0;
            for (int j = Keypoints[keyN].pt.x - radios ; j < Keypoints[keyN].pt.x + radios ; j+=4 ) {
                
                for (int u = 0; u<4 ; u++) {
                    for (int v = 0 ; v<4 ; v++) {
                        if (i+u>=0 && i+u<Image.rows && j+v>=0 && j+v<Image.cols) {
                            
                            float magn = pow(Ix.at<float>(i+u,j+v), 2) + pow(Iy.at<float>(i+u,j+v), 2);
                            magn = cv::sqrt(magn);
                            if (magn > 0) {
                                float teta = atan(Iy.at<float>(i+u,j+v)/Ix.at<float>(i+u,j+v)) * 180 / M_PI;
                                if(teta>0){
                                    if (Ix.at<float>(i+u,j+v) < 0 || (Ix.at<float>(i+u,j+v) == 0 && Iy.at<float>(i+u,j+v) < 0)) {
                                        teta+=180;
                                    }
                                }else if(teta == 0 && Ix.at<float>(i+u,j+v) < 0){
                                    teta += 180;
                                }else if (teta < 0 ){
                                    if (Ix.at<float>(i+u,j+v) > 0) {
                                        teta += 360;
                                    } else if (Ix.at<float>(i+u,j+v) < 0){
                                        teta += 180;
                                    } else if (Ix.at<float>(i+u,j+v) == 0){
                                        if (Iy.at<float>(i+u,j+v) < 0) {
                                            teta += 360;
                                        }else if (Iy.at<float>(i+u,j+v) > 0){
                                            teta += 180;
                                        }else{
                                            std::cout<< "BIG ERROR" << std::endl;
                                        }
                                    }
                                }
                                
                                teta -= 90;
                                if (teta<0) {
                                    teta+= 360;
                                }
                                int region = teta / 45;
                                float alpha = teta - (45*region);
                                alpha = alpha / 180 * M_PI;
                                descriptors.at<float>(keyN, r*32 + c*8 + region ) += magn*(cos(alpha)+sin(alpha)) ;
                                descriptors.at<float>(keyN, (r*32 + c*8) + region + 1 ) += magn*sin(alpha)*sqrt(2) ;
                                
                            }
                        }
                    }
                }
                c++;
            }
            r++;
        }
    }
    return descriptors;
}

std::vector<cv::DMatch> FindMatches(cv::Mat descriptor1, cv::Mat descriptor2, float treshold, float ratioTreshold){
    std::vector<cv::DMatch> matches;
    for (int i = 0; i<descriptor1.rows ; i++) {
        float bestMatchDist = 1000000;
        int bestMatchInd = 0;
        
        float SecondMatchDist = 1000000;
        int SecondMatchInd = 0;
        
        for (int j = 0 ; j<descriptor2.rows ; j++) {
            float distance = 0;
            for (int k = 0 ; k<descriptor1.cols ; k++) {
                distance += cv::abs(descriptor1.at<float>(i,k)-descriptor2.at<float>(j,k));
            }
            
            if (distance < bestMatchDist) {
                SecondMatchDist = bestMatchDist;
                SecondMatchInd = bestMatchInd;
                bestMatchDist = distance;
                bestMatchInd = j;
            }else if (distance < SecondMatchDist){
                SecondMatchDist = distance;
                SecondMatchInd = j;
            }
        }
        float ratio = bestMatchDist/SecondMatchDist;
        if (bestMatchDist<treshold && ratio < ratioTreshold) {
            matches.push_back(cv::DMatch(i, bestMatchInd, bestMatchDist));
        }
    }
    return matches;
}

int main(int argc, const char * argv[]) {
    // insert code here...
    // variables
    int sigma = 2;
    float distanceTreshold = 11000;
    float HarrisTreshold = 50;
    float rationTestTreshold = 0.75;
    //input images
    cv::Mat inputImage1 = cv::imread("Yosemite1.jpg");
    cv::Mat inputImage2 = cv::imread("Yosemite2.jpg");
    //
    cv::Mat GrayScale1,GrayScale2;
    cvtColor(inputImage1, GrayScale1, CV_BGR2GRAY);
    cvtColor(inputImage2, GrayScale2, CV_BGR2GRAY);
    std::cout << "started!" << std::endl;
    std::vector<cv::KeyPoint> Keys1 = HarrisKeypointDetector(GrayScale1, sigma, HarrisTreshold);
    std::vector<cv::KeyPoint> Keys2 = HarrisKeypointDetector(GrayScale2, sigma, HarrisTreshold);
    std::cout << "keys detected!" << std::endl;
    cv::Mat descriptor1 = SIFTDescriptor(GrayScale1, Keys1);
    cv::Mat descriptor2 = SIFTDescriptor(GrayScale2, Keys2);
    std::cout << "descriptors created!" << std::endl;
    std::vector<cv::DMatch> matches1 = FindMatches(descriptor1, descriptor2, distanceTreshold, rationTestTreshold);
    std::vector<cv::DMatch> matches2 = FindMatches(descriptor2, descriptor1, distanceTreshold, rationTestTreshold);
    std::cout << "matches found!"  << std::endl;
    std::vector<cv::DMatch> matches;
    for (int i=0 ; i<matches1.size(); i++) {
        bool exist = false;
        for (int j =0; j<matches2.size(); j++) {
            if (matches1[i].queryIdx == matches2[j].trainIdx && matches1[i].trainIdx == matches2[j].queryIdx) {
                exist = true;
            }
        }
        if (exist) {
            matches.push_back(matches1[i]);
        }
    }
    cv::Mat output;
    drawMatches(inputImage1, Keys1, inputImage2, Keys2, matches, output);
    namedWindow("Matches", cv::WINDOW_NORMAL);
    imshow("Matches", output);
    //output directory
    imwrite("output.jpg", output);
    std::cout << "the end";
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
