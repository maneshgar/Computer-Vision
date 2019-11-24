#ifndef __CLASS_H__
#define __CLASS_H__

#include "opencv2/opencv.hpp"

#define NUM_TILES_X         9
#define NUM_TILES_Y         6
#define TILE_SIZE          25

bool findCorrespondences(cv::Mat const &frame, std::vector<cv::Point2f> &corners)    {
    
    cv::Size pattern_size(NUM_TILES_X, NUM_TILES_Y);
    cv::Mat gray_image;
    cvtColor(frame, gray_image, CV_BGR2GRAY);
    
    bool pattern_found = cv::findChessboardCorners(gray_image, pattern_size, corners,
                                                   cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
    
    if(pattern_found)    {
        cv::cornerSubPix(gray_image, corners, cv::Size(11, 11), cv::Size(-1, -1),
                         cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
        
        cv::Mat dummy = frame;
        cv::drawChessboardCorners(dummy, pattern_size, cv::Mat(corners), pattern_found);
        
        cv::imshow("Current frame", dummy);
        while(true) {
            int key = cv::waitKey(0);
            switch(key) {
                case 13: return true;
                    break;
                case 8: //backspace
                    return false;
            }
        }
        
    }
    return false;
}

void captureFrames(cv::Size &image_size, std::vector<std::vector<cv::Point2f>> &calibration_corners)    {
    cv::VideoCapture cap(0);
    if (!cap.isOpened())    {
        return;
    }
    
    cv::Mat current_frame;
    int number_of_images=0;
    cv::namedWindow("Current frame");
    char file_name[100];
    bool not_done = true;
    std::vector<cv::Point2f> corners;
    while(not_done) {
        cap >> current_frame;
        image_size = current_frame.size();
        cv::imshow("Current frame", current_frame);
        int key = cv::waitKey(20);
        switch( key)    {
            case 13: //ENTER
                corners.clear();
                if (findCorrespondences(current_frame, corners)) {
                    sprintf(file_name,"calibration_image_%d.png",number_of_images++);
                    cv::imwrite(file_name,current_frame);
                    calibration_corners.push_back(corners);
                    std::cout << "Calibration image " << number_of_images << std::endl;
                }
                break;
            case 27: not_done = false;
                break;
        }
    }
    std::cout << "Total calibration images: " << number_of_images << std::endl;
    
    return;
}

void calculateParameters(std::vector<std::vector<cv::Point2f> > const &calibration_corners, cv::Size image_size, cv::Mat &camera_matrix, cv::Mat &distortion_coeffs, std::vector<cv::Mat> &rvecs, std::vector<cv::Mat> &tvecs)  {
    int nums = calibration_corners.size();
    
    std::vector<std::vector<cv::Point3f> > all_object_points;
    for (int i=0;i<nums;i++)    {
        std::vector<cv::Point3f> object_points;
        
        for (int j=0;j<calibration_corners[i].size();j++)   {
            object_points.push_back(cv::Point3f(TILE_SIZE * float(j %NUM_TILES_X) ,
                                                TILE_SIZE * float(j/NUM_TILES_X), 0));
        }
        
        all_object_points.push_back(object_points);
    }
    
    cv::calibrateCamera(all_object_points, calibration_corners, image_size, camera_matrix, distortion_coeffs, rvecs, tvecs);
    
    std::cout << "Camera matrix: " << camera_matrix << std::endl;
    std::cout << "Distortion coeffs: " << distortion_coeffs << std::endl;
    for (int i=0;i<nums;i++)    {
        std::cout << rvecs[i] << std::endl;
        std::cout << tvecs[i] << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;;
    }
    
    return;
}

bool calibrate()    {
    cv::Size image_size;
    std::vector<std::vector<cv::Point2f>> calibration_corners;
    captureFrames(image_size, calibration_corners);
    cv::Mat camera_matrix;
    cv::Mat distortion_coeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    calculateParameters(calibration_corners, image_size, camera_matrix, distortion_coeffs, rvecs,tvecs);
    return false;
}

int main(){
    calibrate();
    return 0;
}


#endif // __CLASS_H__
