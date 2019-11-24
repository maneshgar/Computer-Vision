#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]){
    
    Mat img = imread("oldwell_mosaic.png");
    cvtColor(img, img, CV_BGR2GRAY);
    img.convertTo(img, CV_32F);
    img /= 255.0 ;


    if (img.empty()){
        cout << "Error : Image cannot be loaded..!!" << endl;
        
        return -1;
    }

    int Width = img.cols;
    int Height = img.rows;
    
    Mat Red   = Mat::zeros(Height,Width,CV_32FC1);
    Mat Green = Mat::zeros(Height,Width,CV_32FC1);
    Mat Blue  = Mat::zeros(Height,Width,CV_32FC1);
    
    waitKey(0);
    //Extract 3 Colors channel
    //    //Extract the Red, Blue and Green
        for (int i = 0 ; i<Height ; i++) {
            for (int j = 0 ; j<Width ; j++) {
                float intensity = img.at<float>(i, j);
                if(i%2 == 0){
                    if(j%2 == 0){
                        //RED
                        Blue.at<float>(i,j) = 0;
                        Green.at<float>(i,j) = 0;
                        Red.at<float>(i,j) = intensity;
                    }else{
                        //GREEN
                        Blue.at<float>(i,j) = 0;
                        Green.at<float>(i,j) = intensity;
                        Red.at<float>(i,j) = 0;
                    }
                }else{
                    if(j%2 == 0){
                        //GREEN
                        Blue.at<float>(i,j) = 0;
                        Green.at<float>(i,j) = intensity;
                        Red.at<float>(i,j) = 0;
                    }else{
                        //BLUE
                        Blue.at<float>(i,j) = intensity;
                        Green.at<float>(i,j) = 0;
                        Red.at<float>(i,j) = 0;
                    }
                }
            }
        }

    //2D filters
    Mat kernelG = (Mat_<float>(3,3) << 0.0, 0.25, 0.0, 0.25, 1, 0.25, 0.0, 0.25, 0.0 );
    Mat kernelB = (Mat_<float>(3,3) << 0.25, 0.5, 0.25, 0.5, 1, 0.5, 0.25, 0.5, 0.25);
    Mat kernelR = (Mat_<float>(3,3) << 0.25, 0.5, 0.25, 0.5, 1, 0.5, 0.25, 0.5, 0.25);
    

    Mat result_G(Height, Width, CV_32FC1);
    Mat result_B(Height, Width, CV_32FC1);
    Mat result_R(Height, Width, CV_32FC1);
    
    
    filter2D(Blue,result_B,-1,kernelB);
    filter2D(Green,result_G,-1,kernelG);
    filter2D(Red,result_R,-1,kernelR);

    std::vector<Mat> channels;
    channels.resize(3);
    channels[0] = result_B;
    channels[1] = result_G;
    channels[2] = result_R;
    
    Mat final_img1;
    merge(channels, final_img1);
    imshow("final1", final_img1);
    waitKey(0);
    /////////////////
    Mat R_G = Mat::zeros(Height,Width,CV_32FC1);
    Mat B_G = Mat::zeros(Height,Width,CV_32FC1);

    R_G = abs(result_R - result_G);
    B_G = abs(result_B - result_G);
    
    vector<Mat> channels2;
    
    channels2.push_back(B_G);
    channels2.push_back(result_G);
    channels2.push_back(R_G);
    
    Mat final_img2;
    merge(channels2, final_img2);
    imshow("final2", final_img2);
    waitKey(0);
    
    
    
    Mat Diff1,Diff2;
    
    Mat RGB_img = imread("oldwell.jpg");
    
    /*subtract(RGB_img, final_img1, Diff1, noArray(), -1);
     subtract(RGB_img, final_img2, Diff2, noArray(), -1);
     */
    final_img1.convertTo(final_img1, CV_16S);
    final_img2.convertTo(final_img2, CV_16S);
    RGB_img.convertTo(RGB_img, CV_16S);
    
    Diff1 = abs(RGB_img - final_img1);
    Diff2 = abs(RGB_img - final_img2);
    
    Diff1.convertTo(Diff1,CV_32F);
    Diff2.convertTo(Diff2,CV_32F);
    
    namedWindow("outputImage1", CV_WINDOW_AUTOSIZE); //create a window original image
    imshow("outputImage1", Diff1);
    
    namedWindow("outputImage2", CV_WINDOW_AUTOSIZE); //create a window original image
    imshow("outputImage2", Diff2);
    
    waitKey(0); //wait infinite 
    
    return 0;
}
