//
//  main.cpp
//  Hybrid Images
//
//  Created by Behnam Maneshgar on 2016-01-24.
//  Copyright © 2016 Behnam Maneshgar. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void filtering(Mat const &input_image, Mat const &kernel, Mat &output_image){
    if (kernel.cols % 2 != 1 || kernel.rows % 2 != 1) {
        cout << endl << "Filtering error: incorrect kernel size! " << endl;
        return;
    }
    
    int nrows = input_image.rows;
    int ncols = input_image.cols;
    
    int HeightPadding = (kernel.rows-1)/2;
    int WidthPadding  = (kernel.cols-1)/2;
    
    Mat PaddedImage;
    copyMakeBorder(input_image, PaddedImage, HeightPadding, HeightPadding, WidthPadding, WidthPadding, BORDER_CONSTANT,0);
    
    PaddedImage.convertTo(PaddedImage, CV_32FC3);
    Mat output(nrows, ncols, CV_32FC3);
    //iterate on the main image
    for (int i = 0 ; i < nrows ; i++) {
        for (int j =0 ; j < ncols ; j++) {
            Vec3f tempRes = 0;
            //iterate on the kernel
            for (int x =0; x<kernel.rows ; x++) {
                for (int y=0 ; y<kernel.cols ; y++) {
                    tempRes += kernel.at<float>(x,y) * PaddedImage.at<Vec3f>(i+x , j+y);
                }
            }
            output.at<Vec3f>(i,j) = tempRes;
        }
    }
    output.convertTo(output, CV_16S);
    output_image = output.clone();
    return;
}

void GaussianKernel(int sizex, int sizey, double sigma, Mat &kernel){
    sizex = 3;
    sizey = 3;
    sigma = 5;
    if (sizex % 2 != 1 || sizey % 2 != 1) {
        cout << endl << "Gaussian error: incorrect kernel dimension! " << endl;
        return;
    }
    
    Mat Gauss(sizex,sizey,CV_32FC1);
    int xrange = int(sizex-1)/2;
    int yrange = int(sizey-1)/2;
    float summation = 0;
    for (int u = -xrange; u <= xrange; u++) {
        for (int v = -yrange; v <= yrange; v++) {
            
            float test = exp(-( (pow(u, 2)/(2*pow(sigma, 2))) + (pow(v, 2) / (2*pow(sigma, 2)))));
            summation += test;
            Gauss.at<float>((u+xrange), (v+yrange)) = test;
        }
    }
    Gauss = Gauss/summation;
    kernel = Gauss.clone();
    std::cout << kernel << std::endl;
}

void hybridImage(int mode){
    
    //Input Paths
    Mat Image1 = imread("dog.bmp");
    Mat Image2 = imread("cat.bmp");
    
    Mat GaussL;
    Mat GaussH;
    Mat HighPass;
    Mat LowPass;
    
    //Gaussian Blur
    Mat kernel;
    
    cout << "Filtering started ..." << endl;
    GaussianKernel(43, 43, 43, kernel);
    cout << "Kernel1 generated!" << endl;
    filtering(Image1, kernel, GaussL);
    cout << "Image1 filtered!" << endl;
    LowPass = GaussL.clone();
    
    Mat temp;
    Mat Hybrid;
    
    Mat X;
    Mat Y;
    
    switch (mode) {
        case 1:
            //Original - Gauss
            GaussianKernel(11, 11, 7, kernel);
            cout << "Kernel2 generated!" << endl;
            filtering(Image2, kernel, GaussH);
            cout << "Image2 filtered!" << endl;
            Image2.convertTo(Image2, CV_16S);
            HighPass = Image2 - GaussH;
            Image2.convertTo(Image2, CV_8U);
            GaussH.convertTo(GaussH, CV_8U);
            add(HighPass, 128, HighPass);
            LowPass.convertTo(LowPass, CV_16S);
            Hybrid = (HighPass + LowPass)/2;
            Hybrid.convertTo(Hybrid, CV_8U);
            HighPass.convertTo(HighPass, CV_8U);
            LowPass.convertTo(LowPass, CV_8U);
            
            break;
            
        case 2:
            Image2.convertTo(Image2, CV_16S);
            Sobel(Image2, HighPass, -1, 1, 1);
            add(HighPass, 128, HighPass);
            Hybrid = (HighPass + LowPass)/2;
            Hybrid.convertTo(Hybrid, CV_8U);
            HighPass.convertTo(HighPass, CV_8U);
            break;
            
        case 3:
            //Sobel operator
            GaussianKernel(5, 5, 2, kernel);
            filtering(Image1, kernel, GaussL);
            LowPass = GaussL.clone();

            Image2.convertTo(Image2, CV_32F);
            Sobel(Image2, X, -1, 1, 0);
            Sobel(Image2, Y, -1, 0, 1);
            X = X.mul(X);
            Y = Y.mul(Y);
            X = X+Y;
            cv::sqrt(X, HighPass);
            HighPass.convertTo(HighPass, CV_16S);
            add(HighPass, 128, HighPass);
            HighPass.convertTo(HighPass, CV_8U);
            LowPass.convertTo(LowPass, CV_8U);
            Hybrid = (HighPass + LowPass)/2;
            Hybrid.convertTo(Hybrid, CV_8U);
            break;
            
        case 4:
            //Difference of Gayssians
            Image2.convertTo(Image2, CV_16S);
            GaussianKernel(11, 11, 7, kernel);
            cout << "Kernel2 generated!" << endl;
            filtering(Image2, kernel, GaussH);
            cout << "Image2 filtered!" << endl;
            
            GaussianKernel(9, 9, 1, kernel);
            cout << "Kernel3 generated!" << endl;
            filtering(Image2, kernel, temp);

            GaussH.convertTo(GaussH, CV_16S);
            temp.convertTo(temp, CV_16S);
            HighPass = GaussH - temp;
            add(HighPass, 128, HighPass);
            LowPass.convertTo(LowPass, CV_16S);
            Hybrid = (HighPass + LowPass)/2;
            Hybrid.convertTo(Hybrid, CV_8U);
            HighPass.convertTo(HighPass, CV_8U);
            break;
            
        case 5:
            //Laplacian of Gaussian
            GaussianKernel(3, 3, 1, kernel);
            cout << "Kernel2 generated!" << endl;
            filtering(Image2, kernel, GaussH);
            cout << "Image2 filtered!" << endl;
            
            Laplacian(GaussH, HighPass, -1);
            
            add(HighPass, 128, HighPass);
            LowPass.convertTo(LowPass, CV_16S);
            Hybrid = (HighPass + LowPass)/2;
            Hybrid.convertTo(Hybrid, CV_8U);
            HighPass.convertTo(HighPass, CV_8U);
            break;
            
        default:
            cout << "Input is not valid!" << endl;
            return;
            break;
    }
    
    imshow("Hybrid", Hybrid);
    imshow("High Frequency", HighPass);
    
    //Out put path
    imwrite("outHibrid.bmp", Hybrid);
    imwrite("outHighF.bmp", HighPass);
    
    cout << endl << "Finished!\nThank you for using this application." << endl;
    waitKey(0);
}

int main(int argc, const char * argv[]) {
    int i;
    cout << "Please choose type of filtering::\n1- simply Original - Gaussian\n2- Sobel operator by using the following method :: Sobel(src, dst, ddepth, 1, 1)\n3- Sobel operator by computing dx and dy seperatly\n4- Difference of Gayssians\n5- Laplacian of Gaussian\n" << endl;
    cin >> i;
    hybridImage(i);
    waitKey(0);
    return 0;
}
