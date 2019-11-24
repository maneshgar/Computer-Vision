//
//  main.cpp
//  3D Reconstruction
//
//  Created by Behnam Maneshgar on 2016-03-12.
//  Copyright © 2016 Behnam Maneshgar. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <math.h>

#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Core>

using namespace std;
using namespace cv;
int maskThreshold = 200;

void findCenter_Radios(Point3f *center, double *radios, Mat maskImage){
    cvtColor(maskImage, maskImage, CV_BGR2GRAY);
    int TopRow=0, BotRow=0, LeftCol=0, RightCol=0;
    bool TopFound=false, BotFound=false, LeftFound=false, RightFound=false;
    bool rowExist=false, colExist=false;
    //finds the first and the last rows or columns that the Intensity more that threshold appears.
    //This will give the center and radios
    //search for Top and Down
    for (int i=0; i<maskImage.rows; i++) {
        rowExist = false;
        for (int j=0; j<maskImage.cols; j++) {
            if (maskImage.at<uchar>(i,j) > maskThreshold) {
                rowExist = true;
            }
        }
        if (rowExist && !TopFound) {
            TopFound = true;
            TopRow = i;
        }
        if (!rowExist && TopFound && !BotFound) {
            BotFound=true;
            BotRow = i-1;
            break;
        }
    }
    //search for left and right
    for (int i=0; i<maskImage.cols; i++) {
        colExist = false;
        for (int j=0; j<maskImage.rows; j++) {
            if (maskImage.at<uchar>(j,i)>maskThreshold) {
                colExist = true;
            }
        }
        if (colExist && !LeftFound) {
            LeftFound = true;
            LeftCol = i;
        }
        if (!colExist && LeftFound && !RightFound) {
            RightFound=true;
            RightCol = i-1;
            break;
        }
    }
    center->x = (RightCol + LeftCol) / 2;
    center->y = (BotRow + TopRow) / 2;
    center->z = 0;
    
    *radios = (RightCol - LeftCol) + (BotRow - TopRow);
    *radios /= 4;
    return;
}

double mapToSphere(Point3f reflectionPoint, Point3f center, double radios){
    //applies the formula of the sphere to find the z loc.
    double x,y,z;
    x = reflectionPoint.x - center.x;
    y = reflectionPoint.y - center.y;
    z = sqrt( pow(radios, 2) - (pow(x, 2) + pow(y, 2)) );
    return z;
}

Vec3f findDistanceVector(Point3f StartPoint, Point3f EndPoint){
    //calculates the difference between two points in the images
    Vec3f vector;
    vector[0] = EndPoint.x - StartPoint.x;
    vector[1] = -(EndPoint.y - StartPoint.y);
    vector[2] = EndPoint.z - StartPoint.z;
    return vector;
}

void findLightingVector(Vec3f *lightVector,Vec3f Normal,Vec3f viewVector){
    //calculates the light vector by normal and viewvector
    *lightVector = (2*(Normal.ddot(viewVector))*Normal)-viewVector;
    return;
}

Vec3f normalizeVector(Vec3f vector){
    Vec3f normalized;
    //find the lenght of the vector and devide each element by that
    double lenght = sqrt(pow(vector[0], 2) + pow(vector[1], 2) + pow(vector[2], 2));
    normalized[0] = vector[0]/lenght;
    normalized[1] = vector[1]/lenght;
    normalized[2] = vector[2]/lenght;
    return normalized;
}

Vec3f computeLightDirection(Mat maskImage, string Address){
    string prefixAdd = "Project/";
    Address = prefixAdd+Address;
    Mat reflectionImage = imread(Address);
    
    Point3f circleCenter, reflectionPoint, viewPoint;
    Vec3f Normal, viewDirection, lightDirection;
    double radios, temp;
    //calculates the center of the circle and the radios
    findCenter_Radios(&circleCenter, &radios, maskImage);
    //calculates the center of the shiny patch
    findCenter_Radios(&reflectionPoint, &temp, reflectionImage);
    //map the 2D dimension to 3D
    reflectionPoint.z = mapToSphere(reflectionPoint, circleCenter, radios);
    //Calculates the distance between center of the circle and shiny part of the sphere
    Normal = findDistanceVector(circleCenter, reflectionPoint);
    Normal = normalizeVector(Normal);
    //orhtogonal viewing
    viewDirection = {0,0,1};
    //calculates the light direction vector
    findLightingVector(&lightDirection, Normal, viewDirection);
    lightDirection = normalizeVector(lightDirection);
    return lightDirection;
}

Mat getLightDirections(){
    Mat maskImage = imread("chrome.mask.png");
    ifstream chromeFList ("chrome.txt");
    string line;
    getline (chromeFList,line);
    int numberOfLines = stoi(line);
    Mat LD(Mat(3, numberOfLines, CV_32FC1));
    //for each Image call computeLightDirection and set the result
    for (int i=0 ; i<numberOfLines ; i++) {
        getline (chromeFList,line);
        Vec3f lightDirection = computeLightDirection(maskImage, line);
        LD.at<float>(0,i) = lightDirection[0];
        LD.at<float>(1,i) = lightDirection[1];
        LD.at<float>(2,i) = lightDirection[2];
    }
    chromeFList.close();
    return LD;
}

vector<Mat> readInputImages(Mat maskImage){
    //read pathes from the file
    //Open each one and applies mask image
    //push to a vector in cv_32F type.
    vector<Mat> inputImages;
    string prefixAdd = "Project/";
    ifstream imagesPathList ("Project/psmImages/buddha.txt");
    string line;
    getline (imagesPathList,line);
    int numberOfLines = stoi(line);
    for (int i=0 ; i<numberOfLines ; i++) {
        getline (imagesPathList,line);
        line = prefixAdd+line;
        Mat image = imread(line);
        image = image.mul(maskImage);
        image.convertTo(image, CV_32FC1);
        inputImages.push_back(image);
    }
    imagesPathList.close();
    return inputImages;
}

Mat getPointIntensities(int row, int col, vector<Mat> images){
    //stores the point intensities for a pixel to a matrix
    Mat intensities(Mat(3, (int)images.size(), CV_32FC1));
    for (int i=0 ; i<images.size(); i++) {
        intensities.at<float>(0,i) = images[i].at<Vec3f>(row,col)[0];
        intensities.at<float>(1,i) = images[i].at<Vec3f>(row,col)[1];
        intensities.at<float>(2,i) = images[i].at<Vec3f>(row,col)[2];
    }
    return intensities;
}

Mat refineMask(Mat mask){
    //applies a threshold to the mask image
    //removes the low level mask Intensities.
    //convert all to 0 and 1 elements
    Mat refined(Mat(mask.rows, mask.cols, CV_8UC3));
    cvtColor(mask, mask, CV_BGR2GRAY);
    mask.convertTo(mask, CV_32F);
    for (int i=0 ; i<mask.rows ; i++) {
        for (int j=0; j<mask.cols ; j++) {
            Vec3b temp = refined.at<Vec3f>(i,j);
            if (mask.at<float>(i,j) < maskThreshold) {
                temp[0] = 0;
                temp[1] = 0;
                temp[2] = 0;
                refined.at<Vec3b>(i,j)=temp;
            }else{
                temp[0] = 1;
                temp[1] = 1;
                temp[2] = 1;
                refined.at<Vec3b>(i,j)=temp;
            }
        }
    }
    return refined;
}

Mat computG(Mat Intesities, Mat LightD){
    Mat G = (Intesities * LightD.t()) * ((LightD * LightD.t()).inv());
    return G;
}

void getElbidos(Mat *normalVectors, Mat *elbidos, vector<Mat> inputImages, Mat lightDirections, Mat maskImage){
    maskImage.convertTo(maskImage, CV_32F);
    Mat pointIntesities;
    //goes throght all the pixels
    for (int i=0 ; i<maskImage.rows ; i++) {
        for (int j=0 ; j<maskImage.cols ; j++) {
            //it works if it is inside the mask.
            if (maskImage.at<Vec3f>(i,j)[0] == 1) {
                Mat G;
                float Kdb, Kdg, Kdr;
                Mat Normal;
                Vec3f NormalVec;
                pointIntesities = getPointIntensities(i,j,inputImages);
                //computes the G matirx
                G = computG(pointIntesities.row(0), lightDirections);
                Kdb = norm(G);
                Normal = G/Kdb;
                NormalVec = normalizeVector(Normal);
                normalVectors->at<Vec3f>(i,j) = NormalVec;
                //next step move to color channels
                //saves all the directions in a Mat
                //goes for the green
                G = computG(pointIntesities.row(1), lightDirections);
                Kdg = norm(G);
                //goes for the red
                G = computG(pointIntesities.row(2), lightDirections);
                Kdr = norm(G);
                elbidos->at<Vec3f>(i,j) = {Kdb, Kdg, Kdr};
            }else{
                //all the pixels out of the mask should have normal 0,0,1
                normalVectors->at<Vec3f>(i,j)[0] = 0;
                normalVectors->at<Vec3f>(i,j)[1] = 0;
                normalVectors->at<Vec3f>(i,j)[2] = 1;
            }
        }
    }
    return;
}

Mat readLightDirections(){
    string prefixAdd = "Project/psmImages";
    ifstream imagesPathList ("Project/psmImages/lights.txt");
    string line;
    getline (imagesPathList,line);
    int numberOfLines = stoi(line);
    Mat LD(Mat(3, numberOfLines, CV_32FC1));
    for (int i=0 ; i<numberOfLines ; i++) {
        getline(imagesPathList, line);   //read stream line by line
        std::istringstream in(line);      //make a stream for the line itself
        float x, y, z;
        in >> x >> y >> z;       //now read the whitespace-separated floats
        //then stores at the light directions matrix
        LD.at<float>(0,i) = x;
        LD.at<float>(1,i) = y;
        LD.at<float>(2,i) = z;
    }
    return LD;
}

int main(int argc, const char * argv[]) {
    Mat lightDirections;
    vector<Mat> inputImages;
    string maskPath = "Project/psmImages/gray/gray.mask.jpg";
    //calculates the light directions from the center of the bright patch
    lightDirections = getLightDirections();
    // an optional function to read the light directions from the file
//    lightDirections = readLightDirections();
    cout << "light directions got!" << endl;

    Mat maskImageT = imread(maskPath);
    Mat maskImage = refineMask(maskImageT);
    cout << "Initialising ..." << endl;
    //read all the input images and store them in a vector
    inputImages = readInputImages(maskImage);
    cout << "Input Images read!" << endl;
    
    Mat elbidos(Mat(maskImage.rows, maskImage.cols, CV_32FC3));
    Mat normalVectors(Mat(maskImage.rows, maskImage.cols, CV_32FC3));
    //computes the elbidos and normals vectors
    getElbidos(&normalVectors, &elbidos, inputImages, lightDirections, maskImage);

    elbidos.convertTo(elbidos, CV_8U);
    imwrite("Project/psmImages/out/elbido.png", elbidos);
    
    imshow("test", normalVectors);
    imwrite("Project/psmImages/out/Normals.png", normalVectors*255);
    
    
    
    //finding depth
    int nOfPixels = maskImage.rows * maskImage.cols;
    int sizeA = 2 * nOfPixels;
    Eigen::SparseMatrix<double> A(sizeA, sizeA);
    Eigen::VectorXd x(sizeA);
    Eigen::VectorXd B(sizeA);
    //fills the A and B matrix for least square
    //counter is the ofset of the element on the matrix
    //writes equations for x and y directions one by one
    int counter = 0;
    for (int i=0; i<maskImage.rows; i++) {
        for (int j=0; j<maskImage.cols; j++) {
            A.insert(counter, counter) = normalVectors.at<Vec3f>(i,j)[2];
            B[counter] = -normalVectors.at<Vec3f>(i,j)[0];
            counter++;
            A.insert(counter, counter) = normalVectors.at<Vec3f>(i,j)[2];
            B[counter] = -normalVectors.at<Vec3f>(i,j)[1];
            counter++;
        }
    }
    //solve the least square with eigen
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double> > solver;
    solver.compute(A);
    cout << "Computed " << endl;
    if(solver.info()!= Eigen::Success) {
        // decomposition failed
        cout << "decomposition failed\n";
        return 0;
    }
    x = solver.solve(B);
    if(solver.info()!= Eigen::Success) {
        // solving failed
        cout << "solving failed\n";
        return 0;
    }
    cout << "Solved!\n" ;
    
    Mat DifX(Mat(maskImage.rows, maskImage.cols, CV_32FC1));
    Mat DifY(Mat(maskImage.rows, maskImage.cols, CV_32FC1));
    Mat ImageX(Mat(maskImage.rows, maskImage.cols, CV_32FC1));
    Mat ImageY(Mat(maskImage.rows, maskImage.cols, CV_32FC1));
    Mat Image(Mat(maskImage.rows, maskImage.cols, CV_8UC1));
    
    //xtract the answers from a single matrix
    counter = 0;
    for (int i=0 ; i<DifX.rows; i++) {
        for (int j=0; j<DifX.cols; j++) {
            DifX.at<float>(i,j) = x[counter];
            counter++;
            DifY.at<float>(i,j) = x[counter];
            counter++;
        }
    }
    //Calculates ImageX from differences vector
    for (int i =0; i<maskImage.rows; i++) {
        ImageX.at<float>(i,0) = 0;
        for (int j=0; j<maskImage.cols; j++) {
            if (j != maskImage.cols-1 && j != 0) {
                ImageX.at<float>(i,j) = ImageX.at<float>(i,j-1) + DifX.at<float>(i,j);
            }
        }
    }
    ImageX.convertTo(ImageX, CV_8U);
    //calculates ImageY form differences vector
    for (int i =0; i<maskImage.cols; i++) {
        ImageY.at<float>(maskImage.rows-1,i)=60;
        for (int j=maskImage.rows-1; j>=0; j--) {
            if (j != 0 && j != maskImage.rows-1) {
                ImageY.at<float>(j,i) = ImageY.at<float>(j+1,i) + DifY.at<float>(j,i);
            }
        }
    }
    ImageY.convertTo(ImageY, CV_8U);

    //take the average of the ImageX and ImageY
    for (int i =0; i<maskImage.rows; i++) {
        for (int j=0; j<maskImage.cols; j++) {
            if (maskImage.at<Vec3b>(i,j)[0] != 0) {
                Image.at<uchar>(i,j) = (uchar)((ImageX.at<uchar>(i,j) + ImageY.at<uchar>(i,j))/2);
            }else{
                Image.at<float>(i,j) = 0;
            }
        }
    }
    
    //scale to 0-255
    float min=1000, max=-1000;
    for (int i =0; i<maskImage.rows; i++) {
        for (int j=0; j<maskImage.cols; j++) {
            if (Image.at<uchar>(i,j)>max) {
                max = Image.at<uchar>(i,j);
            }
            if (Image.at<uchar>(i,j)<min) {
                min = Image.at<uchar>(i,j);
            }
            
        }
    }
    for (int i =0; i<maskImage.rows; i++) {
        for (int j=0; j<maskImage.cols; j++) {
            Image.at<uchar>(i,j) -= min;
            Image.at<uchar>(i,j) = Image.at<uchar>(i,j) *255/(max-min);
        }
    }
    //writes the calculated depth map
    imwrite("Project/psmImages/out/image.png", Image);
    imshow("Image", Image);
    waitKey(0);
    
    return 0;
}
