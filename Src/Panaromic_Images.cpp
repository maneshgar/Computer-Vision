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
    for (int keyN = 0 ; keyN < (int)Keypoints.size() ; keyN++) {
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
        for (int j = 0 ; j<descriptor2.rows ; j++) {
            float distance = 0;
            for (int k = 0 ; k<descriptor1.cols ; k++) {
                distance += std::abs(descriptor1.at<float>(i,k)-descriptor2.at<float>(j,k));
            }
            
            if (distance < bestMatchDist) {
                SecondMatchDist = bestMatchDist;
                bestMatchDist = distance;
                bestMatchInd = j;
            }else if (distance < SecondMatchDist){
                SecondMatchDist = distance;
            }
        }
        float ratio = bestMatchDist/SecondMatchDist;
        if (bestMatchDist<treshold && ratio < ratioTreshold) {
            matches.push_back(cv::DMatch(i, bestMatchInd, bestMatchDist));
        }
    }
    return matches;
}

void project(float  x, float y, cv::Mat H, float *xout, float *yout){
    H.convertTo(H, CV_32F);
    *xout = (H.at<float>(0,0) * x + H.at<float>(0,1) * y + H.at<float>(0,2)) / (H.at<float>(2,0) * x + H.at<float>(2,1) * y + H.at<float>(2,2));
    *yout = (H.at<float>(1,0) * x + H.at<float>(1,1) * y + H.at<float>(1,2)) / (H.at<float>(2,0) * x + H.at<float>(2,1) * y + H.at<float>(2,2));
    return;
}

int computeInlierCount(cv::Mat H, std::vector<cv::DMatch> Matches, int numMatches, float inlierThreshold, std::vector<cv::KeyPoint> keys1, std::vector<cv::KeyPoint> keys2){
    int count = 0 ;
    for (int i = 0; i < numMatches; ++i) {
        float x1,y1,x2,y2,xp,yp;
        x1 = keys1[Matches[i].queryIdx].pt.x;
        y1 = keys1[Matches[i].queryIdx].pt.y;
        x2 = keys2[Matches[i].trainIdx].pt.x;
        y2 = keys2[Matches[i].trainIdx].pt.y;
        project(x2,y2,H,&xp,&yp);
        float distant;
        distant = std::abs(x1- xp) + std::abs(y1- yp);
        if (distant < inlierThreshold) {
            count++;
        }
    }
    return count;
}

void findInliers(cv::Mat H, std::vector<cv::DMatch> Matches, int numMatches, float inlierThreshold, std::vector<cv::KeyPoint> keys1, std::vector<cv::KeyPoint> keys2, std::vector<cv::DMatch> *inliers){
    for (int i = 0; i < numMatches; ++i) {
        float x1,y1,x2,y2,xp,yp;
        x1 = keys1[Matches[i].queryIdx].pt.x;
        y1 = keys1[Matches[i].queryIdx].pt.y;
        x2 = keys2[Matches[i].trainIdx].pt.x;
        y2 = keys2[Matches[i].trainIdx].pt.y;
        project(x2,y2,H,&xp,&yp);
        float distant;
        distant = std::abs(x1- xp) + std::abs(y1- yp);
        if (distant < inlierThreshold) {
            inliers->push_back(Matches[i]);
        }
    }
    std::cout << "Number of Inliers :: " << inliers->size() << std::endl;
    return;
}

void RANSAC(std::vector<cv::DMatch> matches, int numMatches, int numIterations, float inlierThreshold, cv::Mat *Hom, cv::Mat *homInv, std::vector<cv::KeyPoint> image1Display, std::vector<cv::KeyPoint> image2Display, std::vector<cv::DMatch> *inliers){
    
    int maxInlier = 0;
    cv::Mat homography;
    for (int  i= 0; i < numIterations; i++) {
        std::vector<cv::Point2f> srcPoints;
        std::vector<cv::Point2f> desPoints;
        cv::Mat tempHom;
        cv::RNG rng( 0xFFFFFFFF );
        unsigned int rand;
        for (int var = 0; var < numMatches; var++) {
            rand = rng.uniform(0,numMatches-1);
            cv::DMatch tempDm = matches[rand];
            srcPoints.push_back(image2Display[tempDm.trainIdx].pt);
            desPoints.push_back(image1Display[tempDm.queryIdx].pt);
        }
        tempHom = cv::findHomography(srcPoints, desPoints, 0);
        int numOfInlier = computeInlierCount(tempHom, matches, (int)matches.size(), inlierThreshold, image1Display, image2Display);
        if (numOfInlier > maxInlier) {
            maxInlier = numOfInlier;
            homography = tempHom.clone();
        }
    }
    std::vector<cv::DMatch> inl;
    findInliers(homography, matches, (int)matches.size(), inlierThreshold, image1Display, image2Display, &inl);
    std::vector<cv::Point2f> srcPoints;
    std::vector<cv::Point2f> desPoints;
    for (int var = 0; var < (int)inl.size(); var++) {
        cv::DMatch tempDm = inl[var];
        srcPoints.push_back(image2Display[tempDm.trainIdx].pt);
        desPoints.push_back(image1Display[tempDm.queryIdx].pt);
    }
    *Hom = cv::findHomography(srcPoints, desPoints, 0);
    *homInv = Hom->inv();
    *inliers = inl;
    return;
}

void stitch(cv::Mat Image1, cv::Mat Image2, cv::Mat Hom, cv::Mat homInv, cv::Mat *stitchedImage, cv::Mat *Assigned, bool isFirst){
    
    cv::Mat tempAssigned;
    tempAssigned = cv::Mat::zeros(10000,10000,CV_32FC1);
    
    //x tedad sotoon
    //y tedad satr
    
    float xp,yp;
    cv::Point2f TL,TR,BL,BR;
    cv::Point2f min,max, add, size;
    cv::Point2f image1Weight, image2Weight, Weight;
    TL.x = 0;
    TL.y = 0;
    BL.x = 0;
    BL.y = Image2.rows;
    BR.x = Image2.cols;
    BR.y = Image2.rows;
    TR.x = Image2.cols;
    TR.y = 0;
    
    project(TL.x, TL.y, Hom, &xp, &yp);
    
    min.x = xp;
    min.y = yp;
    max.x = xp;
    max.y = yp;
    
    project(BL.x, BL.y, Hom, &xp, &yp);
    
    if (xp>max.x) {
        max.x = xp;
    }if (xp<min.x) {
        min.x = xp;
    }if (yp>max.y) {
        max.y = yp;
    }if (yp<min.y) {
        min.y = yp;
    }
    
    project(BR.x, BR.y, Hom, &xp, &yp);
    
    if (xp>max.x) {
        max.x = xp;
    }if (xp<min.x) {
        min.x = xp;
    }if (yp>max.y) {
        max.y = yp;
    }if (yp<min.y) {
        min.y = yp;
    }
    project(TR.x, TR.y, Hom, &xp, &yp);
    
    if (xp>max.x) {
        max.x = xp;
    }if (xp<min.x) {
        min.x = xp;
    }if (yp>max.y) {
        max.y = yp;
    }if (yp<min.y) {
        min.y = yp;
    }
    
    if (min.x < 0) {
        add.x = -min.x+1;
    }else {
        add.x = 0;
    }
    if (min.y < 0) {
        add.y = -min.y+1;
    }else {
        add.y = 0;
    }
    
    //Compare with Image1
    if (max.x < Image1.cols) {
        max.x = Image1.cols;
    }if (max.y < Image1.rows) {
        max.y = Image1.rows;
    }if (min.x > 0) {
        min.x = 0;
    }if (min.y > 0) {
        min.y = 0;
    }
    
    size.x = std::abs(max.x-min.x);
    size.y = std::abs(max.y-min.y);
    
    cv::Mat stiched(cv::Size(size.x,size.y),CV_8UC3);
    stiched = cv::Mat::zeros(cv::Size(size.x,size.y),CV_8UC3);
    cv::Mat blend(cv::Size(size.x,size.y),CV_8UC3);
    blend = cv::Mat::zeros(cv::Size(size.x,size.y),CV_32FC1);
    
    //main photo
    for (int i = 0; i < Image1.rows; ++i) {
        for (int j = 0; j < Image1.cols; ++j) {
            stiched.at<cv::Vec3b>(i+(int)add.y, j+(int)add.x) = Image1.at<cv::Vec3b>(i,j);
            if (isFirst) {
                tempAssigned.at<float>(i+(int)add.y, j+(int)add.x) = 1;
            }else {
                tempAssigned.at<float>(i+(int)add.y, j+(int)add.x) = Assigned->at<float>(i,j);
            }
        }
    }
    //Inverse
    for (int i = 0; i < stiched.rows; ++i) {
        for (int j = 0; j < stiched.cols; ++j) {
            project(j-(int)add.x, i-(int)add.y, homInv, &xp, &yp);
            if(xp>0 && xp<Image2.cols && yp>0 && yp< Image2.rows){
                cv::Point2f center;
                center.x = xp;
                center.y = yp;
                cv::Mat patch;
                cv::getRectSubPix(Image2, cv::Size(1,1), center, patch);
                cv::Vec3b color;
                for (int pi = 0; pi < patch.rows; ++pi) {
                    for (int pj = 0; pj < patch.cols; ++pj) {
                        color += patch.at<cv::Vec3b>(pi,pj)/(patch.rows*patch.cols);
                    }
                }
                if (tempAssigned.at<float>(i,j) == 1) {
                    float blendingPercent = 1;
                    blendingPercent *= (1-(std::abs(xp-(Image2.cols/2))/(Image2.cols/2)));
                    blendingPercent *= (1-(std::abs(yp-(Image2.rows/2))/(Image2.rows/2)));
                    blend.at<float>(i,j) = blendingPercent;
                    stiched.at<cv::Vec3b>(i, j) = (blendingPercent*color) + ((1-blendingPercent) * stiched.at<cv::Vec3b>(i, j));
                    
                }else {
                    stiched.at<cv::Vec3b>(i, j) = color;
                    tempAssigned.at<float>(i,j) = 1;
                }
            }
        }
    }
    *stitchedImage = stiched.clone();
    *Assigned = tempAssigned.clone();
}

int main(){
    // insert code here...
    // variables
    int sigma = 2;
    float rationTestTreshold = 0.9;
    //input images
    std::vector<cv::Mat> Images;
    cv::Mat Assigned;
    Assigned = cv::Mat::zeros(10000,10000,CV_32FC1);
    bool isFirst = true;
    cv::Mat img;
    
    
    float distanceTreshold = 15000;
    float HarrisTreshold = 80;
    img = cv::imread("IMG_9363.JPG");
    Images.push_back(img);
    img = cv::imread("IMG_9362.JPG");
    Images.push_back(img);
    img = cv::imread("IMG_9361.JPG");
    Images.push_back(img);
    img = cv::imread("IMG_9360.JPG");
    Images.push_back(img);
    img = cv::imread("IMG_9359.JPG");
    Images.push_back(img);
    img = cv::imread("IMG_9358.JPG");
    Images.push_back(img);
    
    cv::Mat inputImage1 = Images[Images.size()-1];
    Images.pop_back();
    cv::Mat inputImage2, out1, out2;
    do {
        std::cout << "\n\tStarting to proccess new photo\n-------------------------------------------------" << std::endl;
        inputImage2 = Images[Images.size()-1];
        Images.pop_back();
        cv::Mat GrayScale1,GrayScale2;
        cvtColor(inputImage1, GrayScale1, CV_BGR2GRAY);
        cvtColor(inputImage2, GrayScale2, CV_BGR2GRAY);
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
        for (int i=0 ; i< (int)matches1.size(); i++) {
            bool exist = false;
            for (int j =0; j< (int)matches2.size(); j++) {
                if (matches1[i].queryIdx == matches2[j].trainIdx && matches1[i].trainIdx == matches2[j].queryIdx) {
                    exist = true;
                }
            }
            if (exist) {
                matches.push_back(matches1[i]);
            }
        }
        //START SECTION 3 , 4
        cv::Mat Hom,homInv;
        std::vector<cv::DMatch> inliers;
        RANSAC(matches,(int)matches.size(),2,100,&Hom,&homInv, Keys1, Keys2, &inliers);
        std::cout<< "RANSAC Done!" << std::endl;
        cv::Mat image;
        stitch(inputImage1,inputImage2,Hom,homInv,&image, &Assigned, isFirst);
        isFirst = false;
        std::cout << "Photos stitched!" << std::endl;
        inputImage1 = image.clone();
    } while (!Images.empty());
    cv::namedWindow("Panorama Image" , CV_WINDOW_KEEPRATIO);
    cv::imshow("Panorama Image", inputImage1);
    cv::imwrite("output.png", inputImage1);
    cv::waitKey(0);
    return 0;
}

