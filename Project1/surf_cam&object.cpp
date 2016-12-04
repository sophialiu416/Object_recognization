// find keypoints in book image and webcam image, then try to match them. This one doesnt work so well

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <stdio.h>
#include<iostream>
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

///////////////////////////////////////////////////////////////////////////////////////////////////
int main() {
	Mat img1 = imread("book.jpg", IMREAD_GRAYSCALE);
	Mat img_1; 
	cv::resize(img1, img_1, Size(), 0.25, 0.25);  //make the book image smaller
	if (!img_1.data)
	{
		std::cout << " --(!) Error reading images " << std::endl; return -2;
	}

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;
	Ptr<SURF> detector = SURF::create(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	detector->detect(img_1, keypoints_1);

	//-- Step 2: Draw keypoints
	Mat img_keypoints_1; Mat img_keypoints_2;
	drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	Ptr<SURF> extractor = SURF::create();
	Mat descriptors_1, descriptors_2;
	extractor->compute(img_1, keypoints_1, descriptors_1);

	//-- Step 3: Matching descriptor vectors with a brute force matcher
	BFMatcher matcher(NORM_L2);
	std::vector< DMatch > matches;
	Mat img_matches;
	cv::VideoCapture capWebcam(0);       
	if (capWebcam.isOpened() == false) {                              
		std::cout << "error: capWebcam not accessed successfully\n\n";     
		return(0);                                                
	}
	cv::Mat imgOriginal;    
	cv::Mat imgGrayscale;
	char charCheckForEscKey = 0;
	cout << "Please press Esc key to stop"<<endl;
	while (charCheckForEscKey != 27 && capWebcam.isOpened()) {            // until the Esc key is pressed or webcam connection is lost
		bool blnFrameReadSuccessfully = capWebcam.read(imgOriginal);            // get next frame
		if (!blnFrameReadSuccessfully || imgOriginal.empty()) {              
			std::cout << "error: frame not read from webcam\n";             
			break;                                                             
		}

		cv::cvtColor(imgOriginal, imgGrayscale, CV_BGR2GRAY);         
		detector->detect(imgGrayscale, keypoints_2);
		drawKeypoints(imgGrayscale, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		extractor->compute(imgGrayscale, keypoints_2, descriptors_2);
		matcher.match(descriptors_1, descriptors_2, matches);
		drawMatches(img_1, keypoints_1, imgGrayscale, keypoints_2, matches, img_matches);//put the outcome of matches into img_matches

		cv::namedWindow("imgOriginal", CV_WINDOW_NORMAL);      
		cv::namedWindow("object", CV_WINDOW_NORMAL);     
		cv::imshow("imgOriginal", img_keypoints_2);     
		cv::imshow("object", img_keypoints_1);                   
		cv::imshow("surf_Matches", img_matches);
		charCheckForEscKey = cv::waitKey(1);        // delay (in ms) and get key press, if any
	} 

	return(0);
}