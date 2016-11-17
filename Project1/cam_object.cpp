// find keypoints of image and webcam and try to match them

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <stdio.h>
#include<iostream>
#include<conio.h>           // may have to modify this line if not using Windows
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

///////////////////////////////////////////////////////////////////////////////////////////////////
int main() {

	Mat img1 = imread("HP1.jpg", IMREAD_GRAYSCALE);
	Mat img_1;
	cv::resize(img1, img_1, Size(), 0.5, 0.5);
	if (!img_1.data)
	{
		std::cout << " --(!) Error reading images " << std::endl; return -2;
	}

	std::cout << "images loaded" << std::endl;

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;

	Ptr<SURF> detector = SURF::create(minHessian);

	std::vector<KeyPoint> keypoints_1, keypoints_2;

	detector->detect(img_1, keypoints_1);

	//-- Draw keypoints
	Mat img_keypoints_1; Mat img_keypoints_2;

	drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	std::cout << "keypoint founded" << std::endl;

	Ptr<SURF> extractor = SURF::create();
	//	SurfDescriptorExtractor extractor;
	Mat descriptors_1, descriptors_2;//存放特征向量的矩阵

	extractor->compute(img_1, keypoints_1, descriptors_1);

	//-- Step 3: Matching descriptor vectors with a brute force matcher
	BFMatcher matcher(NORM_L2);
	std::vector< DMatch > matches;

	//绘制匹配线段
	Mat img_matches;

	cv::VideoCapture capWebcam(0);            // declare a VideoCapture object and associate to webcam, 0 => use 1st webcam

	if (capWebcam.isOpened() == false) {                                // check if VideoCapture object was associated to webcam successfully
		std::cout << "error: capWebcam not accessed successfully\n\n";      // if not, print error message to std out
		_getch();                                                           // may have to modify this line if not using Windows
		return(0);                                                          // and exit program
	}

	cv::Mat imgOriginal;        // input image
	cv::Mat imgGrayscale;       // grayscale of input image
	cv::Mat imgBlurred;         // intermediate blured image
	cv::Mat imgCanny;           // Canny edge image

	char charCheckForEscKey = 0;

	while (charCheckForEscKey != 27 && capWebcam.isOpened()) {            // until the Esc key is pressed or webcam connection is lost
		bool blnFrameReadSuccessfully = capWebcam.read(imgOriginal);            // get next frame

		if (!blnFrameReadSuccessfully || imgOriginal.empty()) {                 // if frame not read successfully
			std::cout << "error: frame not read from webcam\n";                 // print error message to std out
			break;                                                              // and jump out of while loop
		}

		cv::cvtColor(imgOriginal, imgGrayscale, CV_BGR2GRAY);                   // convert to grayscale

		detector->detect(imgGrayscale, keypoints_2);

		drawKeypoints(imgGrayscale, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

		extractor->compute(imgGrayscale, keypoints_2, descriptors_2);

		matcher.match(descriptors_1, descriptors_2, matches);

		drawMatches(img_1, keypoints_1, imgGrayscale, keypoints_2, matches, img_matches);//将匹配出来的结果放入内存img_matches中

		// declare windows
		cv::namedWindow("imgOriginal", CV_WINDOW_NORMAL);       // note: you can use CV_WINDOW_NORMAL which allows resizing the window
		cv::namedWindow("object", CV_WINDOW_NORMAL);          // or CV_WINDOW_AUTOSIZE for a fixed size window matching the resolution of the image
																// CV_WINDOW_AUTOSIZE is the default
		cv::imshow("imgOriginal", img_keypoints_2);                 // show windows
		cv::imshow("object", img_keypoints_1);                       //
		cv::imshow("surf_Matches", img_matches);//显示的标题为Matches
		charCheckForEscKey = cv::waitKey(1);        // delay (in ms) and get key press, if any
	}   // end while

	return(0);
}