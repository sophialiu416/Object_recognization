//recognize the book from the webcam, but slow (combine with cam_object and feature_homography)
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

///////////////////////////////////////////////////////////////////////////////////////////////////
int main() {
	Mat img1 = imread("book.jpg", IMREAD_GRAYSCALE);
	Mat img_object, img_scene;
	cv::resize(img1, img_object, Size(), 0.25, 0.25);
	cv::namedWindow("tracedObject", CV_WINDOW_NORMAL);  
	cv::imshow("tracedObject", img_object);           
	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;
	Ptr<SURF> detector = SURF::create(minHessian);
	std::vector<KeyPoint> keypoints_object, keypoints_scene;
	detector->detect(img_object, keypoints_object);

	//-- Step 2: Calculate descriptors (feature vectors)
	Ptr<SURF> extractor = SURF::create();
	Mat descriptors_object, descriptors_scene;
	extractor->compute(img_object, keypoints_object, descriptors_object);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	cv::VideoCapture capWebcam(0);       

	if (capWebcam.isOpened() == false) {                          
		std::cout << "error: capWebcam not accessed successfully\n\n";                                                          
		return(0);                                  
	}

	cv::Mat imgOriginal;    
	cv::Mat imgGrayscale;    
	char charCheckForEscKey = 0;
	cout << "Please press Esc key to stop" << endl;
	while (charCheckForEscKey != 27 && capWebcam.isOpened()) {          
		bool blnFrameReadSuccessfully = capWebcam.read(imgOriginal);            // get next frame
		if (!blnFrameReadSuccessfully || imgOriginal.empty()) {            
			std::cout << "error: frame not read from webcam\n";       
			break;                                                          
		}
		cv::cvtColor(imgOriginal, img_scene, CV_BGR2GRAY);                 
		detector->detect(img_scene, keypoints_scene);
		extractor->compute(img_scene, keypoints_scene, descriptors_scene);
		matcher.match(descriptors_object, descriptors_scene, matches);
		double max_dist = 0; double min_dist = 100;
		//-- Quick calculation of max and min distances between keypoints
		for (int i = 0; i < descriptors_object.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}
//		printf("-- Max dist : %f \n", max_dist);
//		printf("-- Min dist : %f \n", min_dist);

		//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
		std::vector< DMatch > good_matches;
		for (int i = 0; i < descriptors_object.rows; i++)
		{
			if (matches[i].distance < 3 * min_dist)
			{
				good_matches.push_back(matches[i]);
			}
		}

		Mat img_matches;
		drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		//-- Localize the object
		std::vector<Point2f> obj;
		std::vector<Point2f> scene;
		for (int i = 0; i < good_matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
		}
		Mat H = findHomography(obj, scene, CV_RANSAC);

		//-- Get the corners from the image_1 ( the object to be "detected" )
		std::vector<Point2f> obj_corners(4);
		obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img_object.cols, 0);
		obj_corners[2] = cvPoint(img_object.cols, img_object.rows); obj_corners[3] = cvPoint(0, img_object.rows);
		std::vector<Point2f> scene_corners(4);
		perspectiveTransform(obj_corners, scene_corners, H);

		//-- Draw lines between the corners (the mapped object in the scene - image_2 )
		line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);

		//-- Show detected matches
		imshow("Good Matches & Object detection", img_matches);
		cv::namedWindow("imgOriginal", CV_WINDOW_NORMAL);   
		cv::imshow("imgOriginal", img_scene);  
		charCheckForEscKey = cv::waitKey(1);  
		} 
    return(0);
}









