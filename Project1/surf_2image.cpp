// compare two images and match their keypoints, simply just use surf, it doesnt work so well

#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

void readme();

/** @function main */
int main(int argc, char** argv)
{

	Mat img1 = imread("book.jpg", IMREAD_GRAYSCALE);
	Mat img2 = imread("bookontable.jpg", IMREAD_GRAYSCALE);
	Mat img_1, img_2;
	cv::resize(img1, img_1, Size(), 0.2, 0.2);
	cv::resize(img2, img_2, Size(), 0.2, 0.2);

	if (!img_1.data || !img_2.data)
	{
		std::cout << " --(!) Error reading images " << std::endl; return -2;
	}

	std::cout << "images loaded" << std::endl;


	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;

	Ptr<SURF> detector = SURF::create(minHessian);

	std::vector<KeyPoint> keypoints_1, keypoints_2;

	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);

	//-- Step 2 : Draw keypoints
	Mat img_keypoints_1; Mat img_keypoints_2;

	drawKeypoints(img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	Ptr<SURF> extractor = SURF::create();
	Mat descriptors_1, descriptors_2;

	extractor->compute(img_1, keypoints_1, descriptors_1);
	extractor->compute(img_2, keypoints_2, descriptors_2);

	//-- Step 3: Matching descriptor vectors with a brute force matcher
	BFMatcher matcher(NORM_L2);
	std::vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
	imshow("surf_Matches", img_matches);
	waitKey(0);

	return 0;
}

/** @function readme */
void readme()
{
	std::cout << " Usage: ./SURF_detector <img1> <img2>" << std::endl;
}