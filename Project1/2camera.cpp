// find keypoints of image and webcam and try to match them

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iomanip>
#include <stdio.h>
#include<iostream>
#include<conio.h>           // may have to modify this line if not using Windows
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

///////////////////////////////////////////////////////////////////////////////////////////////////
int main() {

	cv::VideoCapture capWebcam_left(0);            // declare a VideoCapture object and associate to webcam, 0 => use 1st webcam
	cv::VideoCapture capWebcam_right(1);            // declare a VideoCapture object and associate to webcam, 1 => use 2st webcam

	if (capWebcam_left.isOpened() == false) {                                // check if VideoCapture object was associated to webcam successfully
		std::cout << "error: capWebcam not accessed successfully\n\n";      // if not, print error message to std out
		_getch();                                                           // may have to modify this line if not using Windows
		return(0);                                                          // and exit program
	}

	if (capWebcam_right.isOpened() == false) {                               
		std::cout << "error: capWebcam not accessed successfully\n\n";     
		_getch();                                                         
		return(0);                                                         
	}

	cv::Mat img_left;      
	cv::Mat img_right;      

	char charCheckForEscKey = 0;
	char choice = 'z';
	int count = 0;

	while ( capWebcam_left.isOpened()&& capWebcam_right.isOpened()) {            // until the Esc key is pressed or webcam connection is lost
		bool blnFrameReadSuccessfully_left = capWebcam_left.read(img_left);            // get next frame

		if (!blnFrameReadSuccessfully_left || img_left.empty()) {                 // if frame not read successfully
			std::cout << "error: frame not read from webcam\n";                 // print error message to std out
			break;                                                              // and jump out of while loop
		}

		bool blnFrameReadSuccessfully_right = capWebcam_right.read(img_right);      

		if (!blnFrameReadSuccessfully_right || img_right.empty()) {          
			std::cout << "error: frame not read from webcam\n";          
			break;                                                     
		}


		if (choice == 'c') {
			stringstream l_name, r_name;
			l_name << "left_" << setw(2) << setfill('0') << count << ".jpg";
			r_name << "right_" << setw(2) << setfill('0') << count << ".jpg";
			imwrite(l_name.str(), img_left);
			imwrite(r_name.str(), img_right);
			cout << "Saved set " << count << endl;
			count++;
		}

		cv::namedWindow("camera_left", CV_WINDOW_NORMAL);   
		cv::imshow("camera_left", img_left);                
		cv::namedWindow("camera_right", CV_WINDOW_NORMAL);        
		cv::imshow("camera_right", img_right);                   
		choice = char(waitKey(1));
//		charCheckForEscKey = cv::waitKey(1);        // delay (in ms) and get key press, if any
	}   // end while


	return(0);
}