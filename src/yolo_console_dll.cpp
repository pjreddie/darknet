#include <iostream>
#include <string>
#include <vector>

//#define OPENCV

#include "yolo_v2_class.hpp"	// imported functions from DLL


#ifdef OPENCV
#include <opencv2/opencv.hpp>			// C++
#pragma comment(lib, "opencv_core249.lib")
#pragma comment(lib, "opencv_imgproc249.lib")
#pragma comment(lib, "opencv_highgui249.lib")
void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec) {
	for (auto &i : result_vec) {
		cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), cv::Scalar(50, 200, 50), 3);
	}
	cv::imshow("window name", mat_img);
	cv::waitKey(0);
}
#endif	// OPENCV


void show_result(std::vector<bbox_t> result_vec) {
	for (auto &i : result_vec) {
		std::cout << "obj_id = " << i.obj_id << " - x = " << i.x << ", y = " << i.y 
			<< ", w = " << i.w << ", h = " << i.h
			<< ", prob = " << i.prob << std::endl;
	}
}



int main() 
{
	Detector detector("yolo-voc.cfg", "yolo-voc.weights");

	while (true) 
	{
		std::string filename;
		std::cout << "input image filename: ";
		std::cin >> filename;
		if (filename.size() == 0) break;

#ifdef OPENCV
		cv::Mat mat_img = cv::imread(filename);
		std::vector<bbox_t> result_vec = detector.detect(mat_img);
		draw_boxes(mat_img, result_vec);
#else
		std::vector<bbox_t> result_vec = detector.detect(filename);
#endif
		show_result(result_vec);
	}

	return 0;
}