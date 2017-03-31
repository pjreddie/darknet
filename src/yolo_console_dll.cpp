#include <iostream>
#include <iomanip> 
#include <string>
#include <vector>
#include <fstream>

#define OPENCV

#include "yolo_v2_class.hpp"	// imported functions from DLL


#ifdef OPENCV
#include <opencv2/opencv.hpp>			// C++
#pragma comment(lib, "opencv_core249.lib")
#pragma comment(lib, "opencv_imgproc249.lib")
#pragma comment(lib, "opencv_highgui249.lib")
void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names, unsigned int wait_msec = 0) {
	for (auto &i : result_vec) {
		cv::Scalar color(60, 160, 260);
		cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 3);
		if(obj_names.size() > i.obj_id)
			putText(mat_img, obj_names[i.obj_id], cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
		if(i.track_id > 0)
			putText(mat_img, std::to_string(i.track_id), cv::Point2f(i.x+5, i.y + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, color);
	}
	cv::imshow("window name", mat_img);
	cv::waitKey(wait_msec);
}
#endif	// OPENCV


void show_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names) {
	for (auto &i : result_vec) {
		if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
		std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y 
			<< ", w = " << i.w << ", h = " << i.h
			<< std::setprecision(3) << ", prob = " << i.prob << std::endl;
	}
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
	std::ifstream file(filename);
	std::vector<std::string> file_lines;
	if (!file.is_open()) return file_lines;
	for(std::string line; file >> line;) file_lines.push_back(line);
	std::cout << "object names loaded \n";
	return file_lines;
}


int main() 
{
	Detector detector("yolo-voc.cfg", "yolo-voc.weights");

	auto obj_names = objects_names_from_file("data/voc.names");

	while (true) 
	{
		std::string filename;
		std::cout << "input image or video filename: ";
		std::cin >> filename;
		if (filename.size() == 0) break;
		
		try {
#ifdef OPENCV
			std::string const file_ext = filename.substr(filename.find_last_of(".") + 1);
			if (file_ext == "avi" || file_ext == "mp4" || file_ext == "mjpg" || file_ext == "mov") {	// video file
				cv::Mat frame;
				detector.nms = 0.02;	// comment it - if track_id is not required
				for(cv::VideoCapture cap(filename); cap >> frame, cap.isOpened();) {
					std::vector<bbox_t> result_vec = detector.detect(frame, 0.2);
					result_vec = detector.tracking(result_vec);	// comment it - if track_id is not required

					draw_boxes(frame, result_vec, obj_names, 3);
					show_result(result_vec, obj_names);
				}
			}
			else {	// image file
				cv::Mat mat_img = cv::imread(filename);
				std::vector<bbox_t> result_vec = detector.detect(mat_img);
				draw_boxes(mat_img, result_vec, obj_names);
				show_result(result_vec, obj_names);
			}
#else
			//std::vector<bbox_t> result_vec = detector.detect(filename);

			auto img = detector.load_image(filename);
			std::vector<bbox_t> result_vec = detector.detect(img);
			detector.free_image(img);
			show_result(result_vec, obj_names);
#endif			
		}
		catch (std::exception &e) { std::cerr << "exception: " << e.what() << "\n"; getchar(); }
		catch (...) { std::cerr << "unknown exception \n"; getchar(); }
	}

	return 0;
}