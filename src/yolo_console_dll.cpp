#include <iostream>
#include <string>
#include <vector>
#include <fstream>

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

void show_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names) {
	for (auto &i : result_vec) {
		if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
		std::cout << "obj_id = " << i.obj_id << " - x = " << i.x << ", y = " << i.y 
			<< ", w = " << i.w << ", h = " << i.h
			<< ", prob = " << i.prob << std::endl;
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
		std::cout << "input image filename: ";
		std::cin >> filename;
		if (filename.size() == 0) break;
		
		try {
#ifdef OPENCV
			cv::Mat mat_img = cv::imread(filename);
			std::vector<bbox_t> result_vec = detector.detect(mat_img);
			draw_boxes(mat_img, result_vec);
#else
			//std::vector<bbox_t> result_vec = detector.detect(filename);

			auto img = detector.load_image(filename);
			std::vector<bbox_t> result_vec = detector.detect(img);
			detector.free_image(img);
#endif
			show_result(result_vec, obj_names);
		}
		catch (std::exception &e) { std::cerr << "exception: " << e.what() << "\n"; getchar(); }
		catch (...) { std::cerr << "unknown exception \n"; getchar(); }
	}

	return 0;
}