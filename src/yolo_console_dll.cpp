#include <iostream>
#include <iomanip> 
#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable

#ifdef _WIN32
#define OPENCV
#endif

#include "yolo_v2_class.hpp"	// imported functions from DLL

#ifdef OPENCV
#include <opencv2/opencv.hpp>			// C++
#include "opencv2/core/version.hpp"
#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio.hpp"
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)""CVAUX_STR(CV_VERSION_MINOR)""CVAUX_STR(CV_VERSION_REVISION)
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
#else
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)""CVAUX_STR(CV_VERSION_MAJOR)""CVAUX_STR(CV_VERSION_MINOR)
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#endif


void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names, 
	unsigned int wait_msec = 0, int current_det_fps = -1, int current_cap_fps = -1)
{
	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

	for (auto &i : result_vec) {
		int const offset = i.obj_id * 123457 % 6;
		int const color_scale = 150 + (i.obj_id * 123457) % 100;
		cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
		color *= color_scale;
		cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 5);
		if (obj_names.size() > i.obj_id) {
			std::string obj_name = obj_names[i.obj_id];
			if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
			cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
			int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 3, 0), std::max((int)i.y - 30, 0)), 
				cv::Point2f(std::min((int)i.x + max_width, mat_img.cols-1), std::min((int)i.y, mat_img.rows-1)), 
				color, CV_FILLED, 8, 0);
			putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
		}
	}
	if (current_det_fps >= 0 && current_cap_fps >= 0) {
		std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
		putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
	}
	cv::imshow("window name", mat_img);
	cv::waitKey(wait_msec);
}
#endif	// OPENCV


void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names) {
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
	for(std::string line; getline(file, line);) file_lines.push_back(line);
	std::cout << "object names loaded \n";
	return file_lines;
}


int main(int argc, char *argv[])
{
	std::string filename;
	if (argc > 1) filename = argv[1];

	Detector detector("cfg/yolo-voc.cfg", "yolo-voc.weights");

	auto obj_names = objects_names_from_file("data/voc.names");
	std::string out_videofile = "result.avi";
	bool const save_output_videofile = false;

	while (true) 
	{		
		std::cout << "input image or video filename: ";
		if(filename.size() == 0) std::cin >> filename;
		if (filename.size() == 0) break;
		
		try {
#ifdef OPENCV
			std::string const file_ext = filename.substr(filename.find_last_of(".") + 1);
			std::string const protocol = filename.substr(0, 7);
			if (file_ext == "avi" || file_ext == "mp4" || file_ext == "mjpg" || file_ext == "mov" || 	// video file
				protocol == "rtmp://" || protocol == "rtsp://" || protocol == "http://" || protocol == "https:/")	// video network stream
			{
				cv::Mat cap_frame, cur_frame, det_frame, write_frame;
				std::shared_ptr<image_t> det_image;
				std::vector<bbox_t> result_vec, thread_result_vec;
				detector.nms = 0.02;	// comment it - if track_id is not required
				std::atomic<bool> consumed, videowrite_ready;
				consumed = true;
				videowrite_ready = true;
				std::atomic<int> fps_det_counter, fps_cap_counter;
				fps_det_counter = 0;
				fps_cap_counter = 0;
				int current_det_fps = 0, current_cap_fps = 0;
				std::thread t_detect, t_cap, t_videowrite;
				std::mutex mtx;
				std::condition_variable cv;
				std::chrono::steady_clock::time_point steady_start, steady_end;
				cv::VideoCapture cap(filename); cap >> cur_frame;
				int const video_fps = cap.get(CV_CAP_PROP_FPS);
				cv::Size const frame_size = cur_frame.size();
				cv::VideoWriter output_video;
				if (save_output_videofile)
					output_video.open(out_videofile, CV_FOURCC('D', 'I', 'V', 'X'), std::max(35, video_fps), frame_size, true);

				while (!cur_frame.empty()) {
					if (t_cap.joinable()) {
						t_cap.join();
						++fps_cap_counter;
						cur_frame = cap_frame.clone();
					}
					t_cap = std::thread([&]() { cap >> cap_frame; });

					// swap result and input-frame
					if(consumed)
					{
						std::unique_lock<std::mutex> lock(mtx);
						det_image = detector.mat_to_image_resize(cur_frame);
						result_vec = thread_result_vec;
						result_vec = detector.tracking(result_vec);	// comment it - if track_id is not required
						consumed = false;
					}
					// launch thread once
					if (!t_detect.joinable()) {
						t_detect = std::thread([&]() {
							auto current_image = det_image;
							consumed = true;
							while (current_image.use_count() > 0) {
								auto result = detector.detect_resized(*current_image, frame_size, 0.24, true);
								++fps_det_counter;
								std::unique_lock<std::mutex> lock(mtx);
								thread_result_vec = result;
								current_image = det_image;
								consumed = true;
								cv.notify_all();
							}
						});
					}

					if (!cur_frame.empty()) {
						steady_end = std::chrono::steady_clock::now();
						if (std::chrono::duration<double>(steady_end - steady_start).count() >= 1) {
							current_det_fps = fps_det_counter;
							current_cap_fps = fps_cap_counter;
							steady_start = steady_end;
							fps_det_counter = 0;
							fps_cap_counter = 0;
						}
						draw_boxes(cur_frame, result_vec, obj_names, 3, current_det_fps, current_cap_fps);
						//show_console_result(result_vec, obj_names);

						if (output_video.isOpened() && videowrite_ready) {
							if (t_videowrite.joinable()) t_videowrite.join();
							write_frame = cur_frame.clone();
							videowrite_ready = false;
							t_videowrite = std::thread([&]() { 
								 output_video << write_frame; videowrite_ready = true;
							});
						}
					}

					// wait detection result for video-file only (not for net-cam)
					if (protocol != "rtsp://" && protocol != "http://" && protocol != "https:/") {
						std::unique_lock<std::mutex> lock(mtx);
						while (!consumed) cv.wait(lock);
					}
				}
				if (t_cap.joinable()) t_cap.join();
				if (t_detect.joinable()) t_detect.join();
				if (t_videowrite.joinable()) t_videowrite.join();
				std::cout << "Video ended \n";
			}
			else if (file_ext == "txt") {	// list of image files
				std::ifstream file(filename);
				if (!file.is_open()) std::cout << "File not found! \n";
				else 
					for (std::string line; file >> line;) {
						std::cout << line << std::endl;
						cv::Mat mat_img = cv::imread(line);
						std::vector<bbox_t> result_vec = detector.detect(mat_img);
						show_console_result(result_vec, obj_names);
						//draw_boxes(mat_img, result_vec, obj_names);
						//cv::imwrite("res_" + line, mat_img);
					}
				
			}
			else {	// image file
				cv::Mat mat_img = cv::imread(filename);
				std::vector<bbox_t> result_vec = detector.detect(mat_img);
				result_vec = detector.tracking(result_vec);	// comment it - if track_id is not required
				draw_boxes(mat_img, result_vec, obj_names);
				show_console_result(result_vec, obj_names);
			}
#else
			//std::vector<bbox_t> result_vec = detector.detect(filename);

			auto img = detector.load_image(filename);
			std::vector<bbox_t> result_vec = detector.detect(img);
			detector.free_image(img);
			show_console_result(result_vec, obj_names);
#endif			
		}
		catch (std::exception &e) { std::cerr << "exception: " << e.what() << "\n"; getchar(); }
		catch (...) { std::cerr << "unknown exception \n"; getchar(); }
		filename.clear();
	}

	return 0;
}