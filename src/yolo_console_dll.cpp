#include <iostream>
#include <iomanip> 
#include <string>
#include <vector>
#include <queue>
#include <fstream>
#include <thread>
#include <atomic>
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable

#ifdef _WIN32
#define OPENCV
#define GPU
#endif

// To use tracking - uncomment the following line. Tracking is supported only by OpenCV 3.x
//#define TRACK_OPTFLOW

#include "yolo_v2_class.hpp"	// imported functions from DLL

#ifdef OPENCV
#include <opencv2/opencv.hpp>			// C++
#include "opencv2/core/version.hpp"
#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio.hpp"
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)""CVAUX_STR(CV_VERSION_MINOR)""CVAUX_STR(CV_VERSION_REVISION)
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
#ifdef TRACK_OPTFLOW
#pragma comment(lib, "opencv_cudaoptflow" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_cudaimgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#endif	// TRACK_OPTFLOW
#else
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)""CVAUX_STR(CV_VERSION_MAJOR)""CVAUX_STR(CV_VERSION_MINOR)
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#endif	// CV_VERSION_EPOCH

class track_kalman {
public:
	cv::KalmanFilter kf;
	int state_size, meas_size, contr_size;


	track_kalman(int _state_size = 10, int _meas_size = 10, int _contr_size = 0)
		: state_size(_state_size), meas_size(_meas_size), contr_size(_contr_size)
	{
		kf.init(state_size, meas_size, contr_size, CV_32F);

		cv::setIdentity(kf.measurementMatrix);
		cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
		cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-5));
		cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1e-2));
		cv::setIdentity(kf.transitionMatrix);
	}

	void set(std::vector<bbox_t> result_vec) {
		for (size_t i = 0; i < result_vec.size() && i < state_size*2; ++i) {
			kf.statePost.at<float>(i * 2 + 0) = result_vec[i].x;
			kf.statePost.at<float>(i * 2 + 1) = result_vec[i].y;
		}
	}

	// Kalman.correct() calculates: statePost = statePre + gain * (z(k)-measurementMatrix*statePre);
	// corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
	std::vector<bbox_t> correct(std::vector<bbox_t> result_vec) {
		cv::Mat measurement(meas_size, 1, CV_32F);
		for (size_t i = 0; i < result_vec.size() && i < meas_size * 2; ++i) {
			measurement.at<float>(i * 2 + 0) = result_vec[i].x;
			measurement.at<float>(i * 2 + 1) = result_vec[i].y;
		}
		cv::Mat estimated = kf.correct(measurement);
		for (size_t i = 0; i < result_vec.size() && i < meas_size * 2; ++i) {
			result_vec[i].x = estimated.at<float>(i * 2 + 0);
			result_vec[i].y = estimated.at<float>(i * 2 + 1);
		}
		return result_vec;
	}

	// Kalman.predict() calculates: statePre = TransitionMatrix * statePost;
	// predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
	std::vector<bbox_t> predict() {
		std::vector<bbox_t> result_vec;
		cv::Mat control;
		cv::Mat prediction = kf.predict(control);
		for (size_t i = 0; i < prediction.rows && i < state_size * 2; ++i) {
			result_vec[i].x = prediction.at<float>(i * 2 + 0);
			result_vec[i].y = prediction.at<float>(i * 2 + 1);
		}
		return result_vec;
	}

};




class extrapolate_coords_t {
public:
	std::vector<bbox_t> old_result_vec;
	std::vector<float> dx_vec, dy_vec, time_vec;
	std::vector<float> old_dx_vec, old_dy_vec;

	void new_result(std::vector<bbox_t> new_result_vec, float new_time) {
		old_dx_vec = dx_vec;
		old_dy_vec = dy_vec;
		if (old_dx_vec.size() != old_result_vec.size()) std::cout << "old_dx != old_res \n";
		dx_vec = std::vector<float>(new_result_vec.size(), 0);
		dy_vec = std::vector<float>(new_result_vec.size(), 0);
		update_result(new_result_vec, new_time, false);
		old_result_vec = new_result_vec;
		time_vec = std::vector<float>(new_result_vec.size(), new_time);
	}

	void update_result(std::vector<bbox_t> new_result_vec, float new_time, bool update = true) {
		for (size_t i = 0; i < new_result_vec.size(); ++i) {
			for (size_t k = 0; k < old_result_vec.size(); ++k) {
				if (old_result_vec[k].track_id == new_result_vec[i].track_id && old_result_vec[k].obj_id == new_result_vec[i].obj_id) {
					float const delta_time = new_time - time_vec[k];
					if (abs(delta_time) < 1) break;
					size_t index = (update) ? k : i;
					float dx = ((float)new_result_vec[i].x - (float)old_result_vec[k].x) / delta_time;
					float dy = ((float)new_result_vec[i].y - (float)old_result_vec[k].y) / delta_time;
					float old_dx = dx, old_dy = dy;

					// if it's shaking
					if (update) {
						if (dx * dx_vec[i] < 0) dx = dx / 2;
						if (dy * dy_vec[i] < 0) dy = dy / 2;
					} else {
						if (dx * old_dx_vec[k] < 0) dx = dx / 2;
						if (dy * old_dy_vec[k] < 0) dy = dy / 2;
					}
					dx_vec[index] = dx;
					dy_vec[index] = dy;

					//if (old_dx == dx && old_dy == dy) std::cout << "not shakin \n";
					//else std::cout << "shakin \n";

					if (dx_vec[index] > 1000 || dy_vec[index] > 1000) {
						//std::cout << "!!! bad dx or dy, dx = " << dx_vec[index] << ", dy = " << dy_vec[index] << 
						//	", delta_time = " << delta_time << ", update = " << update << std::endl;
						dx_vec[index] = 0;
						dy_vec[index] = 0;						
					}
					old_result_vec[k].x = new_result_vec[i].x;
					old_result_vec[k].y = new_result_vec[i].y;
					time_vec[k] = new_time;
					break;
				}
			}
		}
	}

	std::vector<bbox_t> predict(float cur_time) {
		std::vector<bbox_t> result_vec = old_result_vec;
		for (size_t i = 0; i < old_result_vec.size(); ++i) {
			float const delta_time = cur_time - time_vec[i];
			auto &bbox = result_vec[i];
			float new_x = (float) bbox.x + dx_vec[i] * delta_time;
			float new_y = (float) bbox.y + dy_vec[i] * delta_time;
			if (new_x > 0) bbox.x = new_x;
			else bbox.x = 0;
			if (new_y > 0) bbox.y = new_y;
			else bbox.y = 0;
		}
		return result_vec;
	}

};


void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names, 
	int current_det_fps = -1, int current_cap_fps = -1)
{
	int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

	for (auto &i : result_vec) {
		cv::Scalar color = obj_id_to_color(i.obj_id);
		cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);
		if (obj_names.size() > i.obj_id) {
			std::string obj_name = obj_names[i.obj_id];
			if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
			cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
			int const max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
			cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 30, 0)), 
				cv::Point2f(std::min((int)i.x + max_width, mat_img.cols-1), std::min((int)i.y, mat_img.rows-1)), 
				color, CV_FILLED, 8, 0);
			putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
		}
	}
	if (current_det_fps >= 0 && current_cap_fps >= 0) {
		std::string fps_str = "FPS detection: " + std::to_string(current_det_fps) + "   FPS capture: " + std::to_string(current_cap_fps);
		putText(mat_img, fps_str, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(50, 255, 0), 2);
	}
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
	std::string  names_file = "data/voc.names";
	std::string  cfg_file = "cfg/yolo-voc.cfg";
	std::string  weights_file = "yolo-voc.weights";
	std::string filename;

	if (argc > 4) {	//voc.names yolo-voc.cfg yolo-voc.weights test.mp4		
		names_file = argv[1];
		cfg_file = argv[2];
		weights_file = argv[3];
		filename = argv[4];
	}
	else if (argc > 1) filename = argv[1];

	float const thresh = (argc > 5) ? std::stof(argv[5]) : 0.20;

	Detector detector(cfg_file, weights_file);

	auto obj_names = objects_names_from_file(names_file);
	std::string out_videofile = "result.avi";
	bool const save_output_videofile = true;
#ifdef TRACK_OPTFLOW
	Tracker_optflow tracker_flow;
	detector.wait_stream = true;
#endif

	while (true) 
	{		
		std::cout << "input image or video filename: ";
		if(filename.size() == 0) std::cin >> filename;
		if (filename.size() == 0) break;
		
		try {
#ifdef OPENCV
			extrapolate_coords_t extrapolate_coords;
			bool extrapolate_flag = false;
			float cur_time_extrapolate = 0, old_time_extrapolate = 0;
			preview_boxes_t large_preview(100, 150, false), small_preview(50, 50, true);
			bool show_small_boxes = false;

			std::string const file_ext = filename.substr(filename.find_last_of(".") + 1);
			std::string const protocol = filename.substr(0, 7);
			if (file_ext == "avi" || file_ext == "mp4" || file_ext == "mjpg" || file_ext == "mov" || 	// video file
				protocol == "rtmp://" || protocol == "rtsp://" || protocol == "http://" || protocol == "https:/")	// video network stream
			{
				cv::Mat cap_frame, cur_frame, det_frame, write_frame;
				std::queue<cv::Mat> track_optflow_queue;
				int passed_flow_frames = 0;
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
				std::condition_variable cv_detected, cv_pre_tracked;
				std::chrono::steady_clock::time_point steady_start, steady_end;
				cv::VideoCapture cap(filename); cap >> cur_frame;
				int const video_fps = cap.get(CV_CAP_PROP_FPS);
				cv::Size const frame_size = cur_frame.size();
				cv::VideoWriter output_video;
				if (save_output_videofile)
					output_video.open(out_videofile, CV_FOURCC('D', 'I', 'V', 'X'), std::max(35, video_fps), frame_size, true);

				while (!cur_frame.empty()) 
				{
					// always sync
					if (t_cap.joinable()) {
						t_cap.join();
						++fps_cap_counter;
						cur_frame = cap_frame.clone();
					}
					t_cap = std::thread([&]() { cap >> cap_frame; });
					++cur_time_extrapolate;

					// swap result bouned-boxes and input-frame
					if(consumed)
					{
						std::unique_lock<std::mutex> lock(mtx);
						det_image = detector.mat_to_image_resize(cur_frame);
						auto old_result_vec = detector.tracking_id(result_vec);
						auto detected_result_vec = thread_result_vec;
						result_vec = detected_result_vec;
#ifdef TRACK_OPTFLOW
						// track optical flow
						if (track_optflow_queue.size() > 0) {
							//std::cout << "\n !!!! all = " << track_optflow_queue.size() << ", cur = " << passed_flow_frames << std::endl;
							cv::Mat first_frame = track_optflow_queue.front();
							tracker_flow.update_tracking_flow(track_optflow_queue.front(), result_vec);

							while (track_optflow_queue.size() > 1) {
								track_optflow_queue.pop();
								result_vec = tracker_flow.tracking_flow(track_optflow_queue.front(), true);
							}
							track_optflow_queue.pop();
							passed_flow_frames = 0;

							result_vec = detector.tracking_id(result_vec);
							auto tmp_result_vec = detector.tracking_id(detected_result_vec, false);
							small_preview.set(first_frame, tmp_result_vec);

							extrapolate_coords.new_result(tmp_result_vec, old_time_extrapolate);
							old_time_extrapolate = cur_time_extrapolate;
							extrapolate_coords.update_result(result_vec, cur_time_extrapolate - 1);
						}
#else
						result_vec = detector.tracking_id(result_vec);	// comment it - if track_id is not required					
						extrapolate_coords.new_result(result_vec, cur_time_extrapolate - 1);
#endif
						// add old tracked objects
						for (auto &i : old_result_vec) {
							auto it = std::find_if(result_vec.begin(), result_vec.end(),
								[&i](bbox_t const& b) { return b.track_id == i.track_id && b.obj_id == i.obj_id; });
							bool track_id_absent = (it == result_vec.end());
							if (track_id_absent) {
								if (i.frames_counter-- > 1)
									result_vec.push_back(i);
							}
							else {
								it->frames_counter = std::min((unsigned)3, i.frames_counter + 1);
							}
						}
#ifdef TRACK_OPTFLOW
						tracker_flow.update_cur_bbox_vec(result_vec);
						result_vec = tracker_flow.tracking_flow(cur_frame, true);	// track optical flow
#endif
						consumed = false;
						cv_pre_tracked.notify_all();
					}
					// launch thread once - Detection
					if (!t_detect.joinable()) {
						t_detect = std::thread([&]() {
							auto current_image = det_image;
							consumed = true;
							while (current_image.use_count() > 0) {
								auto result = detector.detect_resized(*current_image, frame_size.width, frame_size.height, 
									thresh, false);	// true
								++fps_det_counter;
								std::unique_lock<std::mutex> lock(mtx);
								thread_result_vec = result;
								consumed = true;
								cv_detected.notify_all();
								if (detector.wait_stream) {
									while (consumed) cv_pre_tracked.wait(lock);
								}
								current_image = det_image;
							}
						});
					}
					//while (!consumed);	// sync detection

					if (!cur_frame.empty()) {
						steady_end = std::chrono::steady_clock::now();
						if (std::chrono::duration<double>(steady_end - steady_start).count() >= 1) {
							current_det_fps = fps_det_counter;
							current_cap_fps = fps_cap_counter;
							steady_start = steady_end;
							fps_det_counter = 0;
							fps_cap_counter = 0;
						}

						large_preview.set(cur_frame, result_vec);
#ifdef TRACK_OPTFLOW
						++passed_flow_frames;
						track_optflow_queue.push(cur_frame.clone());
						result_vec = tracker_flow.tracking_flow(cur_frame);	// track optical flow
						extrapolate_coords.update_result(result_vec, cur_time_extrapolate);
						small_preview.draw(cur_frame, show_small_boxes);
#endif						
						auto result_vec_draw = result_vec;
						if (extrapolate_flag) {
							result_vec_draw = extrapolate_coords.predict(cur_time_extrapolate);
							cv::putText(cur_frame, "extrapolate", cv::Point2f(10, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(50, 50, 0), 2);
						}
						draw_boxes(cur_frame, result_vec_draw, obj_names, current_det_fps, current_cap_fps);
						//show_console_result(result_vec, obj_names);
						large_preview.draw(cur_frame);

						cv::imshow("window name", cur_frame);
						int key = cv::waitKey(3);	// 3 or 16ms
						if (key == 'f') show_small_boxes = !show_small_boxes;
						if (key == 'p') while (true) if(cv::waitKey(100) == 'p') break;
						if (key == 'e') extrapolate_flag = !extrapolate_flag;

						if (output_video.isOpened() && videowrite_ready) {
							if (t_videowrite.joinable()) t_videowrite.join();
							write_frame = cur_frame.clone();
							videowrite_ready = false;
							t_videowrite = std::thread([&]() { 
								 output_video << write_frame; videowrite_ready = true;
							});
						}
					}

#ifndef TRACK_OPTFLOW
					// wait detection result for video-file only (not for net-cam)
					if (protocol != "rtsp://" && protocol != "http://" && protocol != "https:/") {
						std::unique_lock<std::mutex> lock(mtx);
						while (!consumed) cv_detected.wait(lock);
					}
#endif
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
				result_vec = detector.tracking_id(result_vec);	// comment it - if track_id is not required
				draw_boxes(mat_img, result_vec, obj_names);
				cv::imshow("window name", mat_img);
				cv::waitKey(3);	// 3 or 16ms
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