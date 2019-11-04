#ifndef YOLO_V2_CLASS_HPP
#define YOLO_V2_CLASS_HPP

#ifndef LIB_API
#ifdef LIB_EXPORTS
#if defined(_MSC_VER)
#define LIB_API __declspec(dllexport)
#else
#define LIB_API __attribute__((visibility("default")))
#endif
#else
#if defined(_MSC_VER)
#define LIB_API
#else
#define LIB_API
#endif
#endif
#endif

#define C_SHARP_MAX_OBJECTS 1000

struct bbox_t {
    unsigned int x, y, w, h;       // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;                    // confidence - probability that the object was found correctly
    unsigned int obj_id;           // class of object - from range [0, classes-1]
    unsigned int track_id;         // tracking id for video (0 - untracked, 1 - inf - tracked object)
    unsigned int frames_counter;   // counter of frames on which the object was detected
    float x_3d, y_3d, z_3d;        // center of object (in Meters) if ZED 3D Camera is used
};

struct image_t {
    int h;                        // height
    int w;                        // width
    int c;                        // number of chanels (3 - for RGB)
    float *data;                  // pointer to the image data
};

struct bbox_t_container {
    bbox_t candidates[C_SHARP_MAX_OBJECTS];
};

#ifdef __cplusplus
#include <memory>
#include <vector>
#include <deque>
#include <algorithm>
#include <chrono>
#include <string>
#include <sstream>
#include <iostream>
#include <cmath>

#ifdef OPENCV
#include <opencv2/opencv.hpp>            // C++
#include <opencv2/highgui/highgui_c.h>   // C
#include <opencv2/imgproc/imgproc_c.h>   // C
#endif

extern "C" LIB_API int init(const char *configurationFilename, const char *weightsFilename, int gpu);
extern "C" LIB_API int detect_image(const char *filename, bbox_t_container &container);
extern "C" LIB_API int detect_mat(const uint8_t* data, const size_t data_length, bbox_t_container &container);
extern "C" LIB_API int dispose();
extern "C" LIB_API int get_device_count();
extern "C" LIB_API int get_device_name(int gpu, char* deviceName);
extern "C" LIB_API bool built_with_cuda();
extern "C" LIB_API bool built_with_cudnn();
extern "C" LIB_API bool built_with_opencv();
extern "C" LIB_API void send_json_custom(char const* send_buf, int port, int timeout);

class Detector {
    std::shared_ptr<void> detector_gpu_ptr;
    std::deque<std::vector<bbox_t>> prev_bbox_vec_deque;
    std::string _cfg_filename, _weight_filename;
public:
    const int cur_gpu_id;
    float nms = .4;
    bool wait_stream;

    LIB_API Detector(std::string cfg_filename, std::string weight_filename, int gpu_id = 0);
    LIB_API ~Detector();

    LIB_API std::vector<bbox_t> detect(std::string image_filename, float thresh = 0.2, bool use_mean = false);
    LIB_API std::vector<bbox_t> detect(image_t img, float thresh = 0.2, bool use_mean = false);
    static LIB_API image_t load_image(std::string image_filename);
    static LIB_API void free_image(image_t m);
    LIB_API int get_net_width() const;
    LIB_API int get_net_height() const;
    LIB_API int get_net_color_depth() const;

    LIB_API std::vector<bbox_t> tracking_id(std::vector<bbox_t> cur_bbox_vec, bool const change_history = true,
                                                int const frames_story = 5, int const max_dist = 40);

    LIB_API void *get_cuda_context();

    //LIB_API bool send_json_http(std::vector<bbox_t> cur_bbox_vec, std::vector<std::string> obj_names, int frame_id,
    //    std::string filename = std::string(), int timeout = 400000, int port = 8070);

    std::vector<bbox_t> detect_resized(image_t img, int init_w, int init_h, float thresh = 0.2, bool use_mean = false)
    {
        if (img.data == NULL)
            throw std::runtime_error("Image is empty");
        auto detection_boxes = detect(img, thresh, use_mean);
        float wk = (float)init_w / img.w, hk = (float)init_h / img.h;
        for (auto &i : detection_boxes) i.x *= wk, i.w *= wk, i.y *= hk, i.h *= hk;
        return detection_boxes;
    }

#ifdef OPENCV
    std::vector<bbox_t> detect(cv::Mat mat, float thresh = 0.2, bool use_mean = false)
    {
        if(mat.data == NULL)
            throw std::runtime_error("Image is empty");
        auto image_ptr = mat_to_image_resize(mat);
        return detect_resized(*image_ptr, mat.cols, mat.rows, thresh, use_mean);
    }

    std::shared_ptr<image_t> mat_to_image_resize(cv::Mat mat) const
    {
        if (mat.data == NULL) return std::shared_ptr<image_t>(NULL);

        cv::Size network_size = cv::Size(get_net_width(), get_net_height());
        cv::Mat det_mat;
        if (mat.size() != network_size)
            cv::resize(mat, det_mat, network_size);
        else
            det_mat = mat;  // only reference is copied

        return mat_to_image(det_mat);
    }

    static std::shared_ptr<image_t> mat_to_image(cv::Mat img_src)
    {
        cv::Mat img;
        if (img_src.channels() == 4) cv::cvtColor(img_src, img, cv::COLOR_RGBA2BGR);
        else if (img_src.channels() == 3) cv::cvtColor(img_src, img, cv::COLOR_RGB2BGR);
        else if (img_src.channels() == 1) cv::cvtColor(img_src, img, cv::COLOR_GRAY2BGR);
        else std::cerr << " Warning: img_src.channels() is not 1, 3 or 4. It is = " << img_src.channels() << std::endl;
        std::shared_ptr<image_t> image_ptr(new image_t, [](image_t *img) { free_image(*img); delete img; });
        *image_ptr = mat_to_image_custom(img);
        return image_ptr;
    }

private:

    static image_t mat_to_image_custom(cv::Mat mat)
    {
        int w = mat.cols;
        int h = mat.rows;
        int c = mat.channels();
        image_t im = make_image_custom(w, h, c);
        unsigned char *data = (unsigned char *)mat.data;
        int step = mat.step;
        for (int y = 0; y < h; ++y) {
            for (int k = 0; k < c; ++k) {
                for (int x = 0; x < w; ++x) {
                    im.data[k*w*h + y*w + x] = data[y*step + x*c + k] / 255.0f;
                }
            }
        }
        return im;
    }

    static image_t make_empty_image(int w, int h, int c)
    {
        image_t out;
        out.data = 0;
        out.h = h;
        out.w = w;
        out.c = c;
        return out;
    }

    static image_t make_image_custom(int w, int h, int c)
    {
        image_t out = make_empty_image(w, h, c);
        out.data = (float *)calloc(h*w*c, sizeof(float));
        return out;
    }

#endif    // OPENCV

public:

    bool send_json_http(std::vector<bbox_t> cur_bbox_vec, std::vector<std::string> obj_names, int frame_id,
        std::string filename = std::string(), int timeout = 400000, int port = 8070)
    {
        std::string send_str;

        char *tmp_buf = (char *)calloc(1024, sizeof(char));
        if (!filename.empty()) {
            sprintf(tmp_buf, "{\n \"frame_id\":%d, \n \"filename\":\"%s\", \n \"objects\": [ \n", frame_id, filename.c_str());
        }
        else {
            sprintf(tmp_buf, "{\n \"frame_id\":%d, \n \"objects\": [ \n", frame_id);
        }
        send_str = tmp_buf;
        free(tmp_buf);

        for (auto & i : cur_bbox_vec) {
            char *buf = (char *)calloc(2048, sizeof(char));

            sprintf(buf, "  {\"class_id\":%d, \"name\":\"%s\", \"absolute_coordinates\":{\"center_x\":%d, \"center_y\":%d, \"width\":%d, \"height\":%d}, \"confidence\":%f",
                i.obj_id, obj_names[i.obj_id].c_str(), i.x, i.y, i.w, i.h, i.prob);

            //sprintf(buf, "  {\"class_id\":%d, \"name\":\"%s\", \"relative_coordinates\":{\"center_x\":%f, \"center_y\":%f, \"width\":%f, \"height\":%f}, \"confidence\":%f",
            //    i.obj_id, obj_names[i.obj_id], i.x, i.y, i.w, i.h, i.prob);

            send_str += buf;

            if (!std::isnan(i.z_3d)) {
                sprintf(buf, "\n    , \"coordinates_in_meters\":{\"x_3d\":%.2f, \"y_3d\":%.2f, \"z_3d\":%.2f}",
                    i.x_3d, i.y_3d, i.z_3d);
                send_str += buf;
            }

            send_str += "}\n";

            free(buf);
        }

        //send_str +=  "\n ] \n}, \n";
        send_str += "\n ] \n}";

        send_json_custom(send_str.c_str(), port, timeout);
        return true;
    }
};
// --------------------------------------------------------------------------------


#if defined(TRACK_OPTFLOW) && defined(OPENCV) && defined(GPU)

#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>

class Tracker_optflow {
public:
    const int gpu_count;
    const int gpu_id;
    const int flow_error;


    Tracker_optflow(int _gpu_id = 0, int win_size = 15, int max_level = 3, int iterations = 8000, int _flow_error = -1) :
        gpu_count(cv::cuda::getCudaEnabledDeviceCount()), gpu_id(std::min(_gpu_id, gpu_count-1)),
        flow_error((_flow_error > 0)? _flow_error:(win_size*4))
    {
        int const old_gpu_id = cv::cuda::getDevice();
        cv::cuda::setDevice(gpu_id);

        stream = cv::cuda::Stream();

        sync_PyrLKOpticalFlow_gpu = cv::cuda::SparsePyrLKOpticalFlow::create();
        sync_PyrLKOpticalFlow_gpu->setWinSize(cv::Size(win_size, win_size));    // 9, 15, 21, 31
        sync_PyrLKOpticalFlow_gpu->setMaxLevel(max_level);        // +- 3 pt
        sync_PyrLKOpticalFlow_gpu->setNumIters(iterations);    // 2000, def: 30

        cv::cuda::setDevice(old_gpu_id);
    }

    // just to avoid extra allocations
    cv::cuda::GpuMat src_mat_gpu;
    cv::cuda::GpuMat dst_mat_gpu, dst_grey_gpu;
    cv::cuda::GpuMat prev_pts_flow_gpu, cur_pts_flow_gpu;
    cv::cuda::GpuMat status_gpu, err_gpu;

    cv::cuda::GpuMat src_grey_gpu;    // used in both functions
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> sync_PyrLKOpticalFlow_gpu;
    cv::cuda::Stream stream;

    std::vector<bbox_t> cur_bbox_vec;
    std::vector<bool> good_bbox_vec_flags;
    cv::Mat prev_pts_flow_cpu;

    void update_cur_bbox_vec(std::vector<bbox_t> _cur_bbox_vec)
    {
        cur_bbox_vec = _cur_bbox_vec;
        good_bbox_vec_flags = std::vector<bool>(cur_bbox_vec.size(), true);
        cv::Mat prev_pts, cur_pts_flow_cpu;

        for (auto &i : cur_bbox_vec) {
            float x_center = (i.x + i.w / 2.0F);
            float y_center = (i.y + i.h / 2.0F);
            prev_pts.push_back(cv::Point2f(x_center, y_center));
        }

        if (prev_pts.rows == 0)
            prev_pts_flow_cpu = cv::Mat();
        else
            cv::transpose(prev_pts, prev_pts_flow_cpu);

        if (prev_pts_flow_gpu.cols < prev_pts_flow_cpu.cols) {
            prev_pts_flow_gpu = cv::cuda::GpuMat(prev_pts_flow_cpu.size(), prev_pts_flow_cpu.type());
            cur_pts_flow_gpu = cv::cuda::GpuMat(prev_pts_flow_cpu.size(), prev_pts_flow_cpu.type());

            status_gpu = cv::cuda::GpuMat(prev_pts_flow_cpu.size(), CV_8UC1);
            err_gpu = cv::cuda::GpuMat(prev_pts_flow_cpu.size(), CV_32FC1);
        }

        prev_pts_flow_gpu.upload(cv::Mat(prev_pts_flow_cpu), stream);
    }


    void update_tracking_flow(cv::Mat src_mat, std::vector<bbox_t> _cur_bbox_vec)
    {
        int const old_gpu_id = cv::cuda::getDevice();
        if (old_gpu_id != gpu_id)
            cv::cuda::setDevice(gpu_id);

        if (src_mat.channels() == 1 || src_mat.channels() == 3 || src_mat.channels() == 4) {
            if (src_mat_gpu.cols == 0) {
                src_mat_gpu = cv::cuda::GpuMat(src_mat.size(), src_mat.type());
                src_grey_gpu = cv::cuda::GpuMat(src_mat.size(), CV_8UC1);
            }

            if (src_mat.channels() == 1) {
                src_mat_gpu.upload(src_mat, stream);
                src_mat_gpu.copyTo(src_grey_gpu);
            }
            else if (src_mat.channels() == 3) {
                src_mat_gpu.upload(src_mat, stream);
                cv::cuda::cvtColor(src_mat_gpu, src_grey_gpu, CV_BGR2GRAY, 1, stream);
            }
            else if (src_mat.channels() == 4) {
                src_mat_gpu.upload(src_mat, stream);
                cv::cuda::cvtColor(src_mat_gpu, src_grey_gpu, CV_BGRA2GRAY, 1, stream);
            }
            else {
                std::cerr << " Warning: src_mat.channels() is not: 1, 3 or 4. It is = " << src_mat.channels() << " \n";
                return;
            }

        }
        update_cur_bbox_vec(_cur_bbox_vec);

        if (old_gpu_id != gpu_id)
            cv::cuda::setDevice(old_gpu_id);
    }


    std::vector<bbox_t> tracking_flow(cv::Mat dst_mat, bool check_error = true)
    {
        if (sync_PyrLKOpticalFlow_gpu.empty()) {
            std::cout << "sync_PyrLKOpticalFlow_gpu isn't initialized \n";
            return cur_bbox_vec;
        }

        int const old_gpu_id = cv::cuda::getDevice();
        if(old_gpu_id != gpu_id)
            cv::cuda::setDevice(gpu_id);

        if (dst_mat_gpu.cols == 0) {
            dst_mat_gpu = cv::cuda::GpuMat(dst_mat.size(), dst_mat.type());
            dst_grey_gpu = cv::cuda::GpuMat(dst_mat.size(), CV_8UC1);
        }

        //dst_grey_gpu.upload(dst_mat, stream);    // use BGR
        dst_mat_gpu.upload(dst_mat, stream);
        cv::cuda::cvtColor(dst_mat_gpu, dst_grey_gpu, CV_BGR2GRAY, 1, stream);

        if (src_grey_gpu.rows != dst_grey_gpu.rows || src_grey_gpu.cols != dst_grey_gpu.cols) {
            stream.waitForCompletion();
            src_grey_gpu = dst_grey_gpu.clone();
            cv::cuda::setDevice(old_gpu_id);
            return cur_bbox_vec;
        }

        ////sync_PyrLKOpticalFlow_gpu.sparse(src_grey_gpu, dst_grey_gpu, prev_pts_flow_gpu, cur_pts_flow_gpu, status_gpu, &err_gpu);    // OpenCV 2.4.x
        sync_PyrLKOpticalFlow_gpu->calc(src_grey_gpu, dst_grey_gpu, prev_pts_flow_gpu, cur_pts_flow_gpu, status_gpu, err_gpu, stream);    // OpenCV 3.x

        cv::Mat cur_pts_flow_cpu;
        cur_pts_flow_gpu.download(cur_pts_flow_cpu, stream);

        dst_grey_gpu.copyTo(src_grey_gpu, stream);

        cv::Mat err_cpu, status_cpu;
        err_gpu.download(err_cpu, stream);
        status_gpu.download(status_cpu, stream);

        stream.waitForCompletion();

        std::vector<bbox_t> result_bbox_vec;

        if (err_cpu.cols == cur_bbox_vec.size() && status_cpu.cols == cur_bbox_vec.size())
        {
            for (size_t i = 0; i < cur_bbox_vec.size(); ++i)
            {
                cv::Point2f cur_key_pt = cur_pts_flow_cpu.at<cv::Point2f>(0, i);
                cv::Point2f prev_key_pt = prev_pts_flow_cpu.at<cv::Point2f>(0, i);

                float moved_x = cur_key_pt.x - prev_key_pt.x;
                float moved_y = cur_key_pt.y - prev_key_pt.y;

                if (abs(moved_x) < 100 && abs(moved_y) < 100 && good_bbox_vec_flags[i])
                    if (err_cpu.at<float>(0, i) < flow_error && status_cpu.at<unsigned char>(0, i) != 0 &&
                        ((float)cur_bbox_vec[i].x + moved_x) > 0 && ((float)cur_bbox_vec[i].y + moved_y) > 0)
                    {
                        cur_bbox_vec[i].x += moved_x + 0.5;
                        cur_bbox_vec[i].y += moved_y + 0.5;
                        result_bbox_vec.push_back(cur_bbox_vec[i]);
                    }
                    else good_bbox_vec_flags[i] = false;
                else good_bbox_vec_flags[i] = false;

                //if(!check_error && !good_bbox_vec_flags[i]) result_bbox_vec.push_back(cur_bbox_vec[i]);
            }
        }

        cur_pts_flow_gpu.swap(prev_pts_flow_gpu);
        cur_pts_flow_cpu.copyTo(prev_pts_flow_cpu);

        if (old_gpu_id != gpu_id)
            cv::cuda::setDevice(old_gpu_id);

        return result_bbox_vec;
    }

};

#elif defined(TRACK_OPTFLOW) && defined(OPENCV)

//#include <opencv2/optflow.hpp>
#include <opencv2/video/tracking.hpp>

class Tracker_optflow {
public:
    const int flow_error;


    Tracker_optflow(int win_size = 15, int max_level = 3, int iterations = 8000, int _flow_error = -1) :
        flow_error((_flow_error > 0)? _flow_error:(win_size*4))
    {
        sync_PyrLKOpticalFlow = cv::SparsePyrLKOpticalFlow::create();
        sync_PyrLKOpticalFlow->setWinSize(cv::Size(win_size, win_size));    // 9, 15, 21, 31
        sync_PyrLKOpticalFlow->setMaxLevel(max_level);        // +- 3 pt

    }

    // just to avoid extra allocations
    cv::Mat dst_grey;
    cv::Mat prev_pts_flow, cur_pts_flow;
    cv::Mat status, err;

    cv::Mat src_grey;    // used in both functions
    cv::Ptr<cv::SparsePyrLKOpticalFlow> sync_PyrLKOpticalFlow;

    std::vector<bbox_t> cur_bbox_vec;
    std::vector<bool> good_bbox_vec_flags;

    void update_cur_bbox_vec(std::vector<bbox_t> _cur_bbox_vec)
    {
        cur_bbox_vec = _cur_bbox_vec;
        good_bbox_vec_flags = std::vector<bool>(cur_bbox_vec.size(), true);
        cv::Mat prev_pts, cur_pts_flow;

        for (auto &i : cur_bbox_vec) {
            float x_center = (i.x + i.w / 2.0F);
            float y_center = (i.y + i.h / 2.0F);
            prev_pts.push_back(cv::Point2f(x_center, y_center));
        }

        if (prev_pts.rows == 0)
            prev_pts_flow = cv::Mat();
        else
            cv::transpose(prev_pts, prev_pts_flow);
    }


    void update_tracking_flow(cv::Mat new_src_mat, std::vector<bbox_t> _cur_bbox_vec)
    {
        if (new_src_mat.channels() == 1) {
            src_grey = new_src_mat.clone();
        }
        else if (new_src_mat.channels() == 3) {
            cv::cvtColor(new_src_mat, src_grey, CV_BGR2GRAY, 1);
        }
        else if (new_src_mat.channels() == 4) {
            cv::cvtColor(new_src_mat, src_grey, CV_BGRA2GRAY, 1);
        }
        else {
            std::cerr << " Warning: new_src_mat.channels() is not: 1, 3 or 4. It is = " << new_src_mat.channels() << " \n";
            return;
        }
        update_cur_bbox_vec(_cur_bbox_vec);
    }


    std::vector<bbox_t> tracking_flow(cv::Mat new_dst_mat, bool check_error = true)
    {
        if (sync_PyrLKOpticalFlow.empty()) {
            std::cout << "sync_PyrLKOpticalFlow isn't initialized \n";
            return cur_bbox_vec;
        }

        cv::cvtColor(new_dst_mat, dst_grey, CV_BGR2GRAY, 1);

        if (src_grey.rows != dst_grey.rows || src_grey.cols != dst_grey.cols) {
            src_grey = dst_grey.clone();
            //std::cerr << " Warning: src_grey.rows != dst_grey.rows || src_grey.cols != dst_grey.cols \n";
            return cur_bbox_vec;
        }

        if (prev_pts_flow.cols < 1) {
            return cur_bbox_vec;
        }

        ////sync_PyrLKOpticalFlow_gpu.sparse(src_grey_gpu, dst_grey_gpu, prev_pts_flow_gpu, cur_pts_flow_gpu, status_gpu, &err_gpu);    // OpenCV 2.4.x
        sync_PyrLKOpticalFlow->calc(src_grey, dst_grey, prev_pts_flow, cur_pts_flow, status, err);    // OpenCV 3.x

        dst_grey.copyTo(src_grey);

        std::vector<bbox_t> result_bbox_vec;

        if (err.rows == cur_bbox_vec.size() && status.rows == cur_bbox_vec.size())
        {
            for (size_t i = 0; i < cur_bbox_vec.size(); ++i)
            {
                cv::Point2f cur_key_pt = cur_pts_flow.at<cv::Point2f>(0, i);
                cv::Point2f prev_key_pt = prev_pts_flow.at<cv::Point2f>(0, i);

                float moved_x = cur_key_pt.x - prev_key_pt.x;
                float moved_y = cur_key_pt.y - prev_key_pt.y;

                if (abs(moved_x) < 100 && abs(moved_y) < 100 && good_bbox_vec_flags[i])
                    if (err.at<float>(0, i) < flow_error && status.at<unsigned char>(0, i) != 0 &&
                        ((float)cur_bbox_vec[i].x + moved_x) > 0 && ((float)cur_bbox_vec[i].y + moved_y) > 0)
                    {
                        cur_bbox_vec[i].x += moved_x + 0.5;
                        cur_bbox_vec[i].y += moved_y + 0.5;
                        result_bbox_vec.push_back(cur_bbox_vec[i]);
                    }
                    else good_bbox_vec_flags[i] = false;
                else good_bbox_vec_flags[i] = false;

                //if(!check_error && !good_bbox_vec_flags[i]) result_bbox_vec.push_back(cur_bbox_vec[i]);
            }
        }

        prev_pts_flow = cur_pts_flow.clone();

        return result_bbox_vec;
    }

};
#else

class Tracker_optflow {};

#endif    // defined(TRACK_OPTFLOW) && defined(OPENCV)


#ifdef OPENCV

static cv::Scalar obj_id_to_color(int obj_id) {
    int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };
    int const offset = obj_id * 123457 % 6;
    int const color_scale = 150 + (obj_id * 123457) % 100;
    cv::Scalar color(colors[offset][0], colors[offset][1], colors[offset][2]);
    color *= color_scale;
    return color;
}

class preview_boxes_t {
    enum { frames_history = 30 };    // how long to keep the history saved

    struct preview_box_track_t {
        unsigned int track_id, obj_id, last_showed_frames_ago;
        bool current_detection;
        bbox_t bbox;
        cv::Mat mat_obj, mat_resized_obj;
        preview_box_track_t() : track_id(0), obj_id(0), last_showed_frames_ago(frames_history), current_detection(false) {}
    };
    std::vector<preview_box_track_t> preview_box_track_id;
    size_t const preview_box_size, bottom_offset;
    bool const one_off_detections;
public:
    preview_boxes_t(size_t _preview_box_size = 100, size_t _bottom_offset = 100, bool _one_off_detections = false) :
        preview_box_size(_preview_box_size), bottom_offset(_bottom_offset), one_off_detections(_one_off_detections)
    {}

    void set(cv::Mat src_mat, std::vector<bbox_t> result_vec)
    {
        size_t const count_preview_boxes = src_mat.cols / preview_box_size;
        if (preview_box_track_id.size() != count_preview_boxes) preview_box_track_id.resize(count_preview_boxes);

        // increment frames history
        for (auto &i : preview_box_track_id)
            i.last_showed_frames_ago = std::min((unsigned)frames_history, i.last_showed_frames_ago + 1);

        // occupy empty boxes
        for (auto &k : result_vec) {
            bool found = false;
            // find the same (track_id)
            for (auto &i : preview_box_track_id) {
                if (i.track_id == k.track_id) {
                    if (!one_off_detections) i.last_showed_frames_ago = 0; // for tracked objects
                    found = true;
                    break;
                }
            }
            if (!found) {
                // find empty box
                for (auto &i : preview_box_track_id) {
                    if (i.last_showed_frames_ago == frames_history) {
                        if (!one_off_detections && k.frames_counter == 0) break; // don't show if obj isn't tracked yet
                        i.track_id = k.track_id;
                        i.obj_id = k.obj_id;
                        i.bbox = k;
                        i.last_showed_frames_ago = 0;
                        break;
                    }
                }
            }
        }

        // draw preview box (from old or current frame)
        for (size_t i = 0; i < preview_box_track_id.size(); ++i)
        {
            // get object image
            cv::Mat dst = preview_box_track_id[i].mat_resized_obj;
            preview_box_track_id[i].current_detection = false;

            for (auto &k : result_vec) {
                if (preview_box_track_id[i].track_id == k.track_id) {
                    if (one_off_detections && preview_box_track_id[i].last_showed_frames_ago > 0) {
                        preview_box_track_id[i].last_showed_frames_ago = frames_history; break;
                    }
                    bbox_t b = k;
                    cv::Rect r(b.x, b.y, b.w, b.h);
                    cv::Rect img_rect(cv::Point2i(0, 0), src_mat.size());
                    cv::Rect rect_roi = r & img_rect;
                    if (rect_roi.width > 1 || rect_roi.height > 1) {
                        cv::Mat roi = src_mat(rect_roi);
                        cv::resize(roi, dst, cv::Size(preview_box_size, preview_box_size), cv::INTER_NEAREST);
                        preview_box_track_id[i].mat_obj = roi.clone();
                        preview_box_track_id[i].mat_resized_obj = dst.clone();
                        preview_box_track_id[i].current_detection = true;
                        preview_box_track_id[i].bbox = k;
                    }
                    break;
                }
            }
        }
    }


    void draw(cv::Mat draw_mat, bool show_small_boxes = false)
    {
        // draw preview box (from old or current frame)
        for (size_t i = 0; i < preview_box_track_id.size(); ++i)
        {
            auto &prev_box = preview_box_track_id[i];

            // draw object image
            cv::Mat dst = prev_box.mat_resized_obj;
            if (prev_box.last_showed_frames_ago < frames_history &&
                dst.size() == cv::Size(preview_box_size, preview_box_size))
            {
                cv::Rect dst_rect_roi(cv::Point2i(i * preview_box_size, draw_mat.rows - bottom_offset), dst.size());
                cv::Mat dst_roi = draw_mat(dst_rect_roi);
                dst.copyTo(dst_roi);

                cv::Scalar color = obj_id_to_color(prev_box.obj_id);
                int thickness = (prev_box.current_detection) ? 5 : 1;
                cv::rectangle(draw_mat, dst_rect_roi, color, thickness);

                unsigned int const track_id = prev_box.track_id;
                std::string track_id_str = (track_id > 0) ? std::to_string(track_id) : "";
                putText(draw_mat, track_id_str, dst_rect_roi.tl() - cv::Point2i(-4, 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.9, cv::Scalar(0, 0, 0), 2);

                std::string size_str = std::to_string(prev_box.bbox.w) + "x" + std::to_string(prev_box.bbox.h);
                putText(draw_mat, size_str, dst_rect_roi.tl() + cv::Point2i(0, 12), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);

                if (!one_off_detections && prev_box.current_detection) {
                    cv::line(draw_mat, dst_rect_roi.tl() + cv::Point2i(preview_box_size, 0),
                        cv::Point2i(prev_box.bbox.x, prev_box.bbox.y + prev_box.bbox.h),
                        color);
                }

                if (one_off_detections && show_small_boxes) {
                    cv::Rect src_rect_roi(cv::Point2i(prev_box.bbox.x, prev_box.bbox.y),
                        cv::Size(prev_box.bbox.w, prev_box.bbox.h));
                    unsigned int const color_history = (255 * prev_box.last_showed_frames_ago) / frames_history;
                    color = cv::Scalar(255 - 3 * color_history, 255 - 2 * color_history, 255 - 1 * color_history);
                    if (prev_box.mat_obj.size() == src_rect_roi.size()) {
                        prev_box.mat_obj.copyTo(draw_mat(src_rect_roi));
                    }
                    cv::rectangle(draw_mat, src_rect_roi, color, thickness);
                    putText(draw_mat, track_id_str, src_rect_roi.tl() - cv::Point2i(0, 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0, 0, 0), 1);
                }
            }
        }
    }
};


class track_kalman_t
{
    int track_id_counter;
    std::chrono::steady_clock::time_point global_last_time;
    float dT;

public:
    int max_objects;    // max objects for tracking
    int min_frames;     // min frames to consider an object as detected
    const float max_dist;   // max distance (in px) to track with the same ID
    cv::Size img_size;  // max value of x,y,w,h

    struct tst_t {
        int track_id;
        int state_id;
        std::chrono::steady_clock::time_point last_time;
        int detection_count;
        tst_t() : track_id(-1), state_id(-1) {}
    };
    std::vector<tst_t> track_id_state_id_time;
    std::vector<bbox_t> result_vec_pred;

    struct one_kalman_t;
    std::vector<one_kalman_t> kalman_vec;

    struct one_kalman_t
    {
        cv::KalmanFilter kf;
        cv::Mat state;
        cv::Mat meas;
        int measSize, stateSize, contrSize;

        void set_delta_time(float dT) {
            kf.transitionMatrix.at<float>(2) = dT;
            kf.transitionMatrix.at<float>(9) = dT;
        }

        void set(bbox_t box)
        {
            initialize_kalman();

            kf.errorCovPre.at<float>(0) = 1; // px
            kf.errorCovPre.at<float>(7) = 1; // px
            kf.errorCovPre.at<float>(14) = 1;
            kf.errorCovPre.at<float>(21) = 1;
            kf.errorCovPre.at<float>(28) = 1; // px
            kf.errorCovPre.at<float>(35) = 1; // px

            state.at<float>(0) = box.x;
            state.at<float>(1) = box.y;
            state.at<float>(2) = 0;
            state.at<float>(3) = 0;
            state.at<float>(4) = box.w;
            state.at<float>(5) = box.h;
            // <<<< Initialization

            kf.statePost = state;
        }

        // Kalman.correct() calculates: statePost = statePre + gain * (z(k)-measurementMatrix*statePre);
        // corrected state (x(k)): x(k)=x'(k)+K(k)*(z(k)-H*x'(k))
        void correct(bbox_t box) {
            meas.at<float>(0) = box.x;
            meas.at<float>(1) = box.y;
            meas.at<float>(2) = box.w;
            meas.at<float>(3) = box.h;

            kf.correct(meas);

            bbox_t new_box = predict();
            if (new_box.w == 0 || new_box.h == 0) {
                set(box);
                //std::cerr << " force set(): track_id = " << box.track_id <<
                //    ", x = " << box.x << ", y = " << box.y << ", w = " << box.w << ", h = " << box.h << std::endl;
            }
        }

        // Kalman.predict() calculates: statePre = TransitionMatrix * statePost;
        // predicted state (x'(k)): x(k)=A*x(k-1)+B*u(k)
        bbox_t predict() {
            bbox_t box;
            state = kf.predict();

            box.x = state.at<float>(0);
            box.y = state.at<float>(1);
            box.w = state.at<float>(4);
            box.h = state.at<float>(5);
            return box;
        }

        void initialize_kalman()
        {
            kf = cv::KalmanFilter(stateSize, measSize, contrSize, CV_32F);

            // Transition State Matrix A
            // Note: set dT at each processing step!
            // [ 1 0 dT 0  0 0 ]
            // [ 0 1 0  dT 0 0 ]
            // [ 0 0 1  0  0 0 ]
            // [ 0 0 0  1  0 0 ]
            // [ 0 0 0  0  1 0 ]
            // [ 0 0 0  0  0 1 ]
            cv::setIdentity(kf.transitionMatrix);

            // Measure Matrix H
            // [ 1 0 0 0 0 0 ]
            // [ 0 1 0 0 0 0 ]
            // [ 0 0 0 0 1 0 ]
            // [ 0 0 0 0 0 1 ]
            kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, CV_32F);
            kf.measurementMatrix.at<float>(0) = 1.0f;
            kf.measurementMatrix.at<float>(7) = 1.0f;
            kf.measurementMatrix.at<float>(16) = 1.0f;
            kf.measurementMatrix.at<float>(23) = 1.0f;

            // Process Noise Covariance Matrix Q - result smoother with lower values (1e-2)
            // [ Ex   0   0     0     0    0  ]
            // [ 0    Ey  0     0     0    0  ]
            // [ 0    0   Ev_x  0     0    0  ]
            // [ 0    0   0     Ev_y  0    0  ]
            // [ 0    0   0     0     Ew   0  ]
            // [ 0    0   0     0     0    Eh ]
            //cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-3));
            kf.processNoiseCov.at<float>(0) = 1e-2;
            kf.processNoiseCov.at<float>(7) = 1e-2;
            kf.processNoiseCov.at<float>(14) = 1e-2;// 5.0f;
            kf.processNoiseCov.at<float>(21) = 1e-2;// 5.0f;
            kf.processNoiseCov.at<float>(28) = 5e-3;
            kf.processNoiseCov.at<float>(35) = 5e-3;

            // Measures Noise Covariance Matrix R - result smoother with higher values (1e-1)
            cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));

            //cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1e-2));
            // <<<< Kalman Filter

            set_delta_time(0);
        }


        one_kalman_t(int _stateSize = 6, int _measSize = 4, int _contrSize = 0) :
            kf(_stateSize, _measSize, _contrSize, CV_32F), measSize(_measSize), stateSize(_stateSize), contrSize(_contrSize)
        {
            state = cv::Mat(stateSize, 1, CV_32F);  // [x,y,v_x,v_y,w,h]
            meas = cv::Mat(measSize, 1, CV_32F);    // [z_x,z_y,z_w,z_h]
            //cv::Mat procNoise(stateSize, 1, type)
            // [E_x,E_y,E_v_x,E_v_y,E_w,E_h]

            initialize_kalman();
        }
    };
    // ------------------------------------------



    track_kalman_t(int _max_objects = 1000, int _min_frames = 3, float _max_dist = 40, cv::Size _img_size = cv::Size(10000, 10000)) :
        max_objects(_max_objects), min_frames(_min_frames), max_dist(_max_dist), img_size(_img_size),
        track_id_counter(0)
    {
        kalman_vec.resize(max_objects);
        track_id_state_id_time.resize(max_objects);
        result_vec_pred.resize(max_objects);
    }

    float calc_dt() {
        dT = std::chrono::duration<double>(std::chrono::steady_clock::now() - global_last_time).count();
        return dT;
    }

    static float get_distance(float src_x, float src_y, float dst_x, float dst_y) {
        return sqrtf((src_x - dst_x)*(src_x - dst_x) + (src_y - dst_y)*(src_y - dst_y));
    }

    void clear_old_states() {
        // clear old bboxes
        for (size_t state_id = 0; state_id < track_id_state_id_time.size(); ++state_id)
        {
            float time_sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - track_id_state_id_time[state_id].last_time).count();
            float time_wait = 0.5;    // 0.5 second
            if (track_id_state_id_time[state_id].track_id > -1)
            {
                if ((result_vec_pred[state_id].x > img_size.width) ||
                    (result_vec_pred[state_id].y > img_size.height))
                {
                    track_id_state_id_time[state_id].track_id = -1;
                }

                if (time_sec >= time_wait || track_id_state_id_time[state_id].detection_count < 0) {
                    //std::cerr << " remove track_id = " << track_id_state_id_time[state_id].track_id << ", state_id = " << state_id << std::endl;
                    track_id_state_id_time[state_id].track_id = -1; // remove bbox
                }
            }
        }
    }

    tst_t get_state_id(bbox_t find_box, std::vector<bool> &busy_vec)
    {
        tst_t tst;
        tst.state_id = -1;

        float min_dist = std::numeric_limits<float>::max();

        for (size_t i = 0; i < max_objects; ++i)
        {
            if (track_id_state_id_time[i].track_id > -1 && result_vec_pred[i].obj_id == find_box.obj_id && busy_vec[i] == false)
            {
                bbox_t pred_box = result_vec_pred[i];

                float dist = get_distance(pred_box.x, pred_box.y, find_box.x, find_box.y);

                float movement_dist = std::max(max_dist, static_cast<float>(std::max(pred_box.w, pred_box.h)) );

                if ((dist < movement_dist) && (dist < min_dist)) {
                    min_dist = dist;
                    tst.state_id = i;
                }
            }
        }

        if (tst.state_id > -1) {
            track_id_state_id_time[tst.state_id].last_time = std::chrono::steady_clock::now();
            track_id_state_id_time[tst.state_id].detection_count = std::max(track_id_state_id_time[tst.state_id].detection_count + 2, 10);
            tst = track_id_state_id_time[tst.state_id];
            busy_vec[tst.state_id] = true;
        }
        else {
            //std::cerr << " Didn't find: obj_id = " << find_box.obj_id << ", x = " << find_box.x << ", y = " << find_box.y <<
            //    ", track_id_counter = " << track_id_counter << std::endl;
        }

        return tst;
    }

    tst_t new_state_id(std::vector<bool> &busy_vec)
    {
        tst_t tst;
        // find empty cell to add new track_id
        auto it = std::find_if(track_id_state_id_time.begin(), track_id_state_id_time.end(), [&](tst_t &v) { return v.track_id == -1; });
        if (it != track_id_state_id_time.end()) {
            it->state_id = it - track_id_state_id_time.begin();
            //it->track_id = track_id_counter++;
            it->track_id = 0;
            it->last_time = std::chrono::steady_clock::now();
            it->detection_count = 1;
            tst = *it;
            busy_vec[it->state_id] = true;
        }

        return tst;
    }

    std::vector<tst_t> find_state_ids(std::vector<bbox_t> result_vec)
    {
        std::vector<tst_t> tst_vec(result_vec.size());

        std::vector<bool> busy_vec(max_objects, false);

        for (size_t i = 0; i < result_vec.size(); ++i)
        {
            tst_t tst = get_state_id(result_vec[i], busy_vec);
            int state_id = tst.state_id;
            int track_id = tst.track_id;

            // if new state_id
            if (state_id < 0) {
                tst = new_state_id(busy_vec);
                state_id = tst.state_id;
                track_id = tst.track_id;
                if (state_id > -1) {
                    kalman_vec[state_id].set(result_vec[i]);
                    //std::cerr << " post: ";
                }
            }

            //std::cerr << " track_id = " << track_id << ", state_id = " << state_id <<
            //    ", x = " << result_vec[i].x << ", det_count = " << tst.detection_count << std::endl;

            if (state_id > -1) {
                tst_vec[i] = tst;
                result_vec_pred[state_id] = result_vec[i];
                result_vec_pred[state_id].track_id = track_id;
            }
        }

        return tst_vec;
    }

    std::vector<bbox_t> predict()
    {
        clear_old_states();
        std::vector<bbox_t> result_vec;

        for (size_t i = 0; i < max_objects; ++i)
        {
            tst_t tst = track_id_state_id_time[i];
            if (tst.track_id > -1) {
                bbox_t box = kalman_vec[i].predict();

                result_vec_pred[i].x = box.x;
                result_vec_pred[i].y = box.y;
                result_vec_pred[i].w = box.w;
                result_vec_pred[i].h = box.h;

                if (tst.detection_count >= min_frames)
                {
                    if (track_id_state_id_time[i].track_id == 0) {
                        track_id_state_id_time[i].track_id = ++track_id_counter;
                        result_vec_pred[i].track_id = track_id_counter;
                    }

                    result_vec.push_back(result_vec_pred[i]);
                }
            }
        }
        //std::cerr << "         result_vec.size() = " << result_vec.size() << std::endl;

        //global_last_time = std::chrono::steady_clock::now();

        return result_vec;
    }


    std::vector<bbox_t> correct(std::vector<bbox_t> result_vec)
    {
        calc_dt();
        clear_old_states();

        for (size_t i = 0; i < max_objects; ++i)
            track_id_state_id_time[i].detection_count--;

        std::vector<tst_t> tst_vec = find_state_ids(result_vec);

        for (size_t i = 0; i < tst_vec.size(); ++i) {
            tst_t tst = tst_vec[i];
            int state_id = tst.state_id;
            if (state_id > -1)
            {
                kalman_vec[state_id].set_delta_time(dT);
                kalman_vec[state_id].correct(result_vec_pred[state_id]);
            }
        }

        result_vec = predict();

        global_last_time = std::chrono::steady_clock::now();

        return result_vec;
    }

};
// ----------------------------------------------
#endif    // OPENCV

#endif    // __cplusplus

#endif    // YOLO_V2_CLASS_HPP
