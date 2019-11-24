#include "darknet.h"
#include "yolo_v2_class.hpp"

#include "network.h"

extern "C" {
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include "option_list.h"
#include "stb_image.h"
}
//#include <sys/time.h>

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

#define NFRAMES 3

//static Detector* detector = NULL;
static std::unique_ptr<Detector> detector;

int init(const char *configurationFilename, const char *weightsFilename, int gpu)
{
    detector.reset(new Detector(configurationFilename, weightsFilename, gpu));
    return 1;
}

int detect_image(const char *filename, bbox_t_container &container)
{
    std::vector<bbox_t> detection = detector->detect(filename);
    for (size_t i = 0; i < detection.size() && i < C_SHARP_MAX_OBJECTS; ++i)
        container.candidates[i] = detection[i];
    return detection.size();
}

int detect_mat(const uint8_t* data, const size_t data_length, bbox_t_container &container) {
#ifdef OPENCV
    std::vector<char> vdata(data, data + data_length);
    cv::Mat image = imdecode(cv::Mat(vdata), 1);

    std::vector<bbox_t> detection = detector->detect(image);
    for (size_t i = 0; i < detection.size() && i < C_SHARP_MAX_OBJECTS; ++i)
        container.candidates[i] = detection[i];
    return detection.size();
#else
    return -1;
#endif    // OPENCV
}

int dispose() {
    //if (detector != NULL) delete detector;
    //detector = NULL;
    detector.reset();
    return 1;
}

int get_device_count() {
#ifdef GPU
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
#else
    return -1;
#endif	// GPU
}

bool built_with_cuda(){
#ifdef GPU
    return true;
#else
    return false;
#endif
}

bool built_with_cudnn(){
#ifdef CUDNN
    return true;
#else
    return false;
#endif
}

bool built_with_opencv(){
#ifdef OPENCV
    return true;
#else
    return false;
#endif
}


int get_device_name(int gpu, char* deviceName) {
#ifdef GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu);
    std::string result = prop.name;
    std::copy(result.begin(), result.end(), deviceName);
    return 1;
#else
    return -1;
#endif	// GPU
}

#ifdef GPU
void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        const char *s = cudaGetErrorString(status);
        printf("CUDA Error Prev: %s\n", s);
    }
}
#endif

struct detector_gpu_t {
    network net;
    image images[NFRAMES];
    float *avg;
    float* predictions[NFRAMES];
    int demo_index;
    unsigned int *track_id;
};

LIB_API Detector::Detector(std::string cfg_filename, std::string weight_filename, int gpu_id) : cur_gpu_id(gpu_id)
{
    wait_stream = 0;
#ifdef GPU
    int old_gpu_index;
    check_cuda( cudaGetDevice(&old_gpu_index) );
#endif

    detector_gpu_ptr = std::make_shared<detector_gpu_t>();
    detector_gpu_t &detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());

#ifdef GPU
    //check_cuda( cudaSetDevice(cur_gpu_id) );
    cuda_set_device(cur_gpu_id);
    printf(" Used GPU %d \n", cur_gpu_id);
#endif
    network &net = detector_gpu.net;
    net.gpu_index = cur_gpu_id;
    //gpu_index = i;

    _cfg_filename = cfg_filename;
    _weight_filename = weight_filename;

    char *cfgfile = const_cast<char *>(_cfg_filename.c_str());
    char *weightfile = const_cast<char *>(_weight_filename.c_str());

    net = parse_network_cfg_custom(cfgfile, 1, 0);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    net.gpu_index = cur_gpu_id;
    fuse_conv_batchnorm(net);

    layer l = net.layers[net.n - 1];
    int j;

    detector_gpu.avg = (float *)calloc(l.outputs, sizeof(float));
    for (j = 0; j < NFRAMES; ++j) detector_gpu.predictions[j] = (float*)calloc(l.outputs, sizeof(float));
    for (j = 0; j < NFRAMES; ++j) detector_gpu.images[j] = make_image(1, 1, 3);

    detector_gpu.track_id = (unsigned int *)calloc(l.classes, sizeof(unsigned int));
    for (j = 0; j < l.classes; ++j) detector_gpu.track_id[j] = 1;

#ifdef GPU
    check_cuda( cudaSetDevice(old_gpu_index) );
#endif
}


LIB_API Detector::~Detector()
{
    detector_gpu_t &detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());
    //layer l = detector_gpu.net.layers[detector_gpu.net.n - 1];

    free(detector_gpu.track_id);

    free(detector_gpu.avg);
    for (int j = 0; j < NFRAMES; ++j) free(detector_gpu.predictions[j]);
    for (int j = 0; j < NFRAMES; ++j) if (detector_gpu.images[j].data) free(detector_gpu.images[j].data);

#ifdef GPU
    int old_gpu_index;
    cudaGetDevice(&old_gpu_index);
    cuda_set_device(detector_gpu.net.gpu_index);
#endif

    free_network(detector_gpu.net);

#ifdef GPU
    cudaSetDevice(old_gpu_index);
#endif
}

LIB_API int Detector::get_net_width() const {
    detector_gpu_t &detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());
    return detector_gpu.net.w;
}
LIB_API int Detector::get_net_height() const {
    detector_gpu_t &detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());
    return detector_gpu.net.h;
}
LIB_API int Detector::get_net_color_depth() const {
    detector_gpu_t &detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());
    return detector_gpu.net.c;
}


LIB_API std::vector<bbox_t> Detector::detect(std::string image_filename, float thresh, bool use_mean)
{
    std::shared_ptr<image_t> image_ptr(new image_t, [](image_t *img) { if (img->data) free(img->data); delete img; });
    *image_ptr = load_image(image_filename);
    return detect(*image_ptr, thresh, use_mean);
}

static image load_image_stb(char *filename, int channels)
{
    int w, h, c;
    unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
    if (!data)
        throw std::runtime_error("file not found");
    if (channels) c = channels;
    int i, j, k;
    image im = make_image(w, h, c);
    for (k = 0; k < c; ++k) {
        for (j = 0; j < h; ++j) {
            for (i = 0; i < w; ++i) {
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                im.data[dst_index] = (float)data[src_index] / 255.;
            }
        }
    }
    free(data);
    return im;
}

LIB_API image_t Detector::load_image(std::string image_filename)
{
    char *input = const_cast<char *>(image_filename.c_str());
    image im = load_image_stb(input, 3);

    image_t img;
    img.c = im.c;
    img.data = im.data;
    img.h = im.h;
    img.w = im.w;

    return img;
}


LIB_API void Detector::free_image(image_t m)
{
    if (m.data) {
        free(m.data);
    }
}

LIB_API std::vector<bbox_t> Detector::detect(image_t img, float thresh, bool use_mean)
{
    detector_gpu_t &detector_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());
    network &net = detector_gpu.net;
#ifdef GPU
    int old_gpu_index;
    cudaGetDevice(&old_gpu_index);
    if(cur_gpu_id != old_gpu_index)
        cudaSetDevice(net.gpu_index);

    net.wait_stream = wait_stream;    // 1 - wait CUDA-stream, 0 - not to wait
#endif
    //std::cout << "net.gpu_index = " << net.gpu_index << std::endl;

    image im;
    im.c = img.c;
    im.data = img.data;
    im.h = img.h;
    im.w = img.w;

    image sized;

    if (net.w == im.w && net.h == im.h) {
        sized = make_image(im.w, im.h, im.c);
        memcpy(sized.data, im.data, im.w*im.h*im.c * sizeof(float));
    }
    else
        sized = resize_image(im, net.w, net.h);

    layer l = net.layers[net.n - 1];

    float *X = sized.data;

    float *prediction = network_predict(net, X);

    if (use_mean) {
        memcpy(detector_gpu.predictions[detector_gpu.demo_index], prediction, l.outputs * sizeof(float));
        mean_arrays(detector_gpu.predictions, NFRAMES, l.outputs, detector_gpu.avg);
        l.output = detector_gpu.avg;
        detector_gpu.demo_index = (detector_gpu.demo_index + 1) % NFRAMES;
    }
    //get_region_boxes(l, 1, 1, thresh, detector_gpu.probs, detector_gpu.boxes, 0, 0);
    //if (nms) do_nms_sort(detector_gpu.boxes, detector_gpu.probs, l.w*l.h*l.n, l.classes, nms);

    int nboxes = 0;
    int letterbox = 0;
    float hier_thresh = 0.5;
    detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

    std::vector<bbox_t> bbox_vec;

    for (int i = 0; i < nboxes; ++i) {
        box b = dets[i].bbox;
        int const obj_id = max_index(dets[i].prob, l.classes);
        float const prob = dets[i].prob[obj_id];

        if (prob > thresh)
        {
            bbox_t bbox;
            bbox.x = std::max((double)0, (b.x - b.w / 2.)*im.w);
            bbox.y = std::max((double)0, (b.y - b.h / 2.)*im.h);
            bbox.w = b.w*im.w;
            bbox.h = b.h*im.h;
            bbox.obj_id = obj_id;
            bbox.prob = prob;
            bbox.track_id = 0;
            bbox.frames_counter = 0;
            bbox.x_3d = NAN;
            bbox.y_3d = NAN;
            bbox.z_3d = NAN;

            bbox_vec.push_back(bbox);
        }
    }

    free_detections(dets, nboxes);
    if(sized.data)
        free(sized.data);

#ifdef GPU
    if (cur_gpu_id != old_gpu_index)
        cudaSetDevice(old_gpu_index);
#endif

    return bbox_vec;
}

LIB_API std::vector<bbox_t> Detector::tracking_id(std::vector<bbox_t> cur_bbox_vec, bool const change_history,
    int const frames_story, int const max_dist)
{
    detector_gpu_t &det_gpu = *static_cast<detector_gpu_t *>(detector_gpu_ptr.get());

    bool prev_track_id_present = false;
    for (auto &i : prev_bbox_vec_deque)
        if (i.size() > 0) prev_track_id_present = true;

    if (!prev_track_id_present) {
        for (size_t i = 0; i < cur_bbox_vec.size(); ++i)
            cur_bbox_vec[i].track_id = det_gpu.track_id[cur_bbox_vec[i].obj_id]++;
        prev_bbox_vec_deque.push_front(cur_bbox_vec);
        if (prev_bbox_vec_deque.size() > frames_story) prev_bbox_vec_deque.pop_back();
        return cur_bbox_vec;
    }

    std::vector<unsigned int> dist_vec(cur_bbox_vec.size(), std::numeric_limits<unsigned int>::max());

    for (auto &prev_bbox_vec : prev_bbox_vec_deque) {
        for (auto &i : prev_bbox_vec) {
            int cur_index = -1;
            for (size_t m = 0; m < cur_bbox_vec.size(); ++m) {
                bbox_t const& k = cur_bbox_vec[m];
                if (i.obj_id == k.obj_id) {
                    float center_x_diff = (float)(i.x + i.w/2) - (float)(k.x + k.w/2);
                    float center_y_diff = (float)(i.y + i.h/2) - (float)(k.y + k.h/2);
                    unsigned int cur_dist = sqrt(center_x_diff*center_x_diff + center_y_diff*center_y_diff);
                    if (cur_dist < max_dist && (k.track_id == 0 || dist_vec[m] > cur_dist)) {
                        dist_vec[m] = cur_dist;
                        cur_index = m;
                    }
                }
            }

            bool track_id_absent = !std::any_of(cur_bbox_vec.begin(), cur_bbox_vec.end(),
                [&i](bbox_t const& b) { return b.track_id == i.track_id && b.obj_id == i.obj_id; });

            if (cur_index >= 0 && track_id_absent){
                cur_bbox_vec[cur_index].track_id = i.track_id;
                cur_bbox_vec[cur_index].w = (cur_bbox_vec[cur_index].w + i.w) / 2;
                cur_bbox_vec[cur_index].h = (cur_bbox_vec[cur_index].h + i.h) / 2;
            }
        }
    }

    for (size_t i = 0; i < cur_bbox_vec.size(); ++i)
        if (cur_bbox_vec[i].track_id == 0)
            cur_bbox_vec[i].track_id = det_gpu.track_id[cur_bbox_vec[i].obj_id]++;

    if (change_history) {
        prev_bbox_vec_deque.push_front(cur_bbox_vec);
        if (prev_bbox_vec_deque.size() > frames_story) prev_bbox_vec_deque.pop_back();
    }

    return cur_bbox_vec;
}


void *Detector::get_cuda_context()
{
#ifdef GPU
    int old_gpu_index;
    cudaGetDevice(&old_gpu_index);
    if (cur_gpu_id != old_gpu_index)
        cudaSetDevice(cur_gpu_id);

    void *cuda_context = cuda_get_context();

    if (cur_gpu_id != old_gpu_index)
        cudaSetDevice(old_gpu_index);

    return cuda_context;
#else   // GPU
    return NULL;
#endif  // GPU
}