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

#define FRAMES 3

struct detector_gpu_t{
	float **probs;
	box *boxes;
	network net;
	image images[FRAMES];
	float *avg;
	float *predictions[FRAMES];
};


YOLODLL_API Detector::Detector(std::string cfg_filename, std::string weight_filename, int gpu_id)
{
	int old_gpu_index;
	cudaGetDevice(&old_gpu_index);

	detector_gpu_ptr = std::make_shared<detector_gpu_t>();
	detector_gpu_t &detector_gpu = *reinterpret_cast<detector_gpu_t *>(detector_gpu_ptr.get());

	cudaSetDevice(gpu_id);
	network &net = detector_gpu.net;
	net.gpu_index = gpu_id;
	//gpu_index = i;
	
	char *cfgfile = const_cast<char *>(cfg_filename.data());
	char *weightfile = const_cast<char *>(weight_filename.data());

	net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	net.gpu_index = gpu_id;

	layer l = net.layers[net.n - 1];
	int j;

	detector_gpu.avg = (float *)calloc(l.outputs, sizeof(float));
	for (j = 0; j < FRAMES; ++j) detector_gpu.predictions[j] = (float *)calloc(l.outputs, sizeof(float));
	for (j = 0; j < FRAMES; ++j) detector_gpu.images[j] = make_image(1, 1, 3);

	detector_gpu.boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
	detector_gpu.probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
	for (j = 0; j < l.w*l.h*l.n; ++j) detector_gpu.probs[j] = (float *)calloc(l.classes, sizeof(float));

	cudaSetDevice(old_gpu_index);
}


YOLODLL_API Detector::~Detector() 
{
	detector_gpu_t &detector_gpu = *reinterpret_cast<detector_gpu_t *>(detector_gpu_ptr.get());
	layer l = detector_gpu.net.layers[detector_gpu.net.n - 1];

	free(detector_gpu.avg);
	for (int j = 0; j < FRAMES; ++j) free(detector_gpu.predictions[j]);
	for (int j = 0; j < FRAMES; ++j) if(detector_gpu.images[j].data) free(detector_gpu.images[j].data);

	for (int j = 0; j < l.w*l.h*l.n; ++j) free(detector_gpu.probs[j]);
	free(detector_gpu.boxes);
	free(detector_gpu.probs);

	int old_gpu_index;
	cudaGetDevice(&old_gpu_index);
	cudaSetDevice(detector_gpu.net.gpu_index);

	free_network(detector_gpu.net);

	cudaSetDevice(old_gpu_index);
}


YOLODLL_API std::vector<bbox_t> Detector::detect(std::string image_filename, float thresh)
{
	std::shared_ptr<image_t> image_ptr(new image_t, [](image_t *img) { if (img->data) free(img->data); delete img; });
	*image_ptr = load_image(image_filename);
	return detect(*image_ptr, thresh);
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

YOLODLL_API image_t Detector::load_image(std::string image_filename)
{
	char *input = const_cast<char *>(image_filename.data());
	image im = load_image_stb(input, 3);

	image_t img;
	img.c = im.c;
	img.data = im.data;
	img.h = im.h;
	img.w = im.w;

	return img;
}


YOLODLL_API void Detector::free_image(image_t m)
{
	if (m.data) {
		free(m.data);
	}
}

YOLODLL_API std::vector<bbox_t> Detector::detect(image_t img, float thresh)
{

	detector_gpu_t &detector_gpu = *reinterpret_cast<detector_gpu_t *>(detector_gpu_ptr.get());
	network &net = detector_gpu.net;
	int old_gpu_index;
	cudaGetDevice(&old_gpu_index);
	cudaSetDevice(net.gpu_index);
	//std::cout << "net.gpu_index = " << net.gpu_index << std::endl;

	//float nms = .4;

	image im;
	im.c = img.c;
	im.data = img.data;
	im.h = img.h;
	im.w = img.w;

	image sized = resize_image(im, net.w, net.h);
	layer l = net.layers[net.n - 1];

	float *X = sized.data;

	network_predict(net, X);

	get_region_boxes(l, 1, 1, thresh, detector_gpu.probs, detector_gpu.boxes, 0, 0);
	if (nms) do_nms_sort(detector_gpu.boxes, detector_gpu.probs, l.w*l.h*l.n, l.classes, nms);
	//draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);

	std::vector<bbox_t> bbox_vec;

	for (size_t i = 0; i < (l.w*l.h*l.n); ++i) {
		box b = detector_gpu.boxes[i];
		int const obj_id = max_index(detector_gpu.probs[i], l.classes);
		float const prob = detector_gpu.probs[i][obj_id];
		
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

			bbox_vec.push_back(bbox);
		}
	}

	if(sized.data)
		free(sized.data);

	cudaSetDevice(old_gpu_index);

	return bbox_vec;
}

YOLODLL_API std::vector<bbox_t> Detector::tracking(std::vector<bbox_t> cur_bbox_vec, int const frames_story)
{
	bool prev_track_id_present = false;
	for (auto &i : prev_bbox_vec_deque)
		if (i.size() > 0) prev_track_id_present = true;

	static unsigned int track_id = 1;

	if (!prev_track_id_present) {
		//track_id = 1;
		for (size_t i = 0; i < cur_bbox_vec.size(); ++i)
			cur_bbox_vec[i].track_id = track_id++;
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
					unsigned int cur_dist = sqrt(((float)i.x - k.x)*((float)i.x - k.x) + ((float)i.y - k.y)*((float)i.y - k.y));
					if (cur_dist < 100 && (k.track_id == 0 || dist_vec[m] > cur_dist)) {
						dist_vec[m] = cur_dist;
						cur_index = m;
					}
				}
			}

			bool track_id_absent = !std::any_of(cur_bbox_vec.begin(), cur_bbox_vec.end(), [&](bbox_t const& b) { return b.track_id == i.track_id; });

			if (cur_index >= 0 && track_id_absent)
				cur_bbox_vec[cur_index].track_id = i.track_id;
		}
	}

	for (size_t i = 0; i < cur_bbox_vec.size(); ++i)
		if (cur_bbox_vec[i].track_id == 0)
			cur_bbox_vec[i].track_id = track_id++;

	prev_bbox_vec_deque.push_front(cur_bbox_vec);
	if (prev_bbox_vec_deque.size() > frames_story) prev_bbox_vec_deque.pop_back();

	return cur_bbox_vec;
}