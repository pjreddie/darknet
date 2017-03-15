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

}
//#include <sys/time.h>

#include <vector>
#include <iostream>


#define FRAMES 3
#define ROI_PER_DETECTOR 100


struct detector_gpu_t{
	float **probs;
	box *boxes;
	network net;
	//image det;
	//image det_s;
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

	free(detector_gpu.boxes);
	free(detector_gpu.avg);
	free(detector_gpu.predictions);
	for (int j = 0; j < l.w*l.h*l.n; ++j) free(detector_gpu.probs[j]);
	free(detector_gpu.probs);
}


YOLODLL_API std::vector<bbox_t> Detector::detect(std::string image_filename, float thresh)
{
	char *input = const_cast<char *>(image_filename.data());
	image im = load_image_color(input, 0, 0);

	image_t img;
	img.c = im.c;
	img.data = im.data;
	img.h = im.h;
	img.w = im.w;

	return detect(img, thresh);
}


YOLODLL_API std::vector<bbox_t> Detector::detect(image_t img, float thresh)
{

	detector_gpu_t &detector_gpu = *reinterpret_cast<detector_gpu_t *>(detector_gpu_ptr.get());
	network &net = detector_gpu.net;
	int old_gpu_index;
	cudaGetDevice(&old_gpu_index);
	cudaSetDevice(net.gpu_index);
	//std::cout << "net.gpu_index = " << net.gpu_index << std::endl;

	float nms = .4;

	image im;
	im.c = img.c;
	im.data = img.data;
	im.h = img.h;
	im.w = img.w;

	image sized = resize_image(im, net.w, net.h);
	layer l = net.layers[net.n - 1];

	//box *boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
	//float **probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
	// (int j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

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
			bbox.x = (b.x - b.w / 2.)*im.w;
			bbox.y = (b.y - b.h / 2.)*im.h;
			bbox.w = b.w*im.w;
			bbox.h = b.h*im.h;
			bbox.obj_id = obj_id;
			bbox.prob = prob;

			bbox_vec.push_back(bbox);
		}
	}

	cudaSetDevice(old_gpu_index);

	return bbox_vec;
}