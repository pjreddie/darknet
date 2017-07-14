//
// Created by frivas on 25/04/16.
//

#ifndef DARKNET_DARKNETAPI_H
#define DARKNET_DARKNETAPI_H


#include "parser.h"
#include "detection_layer.h"
#include "utils.h"
#include "yolo.h"
#include <DarknetAPI/DarknetDetection.h>

#include <vector>
#include <iostream>

extern "C" void c_test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh);
extern "C" network c_parse_network_cfg(char *filename);
extern "C" void c_load_weights(network *net, char *filename);
extern "C" void c_set_batch_network(network *net, int b);
extern "C" image c_load_image_color(char *filename, int w, int h);
extern "C" image c_resize_image(image im, int w, int h);
extern "C" float *c_network_predict(network net, float *input);
extern "C" float c_sec(clock_t clocks);
extern "C" void c_get_detection_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);
extern "C" void c_do_nms_sort(box *boxes, float **probs, int total, int classes, float thresh);
extern "C" void c_draw_detections(image im, int num, float thresh, box *boxes, float **probs, char **names, image *labels, int classes);
extern "C" void c_show_image(image p, const char *name);
extern "C" void c_save_image(image im, const char *name);
extern "C" void c_free_image(image m);
extern "C" float c_get_pixel(image m, int x, int y, int c);
extern "C" image c_make_image(int w, int h, int c);
extern "C" image c_copy_image(image p);
extern "C" void c_rgbgr_image(image im);
extern "C" void c_free_network(network net);
extern "C" int c_max_index(float *a, int n);
extern "C" void c_do_nms_obj(box *boxes, float **probs, int total, int classes, float thresh);
//extern "C" void c_get_region_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map, float tree_thresh);
extern "C" void c_get_region_boxes(layer l, int w, int h, int netw, int neth, float thresh, float **probs, box *boxes, float **masks, int only_objectness, int *map, float tree_thresh, int relative);





void addDetection(int num, float thresh, box *boxes, float **probs, char **names, image *labels, int classes, DarknetDetections& detections);
DarknetDetections processImageDetection(network& net, image& im, float thresh = 0.24);
DarknetDetections processImageDetection(network& net,const  cv::Mat & im, float thresh = 0.24);



class DarknetAPI {

    network net;


public:

     DarknetAPI(char *cfgfile, char *weightfile);

     ~DarknetAPI();

    DarknetDetections process(image& im, float thresh = 0.24);
    DarknetDetections process(const cv::Mat& im, float thresh = 0.24);
    std::string processToJson(const cv::Mat& im, float thresh = 0.24);
};



#endif //DARKNET_DARKNETAPI_H
