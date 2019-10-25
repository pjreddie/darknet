//Gaussian YOLOv3 implementation
#ifndef GAUSSIAN_YOLO_LAYER_H
#define GAUSSIAN_YOLO_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_gaussian_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes);
void forward_gaussian_yolo_layer(const layer l, network_state state);
void backward_gaussian_yolo_layer(const layer l, network_state state);
void resize_gaussian_yolo_layer(layer *l, int w, int h);
int gaussian_yolo_num_detections(layer l, float thresh);
int get_gaussian_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter);
void correct_gaussian_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter);

#ifdef GPU
void forward_gaussian_yolo_layer_gpu(const layer l, network_state state);
void backward_gaussian_yolo_layer_gpu(layer l, network_state state);
#endif

#endif
