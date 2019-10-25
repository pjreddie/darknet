//Gaussian YOLOv3 implementation
#ifndef GAUSSIAN_YOLO_LAYER_H
#define GAUSSIAN_YOLO_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_gaussian_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes);
void forward_gaussian_yolo_layer(const layer l, network net);
void backward_gaussian_yolo_layer(const layer l, network net);
void resize_gaussian_yolo_layer(layer *l, int w, int h);
int gaussian_yolo_num_detections(layer l, float thresh);

#ifdef GPU
void forward_gaussian_yolo_layer_gpu(const layer l, network net);
void backward_gaussian_yolo_layer_gpu(layer l, network net);
#endif

#endif
