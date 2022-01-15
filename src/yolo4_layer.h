#ifndef YOLO4_LAYER_H
#define YOLO4_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

layer make_yolo4_layer(int batch, int w, int h, int n, int total, int *mask, int classes, int max_boxes);
void forward_yolo4_layer(const layer l, network net);
void backward_yolo4_layer(const layer l, network net);
void resize_yolo4_layer(layer *l, int w, int h);
int yolo4_num_detections(layer l, float thresh);
int yolo4_num_detections_batch(layer l, float thresh, int batch);
int get_yolo4_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter);
int get_yolo4_detections_batch(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter, int batch);
void correct_yolo4_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative, int letter);

#ifdef GPU
void forward_yolo4_layer_gpu(const layer l, network net);
void backward_yolo4_layer_gpu(layer l, network net);
#endif

#endif
