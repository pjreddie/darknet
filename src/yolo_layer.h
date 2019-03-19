#ifndef YOLO_LAYER_H
#define YOLO_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

dn_layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes);
void forward_yolo_layer(const dn_layer l, dn_network net);
void backward_yolo_layer(const dn_layer l, dn_network net);
void resize_yolo_layer(dn_layer *l, int w, int h);
int yolo_num_detections(dn_layer l, float thresh);

#ifdef GPU
void forward_yolo_layer_gpu(const layer l, network net);
void backward_yolo_layer_gpu(layer l, network net);
#endif

#endif
