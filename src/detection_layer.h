#ifndef DETECTION_LAYER_H
#define DETECTION_LAYER_H

#include "params.h"
#include "layer.h"

typedef layer detection_layer;

detection_layer make_detection_layer(int batch, int inputs, int classes, int coords, int joint, int rescore, int background, int objectness);
void forward_detection_layer(const detection_layer l, network_state state);
void backward_detection_layer(const detection_layer l, network_state state);
int get_detection_layer_output_size(detection_layer l);
int get_detection_layer_locations(detection_layer l);

#ifdef GPU
void forward_detection_layer_gpu(const detection_layer l, network_state state);
void backward_detection_layer_gpu(detection_layer l, network_state state);
#endif

#endif
