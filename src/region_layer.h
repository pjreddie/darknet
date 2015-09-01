#ifndef REGION_LAYER_H
#define REGION_LAYER_H

#include "params.h"
#include "layer.h"

typedef layer region_layer;

region_layer make_region_layer(int batch, int inputs, int n, int size, int classes, int coords, int rescore);
void forward_region_layer(const region_layer l, network_state state);
void backward_region_layer(const region_layer l, network_state state);

#ifdef GPU
void forward_region_layer_gpu(const region_layer l, network_state state);
void backward_region_layer_gpu(region_layer l, network_state state);
#endif

#endif
