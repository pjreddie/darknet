#ifndef ROUTE_LAYER_H
#define ROUTE_LAYER_H
#include "network.h"

typedef struct {
    int batch;
    int outputs;
    int n;
    int   * input_layers;
    int   * input_sizes;
    float * delta;
    float * output;
    #ifdef GPU
    float * delta_gpu;
    float * output_gpu;
    #endif
} route_layer;

route_layer *make_route_layer(int batch, int n, int *input_layers, int *input_size);
void forward_route_layer(const route_layer layer, network net);
void backward_route_layer(const route_layer layer, network net);

#ifdef GPU
void forward_route_layer_gpu(const route_layer layer, network net);
void backward_route_layer_gpu(const route_layer layer, network net);
#endif

#endif
