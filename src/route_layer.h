#ifndef ROUTE_LAYER_H
#define ROUTE_LAYER_H
#include "network.h"
#include "layer.h"

typedef layer route_layer;

#ifdef __cplusplus
extern "C" {
#endif
route_layer make_route_layer(int batch, int n, int *input_layers, int *input_size, int groups, int group_id);
void forward_route_layer(const route_layer l, network_state state);
void backward_route_layer(const route_layer l, network_state state);
void resize_route_layer(route_layer *l, network *net);

#ifdef GPU
void forward_route_layer_gpu(const route_layer l, network_state state);
void backward_route_layer_gpu(const route_layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
#endif
