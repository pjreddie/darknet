#ifndef CONNECTED_LAYER_H
#define CONNECTED_LAYER_H

#include "activations.h"
#include "params.h"
#include "layer.h"

typedef layer connected_layer;

connected_layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation);

void forward_connected_layer(connected_layer layer, network_state state);
void backward_connected_layer(connected_layer layer, network_state state);
void update_connected_layer(connected_layer layer, int batch, float learning_rate, float momentum, float decay);

#ifdef GPU
void forward_connected_layer_gpu(connected_layer layer, network_state state);
void backward_connected_layer_gpu(connected_layer layer, network_state state);
void update_connected_layer_gpu(connected_layer layer, int batch, float learning_rate, float momentum, float decay);
void push_connected_layer(connected_layer layer);
void pull_connected_layer(connected_layer layer);
#endif

#endif

