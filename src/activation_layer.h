#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_activation_layer(int batch, int inputs, ACTIVATION activation);

void forward_activation_layer(layer l, network_state state);
void backward_activation_layer(layer l, network_state state);

#ifdef GPU
void forward_activation_layer_gpu(layer l, network_state state);
void backward_activation_layer_gpu(layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif

#endif
