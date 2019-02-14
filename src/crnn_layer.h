
#ifndef CRNN_LAYER_H
#define CRNN_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int steps, int size, int stride, int pad, ACTIVATION activation, int batch_normalize);

void forward_crnn_layer(layer l, network_state state);
void backward_crnn_layer(layer l, network_state state);
void update_crnn_layer(layer l, int batch, float learning_rate, float momentum, float decay);

#ifdef GPU
void forward_crnn_layer_gpu(layer l, network_state state);
void backward_crnn_layer_gpu(layer l, network_state state);
void update_crnn_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay);
void push_crnn_layer(layer l);
void pull_crnn_layer(layer l);
#endif

#ifdef __cplusplus
}
#endif

#endif
