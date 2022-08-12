
#ifndef CRNN_LAYER_H
#define CRNN_LAYER_H

#include "activations.h"
#include "layer.h"
#include "network.h"

dn_layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int steps, ACTIVATION activation, int batch_normalize);

void forward_crnn_layer(dn_layer l, dn_network net);
void backward_crnn_layer(dn_layer l, dn_network net);
void update_crnn_layer(dn_layer l, update_args a);

#ifdef GPU
void forward_crnn_layer_gpu(layer l, network net);
void backward_crnn_layer_gpu(layer l, network net);
void update_crnn_layer_gpu(layer l, update_args a);
void push_crnn_layer(layer l);
void pull_crnn_layer(layer l);
#endif

#endif

