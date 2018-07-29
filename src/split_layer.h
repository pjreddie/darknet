#ifndef SPLIT_LAYER_H
#define SPLIT_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer split_layer;

split_layer make_split_layer(network *net, int batch, int w, int h, int c, int input_layer, int tensor);
void forward_split_layer(const layer l, network net);
void backward_split_layer(const layer l, network net);
void resize_split_layer(layer *l, int w, int h);

#endif
