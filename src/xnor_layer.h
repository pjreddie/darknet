#ifndef XNOR_LAYER_H
#define XNOR_LAYER_H

#include "layer.h"
#include "network.h"

layer make_xnor_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation, int batch_normalization);
void forward_xnor_layer(const layer l, network_state state);

#endif

