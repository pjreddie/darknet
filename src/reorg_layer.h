#ifndef REORG_LAYER_H
#define REORG_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra);
void resize_reorg_layer(layer *l, int w, int h);
void forward_reorg_layer(const layer l, network_state state);
void backward_reorg_layer(const layer l, network_state state);

#ifdef GPU
void forward_reorg_layer_gpu(layer l, network_state state);
void backward_reorg_layer_gpu(layer l, network_state state);
#endif

#endif

