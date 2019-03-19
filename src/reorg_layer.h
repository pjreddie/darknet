#ifndef REORG_LAYER_H
#define REORG_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

dn_layer make_reorg_layer(int batch, int w, int h, int c, int stride, int reverse, int flatten, int extra);
void resize_reorg_layer(dn_layer *l, int w, int h);
void forward_reorg_layer(const dn_layer l, dn_network net);
void backward_reorg_layer(const dn_layer l, dn_network net);

#ifdef GPU
void forward_reorg_layer_gpu(layer l, network net);
void backward_reorg_layer_gpu(layer l, network net);
#endif

#endif

