#ifndef L2NORM_LAYER_H
#define L2NORM_LAYER_H
#include "layer.h"
#include "network.h"

layer make_l2norm_layer(int batch, int inputs);
void forward_l2norm_layer(const layer l, network net);
void backward_l2norm_layer(const layer l, network net);

#ifdef GPU
void forward_l2norm_layer_gpu(const layer l, network net);
void backward_l2norm_layer_gpu(const layer l, network net);
#endif

#endif
