#ifndef LOGISTIC_LAYER_H
#define LOGISTIC_LAYER_H
#include "layer.h"
#include "network.h"

dn_layer make_logistic_layer(int batch, int inputs);
void forward_logistic_layer(const dn_layer l, dn_network net);
void backward_logistic_layer(const dn_layer l, dn_network net);

#ifdef GPU
void forward_logistic_layer_gpu(const layer l, network net);
void backward_logistic_layer_gpu(const layer l, network net);
#endif

#endif
