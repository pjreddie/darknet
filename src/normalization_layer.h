#ifndef NORMALIZATION_LAYER_H
#define NORMALIZATION_LAYER_H

#include "image.h"
#include "layer.h"
#include "network.h"

dn_layer make_normalization_layer(int batch, int w, int h, int c, int size, float alpha, float beta, float kappa);
void resize_normalization_layer(dn_layer *layer, int h, int w);
void forward_normalization_layer(const dn_layer layer, dn_network net);
void backward_normalization_layer(const dn_layer layer, dn_network net);
void visualize_normalization_layer(dn_layer layer, char *window);

#ifdef GPU
void forward_normalization_layer_gpu(const layer layer, network net);
void backward_normalization_layer_gpu(const layer layer, network net);
#endif

#endif
