#ifndef ISEG_LAYER_H
#define ISEG_LAYER_H

#include "darknet.h"
#include "layer.h"
#include "network.h"

dn_layer make_iseg_layer(int batch, int w, int h, int classes, int ids);
void forward_iseg_layer(const dn_layer l, dn_network net);
void backward_iseg_layer(const dn_layer l, dn_network net);
void resize_iseg_layer(dn_layer *l, int w, int h);
int iseg_num_detections(dn_layer l, float thresh);

#ifdef GPU
void forward_iseg_layer_gpu(const layer l, network net);
void backward_iseg_layer_gpu(layer l, network net);
#endif

#endif
