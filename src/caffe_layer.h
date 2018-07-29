#ifndef CAFFE_LAYER_H
#define CAFFE_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer caffe_layer;

caffe_layer make_caffe_layer(int batch, int w, int h, int c, const char *cfg, const char *weights);
void forward_caffe_layer(const layer l, network net);
void backward_caffe_layer(const layer l, network net);
void resize_caffe_layer(layer *l, int w, int h);

#endif
