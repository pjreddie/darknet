#ifndef ODLA_LAYER_H
#define ODLA_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer odla_layer;

odla_layer make_odla_layer(int batch, int w, int h, int c,
                            odla_params params);
void forward_odla_layer(const layer l, network net);
void backward_odla_layer(const layer l, network net);
void resize_odla_layer(layer *l, int w, int h);

#endif
