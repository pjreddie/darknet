#ifndef UPSAMPLE_DLA_LAYER_H
#define UPSAMPLE_DLA_LAYER_H
#include "darknet.h"

#define ATOMIC_CUBE 32

layer make_upsample_odla_layer(int batch, int w, int h, int c, int stride, int output_layer, int tensor);
void forward_upsample_odla_layer(const layer l, network net);

#endif
