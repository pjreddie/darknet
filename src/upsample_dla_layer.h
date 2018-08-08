#ifndef UPSAMPLE_DLA_LAYER_H
#define UPSAMPLE_DLA_LAYER_H
#include "darknet.h"

#define ATOMIC_CUBE 32

layer make_upsample_dla_layer(int batch, int w, int h, int c, int stride);
void forward_upsample_dla_layer(const layer l, network net);

#endif
