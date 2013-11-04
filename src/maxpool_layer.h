#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "image.h"

typedef struct {
    int stride;
    image output;
} maxpool_layer;

maxpool_layer make_maxpool_layer(int h, int w, int c, int stride);
void run_maxpool_layer(const image input, const maxpool_layer layer);

#endif

