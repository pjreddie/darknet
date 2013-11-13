#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "image.h"

typedef struct {
    int h,w,c;
    int stride;
    double *delta;
    double *output;
} maxpool_layer;

image get_maxpool_image(maxpool_layer layer);
maxpool_layer *make_maxpool_layer(int h, int w, int c, int stride);
void forward_maxpool_layer(const maxpool_layer layer, double *in);

#endif

