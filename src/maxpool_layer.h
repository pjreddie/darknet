#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "image.h"

typedef struct {
    int batch;
    int h,w,c;
    int stride;
    int size;
    int *indexes;
    float *delta;
    float *output;
} maxpool_layer;

image get_maxpool_image(maxpool_layer layer);
maxpool_layer *make_maxpool_layer(int batch, int h, int w, int c, int size, int stride);
void resize_maxpool_layer(maxpool_layer *layer, int h, int w, int c);
void forward_maxpool_layer(const maxpool_layer layer, float *input);
void backward_maxpool_layer(const maxpool_layer layer, float *input, float *delta);

#endif

