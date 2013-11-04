#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "image.h"

typedef struct {
    int n;
    int stride;
    image *kernels;
    image *kernel_updates;
    image upsampled;
    image output;
} convolutional_layer;

convolutional_layer make_convolutional_layer(int w, int h, int c, int n, int size, int stride);
void run_convolutional_layer(const image input, const convolutional_layer layer);
void backpropagate_layer(image input, convolutional_layer layer);
void backpropagate_layer_convolve(image input, convolutional_layer layer);

#endif

