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

convolutional_layer *make_convolutional_layer(int h, int w, int c, int n, int size, int stride);
void run_convolutional_layer(const image input, const convolutional_layer layer);
void learn_convolutional_layer(image input, convolutional_layer layer);
void update_convolutional_layer(convolutional_layer layer, double step);
void backpropagate_convolutional_layer(image input, convolutional_layer layer);
void backpropagate_convolutional_layer_convolve(image input, convolutional_layer layer);

#endif

