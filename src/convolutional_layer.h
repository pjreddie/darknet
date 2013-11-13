#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "image.h"
#include "activations.h"

typedef struct {
    int h,w,c;
    int n;
    int stride;
    image *kernels;
    image *kernel_updates;
    double *biases;
    double *bias_updates;
    image upsampled;
    double *delta;
    double *output;

    double (* activation)();
    double (* gradient)();
} convolutional_layer;

convolutional_layer *make_convolutional_layer(int h, int w, int c, int n, int size, int stride, ACTIVATION activator);
void forward_convolutional_layer(const convolutional_layer layer, double *in);
void backward_convolutional_layer(convolutional_layer layer, double *input, double *delta);
void learn_convolutional_layer(convolutional_layer layer, double *input);

void update_convolutional_layer(convolutional_layer layer, double step);

void backpropagate_convolutional_layer_convolve(image input, convolutional_layer layer);
void visualize_convolutional_layer(convolutional_layer layer);

image get_convolutional_image(convolutional_layer layer);
image get_convolutional_delta(convolutional_layer layer);

#endif

