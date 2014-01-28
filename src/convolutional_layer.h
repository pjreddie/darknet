#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "image.h"
#include "activations.h"

typedef struct {
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int size;
    int stride;
    double *filters;
    double *filter_updates;
    double *filter_momentum;

    double *biases;
    double *bias_updates;
    double *bias_momentum;

    double *col_image;
    double *delta;
    double *output;

    ACTIVATION activation;
} convolutional_layer;

convolutional_layer *make_convolutional_layer(int h, int w, int c, int n, int size, int stride, ACTIVATION activation);
void forward_convolutional_layer(const convolutional_layer layer, double *in);
void learn_convolutional_layer(convolutional_layer layer);
void update_convolutional_layer(convolutional_layer layer, double step, double momentum, double decay);
void visualize_convolutional_layer(convolutional_layer layer, char *window);

//void backward_convolutional_layer(convolutional_layer layer, double *input, double *delta);

//void backpropagate_convolutional_layer_convolve(image input, convolutional_layer layer);
//void visualize_convolutional_filters(convolutional_layer layer, char *window);
//void visualize_convolutional_layer(convolutional_layer layer);

image get_convolutional_image(convolutional_layer layer);
image get_convolutional_delta(convolutional_layer layer);
image get_convolutional_filter(convolutional_layer layer, int i);

#endif

