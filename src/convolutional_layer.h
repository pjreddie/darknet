#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "image.h"
#include "activations.h"

typedef struct {
    int batch;
    int h,w,c;
    int n;
    int size;
    int stride;
    float *filters;
    float *filter_updates;
    float *filter_momentum;

    float *biases;
    float *bias_updates;
    float *bias_momentum;

    float *col_image;
    float *delta;
    float *output;

    ACTIVATION activation;
} convolutional_layer;

convolutional_layer *make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, ACTIVATION activation);
void resize_convolutional_layer(convolutional_layer *layer, int h, int w, int c);
void forward_convolutional_layer(const convolutional_layer layer, float *in);
void learn_convolutional_layer(convolutional_layer layer);
void update_convolutional_layer(convolutional_layer layer, float step, float momentum, float decay);
void visualize_convolutional_layer(convolutional_layer layer, char *window);

void backward_convolutional_layer(convolutional_layer layer, float *delta);

//void backpropagate_convolutional_layer_convolve(image input, convolutional_layer layer);
//void visualize_convolutional_filters(convolutional_layer layer, char *window);
//void visualize_convolutional_layer(convolutional_layer layer);

image get_convolutional_image(convolutional_layer layer);
image get_convolutional_delta(convolutional_layer layer);
image get_convolutional_filter(convolutional_layer layer, int i);

#endif

