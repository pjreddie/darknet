#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#ifdef GPU
#include "opencl.h"
#endif

#include "image.h"
#include "activations.h"

typedef struct {
    float learning_rate;
    float momentum;
    float decay;

    int batch;
    int h,w,c;
    int n;
    int size;
    int stride;
    int pad;
    float *filters;
    float *filter_updates;
    float *filter_momentum;

    float *biases;
    float *bias_updates;
    float *bias_momentum;

    float *col_image;
    float *delta;
    float *output;

    #ifdef GPU
    cl_mem filters_cl;
    cl_mem filter_updates_cl;
    cl_mem filter_momentum_cl;

    cl_mem biases_cl;
    cl_mem bias_updates_cl;
    cl_mem bias_momentum_cl;

    cl_mem col_image_cl;
    cl_mem delta_cl;
    cl_mem output_cl;
    #endif

    ACTIVATION activation;
} convolutional_layer;

#ifdef GPU
void forward_convolutional_layer_gpu(convolutional_layer layer, cl_mem in);
#endif

convolutional_layer *make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation, float learning_rate, float momentum, float decay);
void resize_convolutional_layer(convolutional_layer *layer, int h, int w, int c);
void forward_convolutional_layer(const convolutional_layer layer, float *in);
void update_convolutional_layer(convolutional_layer layer);
image *visualize_convolutional_layer(convolutional_layer layer, char *window, image *prev_filters);

void backward_convolutional_layer(convolutional_layer layer, float *delta);

image get_convolutional_image(convolutional_layer layer);
image get_convolutional_delta(convolutional_layer layer);
image get_convolutional_filter(convolutional_layer layer, int i);

#endif

