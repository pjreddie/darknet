#ifndef DECONVOLUTIONAL_LAYER_H
#define DECONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "params.h"
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

    float *biases;
    float *bias_updates;

    float *col_image;
    float *delta;
    float *output;

    #ifdef GPU
    float * filters_gpu;
    float * filter_updates_gpu;

    float * biases_gpu;
    float * bias_updates_gpu;

    float * col_image_gpu;
    float * delta_gpu;
    float * output_gpu;
    #endif

    ACTIVATION activation;
} deconvolutional_layer;

#ifdef GPU
void forward_deconvolutional_layer_gpu(deconvolutional_layer layer, network_state state);
void backward_deconvolutional_layer_gpu(deconvolutional_layer layer, network_state state);
void update_deconvolutional_layer_gpu(deconvolutional_layer layer, float learning_rate, float momentum, float decay);
void push_deconvolutional_layer(deconvolutional_layer layer);
void pull_deconvolutional_layer(deconvolutional_layer layer);
#endif

deconvolutional_layer *make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, ACTIVATION activation);
void resize_deconvolutional_layer(deconvolutional_layer *layer, int h, int w);
void forward_deconvolutional_layer(const deconvolutional_layer layer, network_state state);
void update_deconvolutional_layer(deconvolutional_layer layer, float learning_rate, float momentum, float decay);
void backward_deconvolutional_layer(deconvolutional_layer layer, network_state state);

image get_deconvolutional_image(deconvolutional_layer layer);
image get_deconvolutional_delta(deconvolutional_layer layer);
image get_deconvolutional_filter(deconvolutional_layer layer, int i);

int deconvolutional_out_height(deconvolutional_layer layer);
int deconvolutional_out_width(deconvolutional_layer layer);

#endif

