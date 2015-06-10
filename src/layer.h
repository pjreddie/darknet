#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include "activations.h"

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST
} LAYER_TYPE;

typedef enum{
    SSE, MASKED
} COST_TYPE;

typedef struct {
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    int batch;
    int inputs;
    int outputs;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int groups;
    int size;
    int stride;
    int pad;
    int crop_width;
    int crop_height;
    int flip;
    float angle;
    float saturation;
    float exposure;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int does_cost;
    int joint;

    float probability;
    float scale;
    int *indexes;
    float *rand;
    float *cost;
    float *filters;
    float *filter_updates;

    float *biases;
    float *bias_updates;

    float *weights;
    float *weight_updates;

    float *col_image;
    int   * input_layers;
    int   * input_sizes;
    float * delta;
    float * output;

    #ifdef GPU
    int *indexes_gpu;
    float * filters_gpu;
    float * filter_updates_gpu;

    float * col_image_gpu;

    float * weights_gpu;
    float * biases_gpu;

    float * weight_updates_gpu;
    float * bias_updates_gpu;

    float * output_gpu;
    float * delta_gpu;
    float * rand_gpu;
    #endif
} layer;

#endif
