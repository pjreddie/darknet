#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include "activations.h"

struct layer;
typedef struct layer layer;

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
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    CRNN
} LAYER_TYPE;

typedef enum{
    SSE, MASKED, SMOOTH
} COST_TYPE;

struct layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int truths;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int groups;
    int size;
    int side;
    int stride;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int steps;
    int hidden;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int does_cost;
    int joint;
    int noadjust;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float class_scale;

    int dontload;
    int dontloadscales;

    float temperature;
    float probability;
    float scale;

    int *indexes;
    float *rand;
    float *cost;
    float *filters;
    char  *cfilters;
    float *filter_updates;
    float *state;

    float *binary_filters;

    float *biases;
    float *bias_updates;

    float *scales;
    float *scale_updates;

    float *weights;
    float *weight_updates;

    float *col_image;
    int   * input_layers;
    int   * input_sizes;
    float * delta;
    float * output;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    #ifdef GPU
    int *indexes_gpu;
    float * state_gpu;
    float * filters_gpu;
    float * filter_updates_gpu;

    float *binary_filters_gpu;
    float *mean_filters_gpu;

    float * spatial_mean_gpu;
    float * spatial_variance_gpu;

    float * mean_gpu;
    float * variance_gpu;

    float * rolling_mean_gpu;
    float * rolling_variance_gpu;

    float * spatial_mean_delta_gpu;
    float * spatial_variance_delta_gpu;

    float * variance_delta_gpu;
    float * mean_delta_gpu;

    float * col_image_gpu;

    float * x_gpu;
    float * x_norm_gpu;
    float * weights_gpu;
    float * weight_updates_gpu;

    float * biases_gpu;
    float * bias_updates_gpu;

    float * scales_gpu;
    float * scale_updates_gpu;

    float * output_gpu;
    float * delta_gpu;
    float * rand_gpu;
    float * squared_gpu;
    float * norms_gpu;
    #endif
};

void free_layer(layer);

#endif
