#ifndef DETECTION_LAYER_H
#define DETECTION_LAYER_H

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

#endif
