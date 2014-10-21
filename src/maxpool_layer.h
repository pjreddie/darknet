#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "image.h"
#include "opencl.h"

typedef struct {
    int batch;
    int h,w,c;
    int stride;
    int size;
    int *indexes;
    float *delta;
    float *output;
    #ifdef GPU
    cl_mem indexes_cl;
    cl_mem output_cl;
    cl_mem delta_cl;
    #endif
} maxpool_layer;

image get_maxpool_image(maxpool_layer layer);
maxpool_layer *make_maxpool_layer(int batch, int h, int w, int c, int size, int stride);
void resize_maxpool_layer(maxpool_layer *layer, int h, int w, int c);
void forward_maxpool_layer(const maxpool_layer layer, float *input);
void backward_maxpool_layer(const maxpool_layer layer, float *delta);

#ifdef GPU
void forward_maxpool_layer_gpu(maxpool_layer layer, cl_mem input);
void backward_maxpool_layer_gpu(maxpool_layer layer, cl_mem delta);
#endif

#endif

