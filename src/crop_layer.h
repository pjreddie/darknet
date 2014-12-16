#ifndef CROP_LAYER_H
#define CROP_LAYER_H

#include "opencl.h"
#include "image.h"

typedef struct {
    int batch;
    int h,w,c;
    int crop_width;
    int crop_height;
    int flip;
    float *output;
#ifdef GPU
    cl_mem output_cl;
#endif
} crop_layer;

image get_crop_image(crop_layer layer);
crop_layer *make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip);
void forward_crop_layer(const crop_layer layer, float *input);

#ifdef GPU
void forward_crop_layer_gpu(crop_layer layer, cl_mem input);
#endif

#endif

