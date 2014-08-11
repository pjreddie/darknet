#ifndef CROP_LAYER_H
#define CROP_LAYER_H

#include "image.h"

typedef struct {
    int batch;
    int h,w,c;
    int crop_width;
    int crop_height;
    int flip;
    float *delta;
    float *output;
} crop_layer;

image get_crop_image(crop_layer layer);
crop_layer *make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip);
void forward_crop_layer(const crop_layer layer, float *input);
void backward_crop_layer(const crop_layer layer, float *input, float *delta);

#endif

