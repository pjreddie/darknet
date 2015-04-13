#ifndef CROP_LAYER_H
#define CROP_LAYER_H

#include "image.h"
#include "params.h"

typedef struct {
    int batch;
    int h,w,c;
    int crop_width;
    int crop_height;
    int flip;
    float angle;
    float *output;
#ifdef GPU
    float *output_gpu;
#endif
} crop_layer;

image get_crop_image(crop_layer layer);
crop_layer *make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle);
void forward_crop_layer(const crop_layer layer, network_state state);

#ifdef GPU
void forward_crop_layer_gpu(crop_layer layer, network_state state);
#endif

#endif

