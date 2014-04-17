#ifndef NORMALIZATION_LAYER_H
#define NORMALIZATION_LAYER_H

#include "image.h"

typedef struct {
    int batch;
    int h,w,c;
    int size;
    float alpha;
    float beta;
    float kappa;
    float *delta;
    float *output;
    float *sums;
} normalization_layer;

image get_normalization_image(normalization_layer layer);
normalization_layer *make_normalization_layer(int batch, int h, int w, int c, int size, float alpha, float beta, float kappa);
void resize_normalization_layer(normalization_layer *layer, int h, int w, int c);
void forward_normalization_layer(const normalization_layer layer, float *in);
void backward_normalization_layer(const normalization_layer layer, float *in, float *delta);
void visualize_normalization_layer(normalization_layer layer, char *window);

#endif

