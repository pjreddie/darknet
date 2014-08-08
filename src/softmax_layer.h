#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

typedef struct {
    int inputs;
    int batch;
    float *delta;
    float *output;
    float *jacobian;
} softmax_layer;

softmax_layer *make_softmax_layer(int batch, int inputs);
void forward_softmax_layer(const softmax_layer layer, float *input);
void backward_softmax_layer(const softmax_layer layer, float *input, float *delta);

#endif
