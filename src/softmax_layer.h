#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

typedef struct {
    int inputs;
    int batch;
    int groups;
    float *delta;
    float *output;
    #ifdef GPU
    float * delta_gpu;
    float * output_gpu;
    #endif
} softmax_layer;

softmax_layer *make_softmax_layer(int batch, int groups, int inputs);
void forward_softmax_layer(const softmax_layer layer, float *input);
void backward_softmax_layer(const softmax_layer layer, float *delta);

#ifdef GPU
void pull_softmax_layer_output(const softmax_layer layer);
void forward_softmax_layer_gpu(const softmax_layer layer, float *input);
void backward_softmax_layer_gpu(const softmax_layer layer, float *delta);
#endif

#endif
