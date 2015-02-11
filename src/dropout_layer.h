#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

typedef struct{
    int batch;
    int inputs;
    float probability;
    float scale;
    float *rand;
    float *output;
    #ifdef GPU
    float * rand_gpu;
    float * output_gpu;
    #endif
} dropout_layer;

dropout_layer *make_dropout_layer(int batch, int inputs, float probability);

void forward_dropout_layer(dropout_layer layer, float *input);
void backward_dropout_layer(dropout_layer layer, float *delta);
void resize_dropout_layer(dropout_layer *layer, int inputs);

#ifdef GPU
void forward_dropout_layer_gpu(dropout_layer layer, float * input);
void backward_dropout_layer_gpu(dropout_layer layer, float * delta);

#endif
#endif
