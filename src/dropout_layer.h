#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

typedef struct{
    int batch;
    int inputs;
    float probability;
} dropout_layer;

dropout_layer *make_dropout_layer(int batch, int inputs, float probability);

void forward_dropout_layer(dropout_layer layer, float *input);
void backward_dropout_layer(dropout_layer layer, float *input, float *delta);

#endif
