#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

typedef struct {
    int inputs;
    double *delta;
    double *output;
} softmax_layer;

softmax_layer *make_softmax_layer(int inputs);
void forward_softmax_layer(const softmax_layer layer, double *input);
void backward_softmax_layer(const softmax_layer layer, double *input, double *delta);

#endif
