#ifndef FREEWEIGHT_LAYER_H
#define FREEWEIGHT_LAYER_H

typedef struct{
    int batch;
    int inputs;
} freeweight_layer;

freeweight_layer *make_freeweight_layer(int batch, int inputs);

void forward_freeweight_layer(freeweight_layer layer, float *input);
void backward_freeweight_layer(freeweight_layer layer, float *input, float *delta);

#endif
