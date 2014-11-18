#include "freeweight_layer.h"
#include "stdlib.h"
#include "stdio.h"

freeweight_layer *make_freeweight_layer(int batch, int inputs)
{
    fprintf(stderr, "Freeweight Layer: %d inputs\n", inputs);
    freeweight_layer *layer = calloc(1, sizeof(freeweight_layer));
    layer->inputs = inputs;
    layer->batch = batch;
    return layer;
} 

void forward_freeweight_layer(freeweight_layer layer, float *input)
{
    int i;
    for(i = 0; i < layer.batch * layer.inputs; ++i){
        input[i] *= 2.*((float)rand()/RAND_MAX);
    }
}

void backward_freeweight_layer(freeweight_layer layer, float *input, float *delta)
{
    // Don't do shit LULZ
}
