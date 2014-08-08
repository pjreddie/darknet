#include "dropout_layer.h"
#include "stdlib.h"
#include "stdio.h"

dropout_layer *make_dropout_layer(int batch, int inputs, float probability)
{
    fprintf(stderr, "Dropout Layer: %d inputs, %f probability\n", inputs, probability);
    dropout_layer *layer = calloc(1, sizeof(dropout_layer));
    layer->probability = probability;
    layer->inputs = inputs;
    layer->batch = batch;
    return layer;
} 

void forward_dropout_layer(dropout_layer layer, float *input)
{
    int i;
    for(i = 0; i < layer.batch * layer.inputs; ++i){
        if((float)rand()/RAND_MAX < layer.probability) input[i] = 0;
        else input[i] /= (1-layer.probability);
    }
}
void backward_dropout_layer(dropout_layer layer, float *input, float *delta)
{
    // Don't do shit LULZ
}
