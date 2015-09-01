#include "layer.h"
#include "cuda.h"
#include <stdlib.h>

void free_layer(layer l)
{
    if(l.type == DROPOUT){
        if(l.rand)           free(l.rand);
#ifdef GPU
        if(l.rand_gpu)             cuda_free(l.rand_gpu);
#endif
        return;
    }
    if(l.indexes)        free(l.indexes);
    if(l.rand)           free(l.rand);
    if(l.cost)           free(l.cost);
    if(l.filters)        free(l.filters);
    if(l.filter_updates) free(l.filter_updates);
    if(l.biases)         free(l.biases);
    if(l.bias_updates)   free(l.bias_updates);
    if(l.weights)        free(l.weights);
    if(l.weight_updates) free(l.weight_updates);
    if(l.col_image)      free(l.col_image);
    if(l.input_layers)   free(l.input_layers);
    if(l.input_sizes)    free(l.input_sizes);
    if(l.delta)          free(l.delta);
    if(l.output)         free(l.output);
    if(l.squared)        free(l.squared);
    if(l.norms)          free(l.norms);

#ifdef GPU
    if(l.indexes_gpu)          cuda_free((float *)l.indexes_gpu);
    if(l.filters_gpu)          cuda_free(l.filters_gpu);
    if(l.filter_updates_gpu)   cuda_free(l.filter_updates_gpu);
    if(l.col_image_gpu)        cuda_free(l.col_image_gpu);
    if(l.weights_gpu)          cuda_free(l.weights_gpu);
    if(l.biases_gpu)           cuda_free(l.biases_gpu);
    if(l.weight_updates_gpu)   cuda_free(l.weight_updates_gpu);
    if(l.bias_updates_gpu)     cuda_free(l.bias_updates_gpu);
    if(l.output_gpu)           cuda_free(l.output_gpu);
    if(l.delta_gpu)            cuda_free(l.delta_gpu);
    if(l.rand_gpu)             cuda_free(l.rand_gpu);
    if(l.squared_gpu)          cuda_free(l.squared_gpu);
    if(l.norms_gpu)            cuda_free(l.norms_gpu);
#endif
}
