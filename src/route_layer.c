#include "route_layer.h"
#include "cuda.h"
#include "blas.h"
#include <stdio.h>

route_layer *make_route_layer(int batch, int n, int *input_layers, int *input_sizes)
{
    printf("Route Layer:");
    route_layer *layer = calloc(1, sizeof(route_layer));
    layer->batch = batch;
    layer->n = n;
    layer->input_layers = input_layers;
    layer->input_sizes = input_sizes;
    int i;
    int outputs = 0;
    for(i = 0; i < n; ++i){
        printf(" %d", input_layers[i]);
        outputs += input_sizes[i];
    }
    printf("\n");
    layer->outputs = outputs;
    layer->delta = calloc(outputs*batch, sizeof(float));
    layer->output = calloc(outputs*batch, sizeof(float));;
    #ifdef GPU
    layer->delta_gpu = cuda_make_array(0, outputs*batch);
    layer->output_gpu = cuda_make_array(0, outputs*batch);
    #endif
    return layer;
}

void forward_route_layer(const route_layer layer, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < layer.n; ++i){
        float *input = get_network_output_layer(net, layer.input_layers[i]);
        int input_size = layer.input_sizes[i];
        for(j = 0; j < layer.batch; ++j){
            copy_cpu(input_size, input + j*input_size, 1, layer.output + offset + j*layer.outputs, 1);
        }
        offset += input_size;
    }
}

void backward_route_layer(const route_layer layer, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < layer.n; ++i){
        float *delta = get_network_delta_layer(net, layer.input_layers[i]);
        int input_size = layer.input_sizes[i];
        for(j = 0; j < layer.batch; ++j){
            copy_cpu(input_size, layer.delta + offset + j*layer.outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
}

#ifdef GPU
void forward_route_layer_gpu(const route_layer layer, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < layer.n; ++i){
        float *input = get_network_output_gpu_layer(net, layer.input_layers[i]);
        int input_size = layer.input_sizes[i];
        for(j = 0; j < layer.batch; ++j){
            copy_ongpu(input_size, input + j*input_size, 1, layer.output_gpu + offset + j*layer.outputs, 1);
        }
        offset += input_size;
    }
}

void backward_route_layer_gpu(const route_layer layer, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < layer.n; ++i){
        float *delta = get_network_delta_gpu_layer(net, layer.input_layers[i]);
        int input_size = layer.input_sizes[i];
        for(j = 0; j < layer.batch; ++j){
            copy_ongpu(input_size, layer.delta_gpu + offset + j*layer.outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
}
#endif
