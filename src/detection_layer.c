#include "detection_layer.h"
#include "activations.h"
#include "softmax_layer.h"
#include "blas.h"
#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>

int get_detection_layer_locations(detection_layer layer)
{
    return layer.inputs / (layer.classes+layer.coords+layer.rescore);
}

int get_detection_layer_output_size(detection_layer layer)
{
    return get_detection_layer_locations(layer)*(layer.classes+layer.coords);
}

detection_layer *make_detection_layer(int batch, int inputs, int classes, int coords, int rescore)
{
    detection_layer *layer = calloc(1, sizeof(detection_layer));

    layer->batch = batch;
    layer->inputs = inputs;
    layer->classes = classes;
    layer->coords = coords;
    layer->rescore = rescore;
    int outputs = get_detection_layer_output_size(*layer);
    layer->output = calloc(batch*outputs, sizeof(float));
    layer->delta = calloc(batch*outputs, sizeof(float));
    #ifdef GPU
    layer->output_gpu = cuda_make_array(0, batch*outputs);
    layer->delta_gpu = cuda_make_array(0, batch*outputs);
    #endif

    fprintf(stderr, "Detection Layer\n");
    srand(0);

    return layer;
}

void forward_detection_layer(const detection_layer layer, float *in, float *truth)
{
    int in_i = 0;
    int out_i = 0;
    int locations = get_detection_layer_locations(layer);
    int i,j;
    for(i = 0; i < layer.batch*locations; ++i){
        int mask = (!truth || !truth[out_i + layer.classes - 1]);
        float scale = 1;
        if(layer.rescore) scale = in[in_i++];
        for(j = 0; j < layer.classes; ++j){
            layer.output[out_i++] = scale*in[in_i++];
        }
        softmax_array(layer.output + out_i - layer.classes, layer.classes, layer.output + out_i - layer.classes);
        activate_array(in+in_i, layer.coords, SIGMOID);
        for(j = 0; j < layer.coords; ++j){
            layer.output[out_i++] = mask*in[in_i++];
        }
    }
}

void backward_detection_layer(const detection_layer layer, float *in, float *delta)
{
    int locations = get_detection_layer_locations(layer);
    int i,j;
    int in_i = 0;
    int out_i = 0;
    for(i = 0; i < layer.batch*locations; ++i){
        float scale = 1;
        float latent_delta = 0;
        if(layer.rescore) scale = in[in_i++];
        for(j = 0; j < layer.classes; ++j){
            latent_delta += in[in_i]*layer.delta[out_i];
            delta[in_i++] = scale*layer.delta[out_i++];
        }
        
        gradient_array(layer.output + out_i, layer.coords, SIGMOID, layer.delta + out_i);
        for(j = 0; j < layer.coords; ++j){
            delta[in_i++] = layer.delta[out_i++];
        }
        if(layer.rescore) delta[in_i-layer.coords-layer.classes-layer.rescore] = latent_delta;
    }
}

#ifdef GPU

void forward_detection_layer_gpu(const detection_layer layer, float *in, float *truth)
{
    int outputs = get_detection_layer_output_size(layer);
    float *in_cpu = calloc(layer.batch*layer.inputs, sizeof(float));
    float *truth_cpu = 0;
    if(truth){
        truth_cpu = calloc(layer.batch*outputs, sizeof(float));
        cuda_pull_array(truth, truth_cpu, layer.batch*outputs);
    }
    cuda_pull_array(in, in_cpu, layer.batch*layer.inputs);
    forward_detection_layer(layer, in_cpu, truth_cpu);
    cuda_push_array(layer.output_gpu, layer.output, layer.batch*outputs);
    free(in_cpu);
    if(truth_cpu) free(truth_cpu);
}

void backward_detection_layer_gpu(detection_layer layer, float *in, float *delta)
{
    int outputs = get_detection_layer_output_size(layer);

    float *in_cpu =    calloc(layer.batch*layer.inputs, sizeof(float));
    float *delta_cpu = calloc(layer.batch*layer.inputs, sizeof(float));

    cuda_pull_array(in, in_cpu, layer.batch*layer.inputs);
    cuda_pull_array(layer.delta_gpu, layer.delta, layer.batch*outputs);
    backward_detection_layer(layer, in_cpu, delta_cpu);
    cuda_push_array(delta, delta_cpu, layer.batch*layer.inputs);

    free(in_cpu);
    free(delta_cpu);
}
#endif

