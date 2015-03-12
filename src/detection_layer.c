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


void forward_detection_layer(const detection_layer layer, network_state state)
{
    int in_i = 0;
    int out_i = 0;
    int locations = get_detection_layer_locations(layer);
    int i,j;
    for(i = 0; i < layer.batch*locations; ++i){
        int mask = (!state.truth || state.truth[out_i + layer.classes + 2]);
        float scale = 1;
        if(layer.rescore) scale = state.input[in_i++];
        for(j = 0; j < layer.classes; ++j){
            layer.output[out_i++] = scale*state.input[in_i++];
        }
        if(!layer.rescore){
            softmax_array(layer.output + out_i - layer.classes, layer.classes, layer.output + out_i - layer.classes);
            activate_array(state.input+in_i, layer.coords, LOGISTIC);
        }
        for(j = 0; j < layer.coords; ++j){
            layer.output[out_i++] = mask*state.input[in_i++];
        }
    }
}

void dark_zone(detection_layer layer, int index, network_state state)
{
    int size = layer.classes+layer.rescore+layer.coords;
    int location = (index%(7*7*size)) / size ;
    int r = location / 7;
    int c = location % 7;
    int class = index%size;
    if(layer.rescore) --class;
    int dr, dc;
    for(dr = -1; dr <= 1; ++dr){
        for(dc = -1; dc <= 1; ++dc){
            if(!(dr || dc)) continue;
            if((r + dr) > 6 || (r + dr) < 0) continue;
            if((c + dc) > 6 || (c + dc) < 0) continue;
            int di = (dr*7 + dc) * size;
            if(state.truth[index+di]) continue;
            layer.delta[index + di] = 0;
        }
    }
}

void backward_detection_layer(const detection_layer layer, network_state state)
{
    int locations = get_detection_layer_locations(layer);
    int i,j;
    int in_i = 0;
    int out_i = 0;
    for(i = 0; i < layer.batch*locations; ++i){
        float scale = 1;
        float latent_delta = 0;
        if(layer.rescore) scale = state.input[in_i++];
        if(!layer.rescore){
            for(j = 0; j < layer.classes-1; ++j){
                if(state.truth[out_i + j]) dark_zone(layer, out_i+j, state);
            }
        }
        for(j = 0; j < layer.classes; ++j){
            latent_delta += state.input[in_i]*layer.delta[out_i];
            state.delta[in_i++] = scale*layer.delta[out_i++];
        }

        if (!layer.rescore) gradient_array(layer.output + out_i, layer.coords, LOGISTIC, layer.delta + out_i);
        for(j = 0; j < layer.coords; ++j){
            state.delta[in_i++] = layer.delta[out_i++];
        }
        if(layer.rescore) state.delta[in_i-layer.coords-layer.classes-layer.rescore] = latent_delta;
    }
}

#ifdef GPU

void forward_detection_layer_gpu(const detection_layer layer, network_state state)
{
    int outputs = get_detection_layer_output_size(layer);
    float *in_cpu = calloc(layer.batch*layer.inputs, sizeof(float));
    float *truth_cpu = 0;
    if(state.truth){
        truth_cpu = calloc(layer.batch*outputs, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, layer.batch*outputs);
    }
    cuda_pull_array(state.input, in_cpu, layer.batch*layer.inputs);
    network_state cpu_state;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_detection_layer(layer, cpu_state);
    cuda_push_array(layer.output_gpu, layer.output, layer.batch*outputs);
    free(cpu_state.input);
    if(cpu_state.truth) free(cpu_state.truth);
}

void backward_detection_layer_gpu(detection_layer layer, network_state state)
{
    int outputs = get_detection_layer_output_size(layer);

    float *in_cpu =    calloc(layer.batch*layer.inputs, sizeof(float));
    float *delta_cpu = calloc(layer.batch*layer.inputs, sizeof(float));
    float *truth_cpu = 0;
    if(state.truth){
        truth_cpu = calloc(layer.batch*outputs, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, layer.batch*outputs);
    }
    network_state cpu_state;
    cpu_state.train = state.train;
    cpu_state.input = in_cpu;
    cpu_state.truth = truth_cpu;
    cpu_state.delta = delta_cpu;

    cuda_pull_array(state.input, in_cpu, layer.batch*layer.inputs);
    cuda_pull_array(layer.delta_gpu, layer.delta, layer.batch*outputs);
    backward_detection_layer(layer, cpu_state);
    cuda_push_array(state.delta, delta_cpu, layer.batch*layer.inputs);

    free(in_cpu);
    free(delta_cpu);
}
#endif

