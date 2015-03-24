#include "detection_layer.h"
#include "activations.h"
#include "softmax_layer.h"
#include "blas.h"
#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>

int get_detection_layer_locations(detection_layer layer)
{
    return layer.inputs / (layer.classes+layer.coords+layer.rescore+layer.background);
}

int get_detection_layer_output_size(detection_layer layer)
{
    return get_detection_layer_locations(layer)*(layer.background + layer.classes + layer.coords);
}

detection_layer *make_detection_layer(int batch, int inputs, int classes, int coords, int rescore, int background, int nuisance)
{
    detection_layer *layer = calloc(1, sizeof(detection_layer));
    
    layer->batch = batch;
    layer->inputs = inputs;
    layer->classes = classes;
    layer->coords = coords;
    layer->rescore = rescore;
    layer->nuisance = nuisance;
    layer->background = background;
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

void dark_zone(detection_layer layer, int class, int start, network_state state)
{
    int index = start+layer.background+class;
    int size = layer.classes+layer.coords+layer.background;
    int location = (index%(7*7*size)) / size ;
    int r = location / 7;
    int c = location % 7;
    int dr, dc;
    for(dr = -1; dr <= 1; ++dr){
        for(dc = -1; dc <= 1; ++dc){
            if(!(dr || dc)) continue;
            if((r + dr) > 6 || (r + dr) < 0) continue;
            if((c + dc) > 6 || (c + dc) < 0) continue;
            int di = (dr*7 + dc) * size;
            if(state.truth[index+di]) continue;
            layer.output[index + di] = 0;
            //if(!state.truth[start+di]) continue;
            //layer.output[start + di] = 1;
        }
    }
}

void forward_detection_layer(const detection_layer layer, network_state state)
{
    int in_i = 0;
    int out_i = 0;
    int locations = get_detection_layer_locations(layer);
    int i,j;
    for(i = 0; i < layer.batch*locations; ++i){
        int mask = (!state.truth || state.truth[out_i + layer.background + layer.classes + 2]);
        float scale = 1;
        if(layer.rescore) scale = state.input[in_i++];
        else if(layer.nuisance){
            layer.output[out_i++] = 1-state.input[in_i++];
            scale = mask;
        }
        else if(layer.background) layer.output[out_i++] = scale*state.input[in_i++];

        for(j = 0; j < layer.classes; ++j){
            layer.output[out_i++] = scale*state.input[in_i++];
        }
        if(layer.nuisance){
            
        }else if(layer.background){
            softmax_array(layer.output + out_i - layer.classes-layer.background, layer.classes+layer.background, layer.output + out_i - layer.classes-layer.background);
            activate_array(state.input+in_i, layer.coords, LOGISTIC);
        }
        for(j = 0; j < layer.coords; ++j){
            layer.output[out_i++] = mask*state.input[in_i++];
        }
    }
    /*
    if(layer.background || 1){
        for(i = 0; i < layer.batch*locations; ++i){
            int index = i*(layer.classes+layer.coords+layer.background);
            for(j= 0; j < layer.classes; ++j){
                if(state.truth[index+j+layer.background]){
                    //dark_zone(layer, j, index, state);
                }
            }
        }
    }
    */
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
        else if (layer.nuisance)   state.delta[in_i++] = -layer.delta[out_i++];
        else if (layer.background) state.delta[in_i++] = scale*layer.delta[out_i++];
        for(j = 0; j < layer.classes; ++j){
            latent_delta += state.input[in_i]*layer.delta[out_i];
            state.delta[in_i++] = scale*layer.delta[out_i++];
        }

        if (layer.nuisance) ;
        else if (layer.background) gradient_array(layer.output + out_i, layer.coords, LOGISTIC, layer.delta + out_i);
        for(j = 0; j < layer.coords; ++j){
            state.delta[in_i++] = layer.delta[out_i++];
        }
        if(layer.rescore) state.delta[in_i-layer.coords-layer.classes-layer.rescore-layer.background] = latent_delta;
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

