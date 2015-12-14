#include "connected_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

connected_layer make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation)
{
    int i;
    connected_layer l = {0};
    l.type = CONNECTED;

    l.inputs = inputs;
    l.outputs = outputs;
    l.batch=batch;

    l.output = calloc(batch*outputs, sizeof(float*));
    l.delta = calloc(batch*outputs, sizeof(float*));

    l.weight_updates = calloc(inputs*outputs, sizeof(float));
    l.bias_updates = calloc(outputs, sizeof(float));

    l.weights = calloc(outputs*inputs, sizeof(float));
    l.biases = calloc(outputs, sizeof(float));


    //float scale = 1./sqrt(inputs);
    float scale = sqrt(2./inputs);
    for(i = 0; i < outputs*inputs; ++i){
        l.weights[i] = 2*scale*rand_uniform() - scale;
    }

    for(i = 0; i < outputs; ++i){
        l.biases[i] = scale;
    }

#ifdef GPU
    l.weights_gpu = cuda_make_array(l.weights, outputs*inputs);
    l.biases_gpu = cuda_make_array(l.biases, outputs);

    l.weight_updates_gpu = cuda_make_array(l.weight_updates, outputs*inputs);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, outputs);

    l.output_gpu = cuda_make_array(l.output, outputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, outputs*batch);
#endif
    l.activation = activation;
    fprintf(stderr, "Connected Layer: %d inputs, %d outputs\n", inputs, outputs);
    return l;
}

void update_connected_layer(connected_layer l, int batch, float learning_rate, float momentum, float decay)
{
    axpy_cpu(l.outputs, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, momentum, l.bias_updates, 1);

    axpy_cpu(l.inputs*l.outputs, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.inputs*l.outputs, momentum, l.weight_updates, 1);
}

void forward_connected_layer(connected_layer l, network_state state)
{
    int i;
    for(i = 0; i < l.batch; ++i){
        copy_cpu(l.outputs, l.biases, 1, l.output + i*l.outputs, 1);
    }
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float *a = state.input;
    float *b = l.weights;
    float *c = l.output;
    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_connected_layer(connected_layer l, network_state state)
{
    int i;
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    for(i = 0; i < l.batch; ++i){
        axpy_cpu(l.outputs, 1, l.delta + i*l.outputs, 1, l.bias_updates, 1);
    }
    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float *a = l.delta;
    float *b = state.input;
    float *c = l.weight_updates;
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta;
    b = l.weights;
    c = state.delta;

    if(c) gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
}

#ifdef GPU

void pull_connected_layer(connected_layer l)
{
    cuda_pull_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
}

void push_connected_layer(connected_layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.inputs*l.outputs);
    cuda_push_array(l.biases_gpu, l.biases, l.outputs);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.inputs*l.outputs);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
}

void update_connected_layer_gpu(connected_layer l, int batch, float learning_rate, float momentum, float decay)
{
    axpy_ongpu(l.outputs, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
    scal_ongpu(l.outputs, momentum, l.bias_updates_gpu, 1);

    axpy_ongpu(l.inputs*l.outputs, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
    axpy_ongpu(l.inputs*l.outputs, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
    scal_ongpu(l.inputs*l.outputs, momentum, l.weight_updates_gpu, 1);
}

void forward_connected_layer_gpu(connected_layer l, network_state state)
{
    int i;
    for(i = 0; i < l.batch; ++i){
        copy_ongpu_offset(l.outputs, l.biases_gpu, 0, 1, l.output_gpu, i*l.outputs, 1);
    }
    int m = l.batch;
    int k = l.inputs;
    int n = l.outputs;
    float * a = state.input;
    float * b = l.weights_gpu;
    float * c = l.output_gpu;
    gemm_ongpu(0,1,m,n,k,1,a,k,b,k,1,c,n);
    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);

/*
    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
    float avg = mean_array(l.output, l.outputs*l.batch);
    printf("%f\n", avg);
    */
}

void backward_connected_layer_gpu(connected_layer l, network_state state)
{
    int i;
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    for(i = 0; i < l.batch; ++i){
        axpy_ongpu_offset(l.outputs, 1, l.delta_gpu, i*l.outputs, 1, l.bias_updates_gpu, 0, 1);
    }
    int m = l.outputs;
    int k = l.batch;
    int n = l.inputs;
    float * a = l.delta_gpu;
    float * b = state.input;
    float * c = l.weight_updates_gpu;
    gemm_ongpu(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = l.batch;
    k = l.outputs;
    n = l.inputs;

    a = l.delta_gpu;
    b = l.weights_gpu;
    c = state.delta;

    if(c) gemm_ongpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
}
#endif
