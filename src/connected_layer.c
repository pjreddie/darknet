#include "connected_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

connected_layer *make_connected_layer(int batch, int inputs, int outputs, ACTIVATION activation, float learning_rate, float momentum, float decay)
{
    int i;
    connected_layer *layer = calloc(1, sizeof(connected_layer));

    layer->learning_rate = learning_rate;
    layer->momentum = momentum;
    layer->decay = decay;

    layer->inputs = inputs;
    layer->outputs = outputs;
    layer->batch=batch;

    layer->output = calloc(batch*outputs, sizeof(float*));
    layer->delta = calloc(batch*outputs, sizeof(float*));

    layer->weight_updates = calloc(inputs*outputs, sizeof(float));
    layer->bias_updates = calloc(outputs, sizeof(float));

    layer->weight_prev = calloc(inputs*outputs, sizeof(float));
    layer->bias_prev = calloc(outputs, sizeof(float));

    layer->weights = calloc(inputs*outputs, sizeof(float));
    layer->biases = calloc(outputs, sizeof(float));


    float scale = 1./sqrt(inputs);
    //scale = .01;
    for(i = 0; i < inputs*outputs; ++i){
        layer->weights[i] = scale*rand_normal();
    }

    for(i = 0; i < outputs; ++i){
        layer->biases[i] = scale;
       // layer->biases[i] = 1;
    }

#ifdef GPU
    layer->weights_gpu = cuda_make_array(layer->weights, inputs*outputs);
    layer->biases_gpu = cuda_make_array(layer->biases, outputs);

    layer->weight_updates_gpu = cuda_make_array(layer->weight_updates, inputs*outputs);
    layer->bias_updates_gpu = cuda_make_array(layer->bias_updates, outputs);

    layer->output_gpu = cuda_make_array(layer->output, outputs*batch);
    layer->delta_gpu = cuda_make_array(layer->delta, outputs*batch);
#endif
    layer->activation = activation;
    fprintf(stderr, "Connected Layer: %d inputs, %d outputs\n", inputs, outputs);
    return layer;
}

void secret_update_connected_layer(connected_layer *layer)
{
    int n = layer->outputs*layer->inputs;
    float dot = dot_cpu(n, layer->weight_updates, 1, layer->weight_prev, 1);
    float mag = sqrt(dot_cpu(n, layer->weight_updates, 1, layer->weight_updates, 1))
                * sqrt(dot_cpu(n, layer->weight_prev, 1, layer->weight_prev, 1));
    float cos = dot/mag;
    if(cos > .3) layer->learning_rate *= 1.1;
    else if (cos < -.3) layer-> learning_rate /= 1.1;

    scal_cpu(n, layer->momentum, layer->weight_prev, 1);
    axpy_cpu(n, 1, layer->weight_updates, 1, layer->weight_prev, 1);
    scal_cpu(n, 0, layer->weight_updates, 1);

    scal_cpu(layer->outputs, layer->momentum, layer->bias_prev, 1);
    axpy_cpu(layer->outputs, 1, layer->bias_updates, 1, layer->bias_prev, 1);
    scal_cpu(layer->outputs, 0, layer->bias_updates, 1);

    axpy_cpu(layer->outputs, layer->learning_rate, layer->bias_prev, 1, layer->biases, 1);

    axpy_cpu(layer->inputs*layer->outputs, -layer->decay, layer->weights, 1, layer->weight_prev, 1);
    axpy_cpu(layer->inputs*layer->outputs, layer->learning_rate, layer->weight_prev, 1, layer->weights, 1);
}

void update_connected_layer(connected_layer layer)
{
    axpy_cpu(layer.outputs, layer.learning_rate, layer.bias_updates, 1, layer.biases, 1);
    scal_cpu(layer.outputs, layer.momentum, layer.bias_updates, 1);

    axpy_cpu(layer.inputs*layer.outputs, -layer.decay, layer.weights, 1, layer.weight_updates, 1);
    axpy_cpu(layer.inputs*layer.outputs, layer.learning_rate, layer.weight_updates, 1, layer.weights, 1);
    scal_cpu(layer.inputs*layer.outputs, layer.momentum, layer.weight_updates, 1);
}

void forward_connected_layer(connected_layer layer, float *input)
{
    int i;
    for(i = 0; i < layer.batch; ++i){
        copy_cpu(layer.outputs, layer.biases, 1, layer.output + i*layer.outputs, 1);
    }
    int m = layer.batch;
    int k = layer.inputs;
    int n = layer.outputs;
    float *a = input;
    float *b = layer.weights;
    float *c = layer.output;
    gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
    activate_array(layer.output, layer.outputs*layer.batch, layer.activation);
}

void backward_connected_layer(connected_layer layer, float *input, float *delta)
{
    int i;
    float alpha = 1./layer.batch;
    gradient_array(layer.output, layer.outputs*layer.batch, layer.activation, layer.delta);
    for(i = 0; i < layer.batch; ++i){
        axpy_cpu(layer.outputs, alpha, layer.delta + i*layer.outputs, 1, layer.bias_updates, 1);
    }
    int m = layer.inputs;
    int k = layer.batch;
    int n = layer.outputs;
    float *a = input;
    float *b = layer.delta;
    float *c = layer.weight_updates;
    gemm(1,0,m,n,k,alpha,a,m,b,n,1,c,n);

    m = layer.batch;
    k = layer.outputs;
    n = layer.inputs;

    a = layer.delta;
    b = layer.weights;
    c = delta;

    if(c) gemm(0,1,m,n,k,1,a,k,b,k,0,c,n);
}

#ifdef GPU

void pull_connected_layer(connected_layer layer)
{
    cuda_pull_array(layer.weights_gpu, layer.weights, layer.inputs*layer.outputs);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.outputs);
    cuda_pull_array(layer.weight_updates_gpu, layer.weight_updates, layer.inputs*layer.outputs);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.outputs);
}

void push_connected_layer(connected_layer layer)
{
    cuda_push_array(layer.weights_gpu, layer.weights, layer.inputs*layer.outputs);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.outputs);
    cuda_push_array(layer.weight_updates_gpu, layer.weight_updates, layer.inputs*layer.outputs);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.outputs);
}

void update_connected_layer_gpu(connected_layer layer)
{
/*
    cuda_pull_array(layer.weights_gpu, layer.weights, layer.inputs*layer.outputs);
    cuda_pull_array(layer.weight_updates_gpu, layer.weight_updates, layer.inputs*layer.outputs);
    printf("Weights: %f updates: %f\n", mag_array(layer.weights, layer.inputs*layer.outputs), layer.learning_rate*mag_array(layer.weight_updates, layer.inputs*layer.outputs));
*/

    axpy_ongpu(layer.outputs, layer.learning_rate, layer.bias_updates_gpu, 1, layer.biases_gpu, 1);
    scal_ongpu(layer.outputs, layer.momentum, layer.bias_updates_gpu, 1);

    axpy_ongpu(layer.inputs*layer.outputs, -layer.decay, layer.weights_gpu, 1, layer.weight_updates_gpu, 1);
    axpy_ongpu(layer.inputs*layer.outputs, layer.learning_rate, layer.weight_updates_gpu, 1, layer.weights_gpu, 1);
    scal_ongpu(layer.inputs*layer.outputs, layer.momentum, layer.weight_updates_gpu, 1);
}

void forward_connected_layer_gpu(connected_layer layer, float * input)
{
    int i;
    for(i = 0; i < layer.batch; ++i){
        copy_ongpu_offset(layer.outputs, layer.biases_gpu, 0, 1, layer.output_gpu, i*layer.outputs, 1);
    }
    int m = layer.batch;
    int k = layer.inputs;
    int n = layer.outputs;
    float * a = input;
    float * b = layer.weights_gpu;
    float * c = layer.output_gpu;
    gemm_ongpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
    activate_array_ongpu(layer.output_gpu, layer.outputs*layer.batch, layer.activation);
}

void backward_connected_layer_gpu(connected_layer layer, float * input, float * delta)
{
    float alpha = 1./layer.batch;
    int i;
    gradient_array_ongpu(layer.output_gpu, layer.outputs*layer.batch, layer.activation, layer.delta_gpu);
    for(i = 0; i < layer.batch; ++i){
        axpy_ongpu_offset(layer.outputs, alpha, layer.delta_gpu, i*layer.outputs, 1, layer.bias_updates_gpu, 0, 1);
    }
    int m = layer.inputs;
    int k = layer.batch;
    int n = layer.outputs;
    float * a = input;
    float * b = layer.delta_gpu;
    float * c = layer.weight_updates_gpu;
    gemm_ongpu(1,0,m,n,k,alpha,a,m,b,n,1,c,n);

    m = layer.batch;
    k = layer.outputs;
    n = layer.inputs;

    a = layer.delta_gpu;
    b = layer.weights_gpu;
    c = delta;

    if(c) gemm_ongpu(0,1,m,n,k,1,a,k,b,k,0,c,n);
}
#endif
