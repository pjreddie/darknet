#include "connected_layer.h"
#include "utils.h"
#include "mini_blas.h"

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
    }

#ifdef GPU
    layer->weights_cl = cl_make_array(layer->weights, inputs*outputs);
    layer->biases_cl = cl_make_array(layer->biases, outputs);

    layer->weight_updates_cl = cl_make_array(layer->weight_updates, inputs*outputs);
    layer->bias_updates_cl = cl_make_array(layer->bias_updates, outputs);

    layer->output_cl = cl_make_array(layer->output, outputs*batch);
    layer->delta_cl = cl_make_array(layer->delta, outputs*batch);
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

    //printf("rate:   %f\n", layer->learning_rate);

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
    gradient_array(layer.output, layer.outputs*layer.batch, layer.activation, layer.delta);
    for(i = 0; i < layer.batch; ++i){
        axpy_cpu(layer.outputs, 1, layer.delta + i*layer.outputs, 1, layer.bias_updates, 1);
    }
    int m = layer.inputs;
    int k = layer.batch;
    int n = layer.outputs;
    float *a = input;
    float *b = layer.delta;
    float *c = layer.weight_updates;
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

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
    cl_read_array(layer.weights_cl, layer.weights, layer.inputs*layer.outputs);
    cl_read_array(layer.biases_cl, layer.biases, layer.outputs);
    cl_read_array(layer.weight_updates_cl, layer.weight_updates, layer.inputs*layer.outputs);
    cl_read_array(layer.bias_updates_cl, layer.bias_updates, layer.outputs);
}

void push_connected_layer(connected_layer layer)
{
    cl_write_array(layer.weights_cl, layer.weights, layer.inputs*layer.outputs);
    cl_write_array(layer.biases_cl, layer.biases, layer.outputs);
    cl_write_array(layer.weight_updates_cl, layer.weight_updates, layer.inputs*layer.outputs);
    cl_write_array(layer.bias_updates_cl, layer.bias_updates, layer.outputs);
}

void update_connected_layer_gpu(connected_layer layer)
{
    axpy_ongpu(layer.outputs, layer.learning_rate, layer.bias_updates_cl, 1, layer.biases_cl, 1);
    scal_ongpu(layer.outputs, layer.momentum, layer.bias_updates_cl, 1);

    axpy_ongpu(layer.inputs*layer.outputs, -layer.decay, layer.weights_cl, 1, layer.weight_updates_cl, 1);
    axpy_ongpu(layer.inputs*layer.outputs, layer.learning_rate, layer.weight_updates_cl, 1, layer.weights_cl, 1);
    scal_ongpu(layer.inputs*layer.outputs, layer.momentum, layer.weight_updates_cl, 1);
    pull_connected_layer(layer);
}

void forward_connected_layer_gpu(connected_layer layer, cl_mem input)
{
    int i;
    for(i = 0; i < layer.batch; ++i){
        copy_ongpu_offset(layer.outputs, layer.biases_cl, 0, 1, layer.output_cl, i*layer.outputs, 1);
    }
    int m = layer.batch;
    int k = layer.inputs;
    int n = layer.outputs;
    cl_mem a = input;
    cl_mem b = layer.weights_cl;
    cl_mem c = layer.output_cl;
    gemm_ongpu(0,0,m,n,k,1,a,k,b,n,1,c,n);
    activate_array_ongpu(layer.output_cl, layer.outputs*layer.batch, layer.activation);
}

void backward_connected_layer_gpu(connected_layer layer, cl_mem input, cl_mem delta)
{
    int i;
    gradient_array_ongpu(layer.output_cl, layer.outputs*layer.batch, layer.activation, layer.delta_cl);
    for(i = 0; i < layer.batch; ++i){
        axpy_ongpu_offset(layer.outputs, 1, layer.delta_cl, i*layer.outputs, 1, layer.bias_updates_cl, 0, 1);
    }
    int m = layer.inputs;
    int k = layer.batch;
    int n = layer.outputs;
    cl_mem a = input;
    cl_mem b = layer.delta_cl;
    cl_mem c = layer.weight_updates_cl;
    gemm_ongpu(1,0,m,n,k,1,a,m,b,n,1,c,n);

    m = layer.batch;
    k = layer.outputs;
    n = layer.inputs;

    a = layer.delta_cl;
    b = layer.weights_cl;
    c = delta;

    if(c) gemm_ongpu(0,1,m,n,k,1,a,k,b,k,0,c,n);
}
#endif
