#include "connected_layer.h"
#include "utils.h"
#include "mini_blas.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

connected_layer *make_connected_layer(int inputs, int outputs, ACTIVATION activation)
{
    fprintf(stderr, "Connected Layer: %d inputs, %d outputs\n", inputs, outputs);
    int i;
    connected_layer *layer = calloc(1, sizeof(connected_layer));
    layer->inputs = inputs;
    layer->outputs = outputs;

    layer->output = calloc(outputs, sizeof(float*));
    layer->delta = calloc(outputs, sizeof(float*));

    layer->weight_updates = calloc(inputs*outputs, sizeof(float));
    layer->weight_momentum = calloc(inputs*outputs, sizeof(float));
    layer->weights = calloc(inputs*outputs, sizeof(float));
    float scale = 2./inputs;
    for(i = 0; i < inputs*outputs; ++i)
        layer->weights[i] = rand_normal()*scale;

    layer->bias_updates = calloc(outputs, sizeof(float));
    layer->bias_momentum = calloc(outputs, sizeof(float));
    layer->biases = calloc(outputs, sizeof(float));
    for(i = 0; i < outputs; ++i)
        //layer->biases[i] = rand_normal()*scale + scale;
        layer->biases[i] = 0;

    layer->activation = activation;
    return layer;
}

void update_connected_layer(connected_layer layer, float step, float momentum, float decay)
{
    int i;
    for(i = 0; i < layer.outputs; ++i){
        layer.bias_momentum[i] = step*(layer.bias_updates[i]) + momentum*layer.bias_momentum[i];
        layer.biases[i] += layer.bias_momentum[i];
    }
    for(i = 0; i < layer.outputs*layer.inputs; ++i){
        layer.weight_momentum[i] = step*(layer.weight_updates[i] - decay*layer.weights[i]) + momentum*layer.weight_momentum[i];
        layer.weights[i] += layer.weight_momentum[i];
    }
    memset(layer.bias_updates, 0, layer.outputs*sizeof(float));
    memset(layer.weight_updates, 0, layer.outputs*layer.inputs*sizeof(float));
}

void forward_connected_layer(connected_layer layer, float *input)
{
    int i;
    memcpy(layer.output, layer.biases, layer.outputs*sizeof(float));
    int m = 1;
    int k = layer.inputs;
    int n = layer.outputs;
    float *a = input;
    float *b = layer.weights;
    float *c = layer.output;
    gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
    for(i = 0; i < layer.outputs; ++i){
        layer.output[i] = activate(layer.output[i], layer.activation);
    }
}

void learn_connected_layer(connected_layer layer, float *input)
{
    int i;
    for(i = 0; i < layer.outputs; ++i){
        layer.delta[i] *= gradient(layer.output[i], layer.activation);
        layer.bias_updates[i] += layer.delta[i];
    }
    int m = layer.inputs;
    int k = 1;
    int n = layer.outputs;
    float *a = input;
    float *b = layer.delta;
    float *c = layer.weight_updates;
    gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
}

void backward_connected_layer(connected_layer layer, float *input, float *delta)
{
    memset(delta, 0, layer.inputs*sizeof(float));

    int m = layer.inputs;
    int k = layer.outputs;
    int n = 1;

    float *a = layer.weights;
    float *b = layer.delta;
    float *c = delta;

    gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
}
/*
   void forward_connected_layer(connected_layer layer, float *input)
   {
   int i, j;
   for(i = 0; i < layer.outputs; ++i){
   layer.output[i] = layer.biases[i];
   for(j = 0; j < layer.inputs; ++j){
   layer.output[i] += input[j]*layer.weights[i*layer.inputs + j];
   }
   layer.output[i] = activate(layer.output[i], layer.activation);
   }
   }
   void learn_connected_layer(connected_layer layer, float *input)
   {
   int i, j;
   for(i = 0; i < layer.outputs; ++i){
   layer.delta[i] *= gradient(layer.output[i], layer.activation);
   layer.bias_updates[i] += layer.delta[i];
   for(j = 0; j < layer.inputs; ++j){
   layer.weight_updates[i*layer.inputs + j] += layer.delta[i]*input[j];
   }
   }
   }
   void backward_connected_layer(connected_layer layer, float *input, float *delta)
   {
   int i, j;

   for(j = 0; j < layer.inputs; ++j){
   delta[j] = 0;
   for(i = 0; i < layer.outputs; ++i){
   delta[j] += layer.delta[i]*layer.weights[i*layer.inputs + j];
   }
   }
   }
 */
