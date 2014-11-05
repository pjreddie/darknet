#include "cost_layer.h"
#include "mini_blas.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

cost_layer *make_cost_layer(int batch, int inputs)
{
    fprintf(stderr, "Cost Layer: %d inputs\n", inputs);
    cost_layer *layer = calloc(1, sizeof(cost_layer));
    layer->batch = batch;
    layer->inputs = inputs;
    layer->delta = calloc(inputs*batch, sizeof(float));
    layer->output = calloc(1, sizeof(float));
    #ifdef GPU
    layer->delta_cl = cl_make_array(layer->delta, inputs*batch);
    #endif
    return layer;
}

void forward_cost_layer(cost_layer layer, float *input, float *truth)
{
    if (!truth) return;
    copy_cpu(layer.batch*layer.inputs, truth, 1, layer.delta, 1);
    axpy_cpu(layer.batch*layer.inputs, -1, input, 1, layer.delta, 1);
    *(layer.output) = dot_cpu(layer.batch*layer.inputs, layer.delta, 1, layer.delta, 1);
}

void backward_cost_layer(const cost_layer layer, float *input, float *delta)
{
    copy_cpu(layer.batch*layer.inputs, layer.delta, 1, delta, 1);
}

#ifdef GPU
void forward_cost_layer_gpu(cost_layer layer, cl_mem input, cl_mem truth)
{
    if (!truth) return;


    copy_ongpu(layer.batch*layer.inputs, truth, 1, layer.delta_cl, 1);
    axpy_ongpu(layer.batch*layer.inputs, -1, input, 1, layer.delta_cl, 1);
    cl_read_array(layer.delta_cl, layer.delta, layer.batch*layer.inputs);
    *(layer.output) = dot_cpu(layer.batch*layer.inputs, layer.delta, 1, layer.delta, 1);
}

void backward_cost_layer_gpu(const cost_layer layer, cl_mem input, cl_mem delta)
{
    copy_ongpu(layer.batch*layer.inputs, layer.delta_cl, 1, delta, 1);
}
#endif

