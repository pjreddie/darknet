#include "dropout_layer.h"
#include "utils.h"
#include <stdlib.h>
#include <stdio.h>

dropout_layer *make_dropout_layer(int batch, int inputs, float probability)
{
    fprintf(stderr, "Dropout Layer: %d inputs, %f probability\n", inputs, probability);
    dropout_layer *layer = calloc(1, sizeof(dropout_layer));
    layer->probability = probability;
    layer->inputs = inputs;
    layer->batch = batch;
    layer->rand = calloc(inputs*batch, sizeof(float));
    layer->scale = 1./(1.-probability);
    #ifdef GPU
    layer->rand_cl = cl_make_array(layer->rand, inputs*batch);
    #endif
    return layer;
} 

void forward_dropout_layer(dropout_layer layer, float *input)
{
    int i;
    for(i = 0; i < layer.batch * layer.inputs; ++i){
        float r = rand_uniform();
        layer.rand[i] = r;
        if(r < layer.probability) input[i] = 0;
        else input[i] *= layer.scale;
    }
}

void backward_dropout_layer(dropout_layer layer, float *delta)
{
    int i;
    for(i = 0; i < layer.batch * layer.inputs; ++i){
        float r = layer.rand[i];
        if(r < layer.probability) delta[i] = 0;
        else delta[i] *= layer.scale;
    }
}

#ifdef GPU
cl_kernel get_dropout_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/dropout_layer.cl", "yoloswag420blazeit360noscope", 0);
        init = 1;
    }
    return kernel;
}

void forward_dropout_layer_gpu(dropout_layer layer, cl_mem input)
{
    int j;
    int size = layer.inputs*layer.batch;
    for(j = 0; j < size; ++j) layer.rand[j] = rand_uniform();
    cl_write_array(layer.rand_cl, layer.rand, layer.inputs*layer.batch);

    cl_kernel kernel = get_dropout_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(input), (void*) &input);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.rand_cl), (void*) &layer.rand_cl);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.probability), (void*) &layer.probability);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.scale), (void*) &layer.scale);
    check_error(cl);

    const size_t global_size[] = {size};

    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}

void backward_dropout_layer_gpu(dropout_layer layer, cl_mem delta)
{
    int size = layer.inputs*layer.batch;

    cl_kernel kernel = get_dropout_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta), (void*) &delta);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.rand_cl), (void*) &layer.rand_cl);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.probability), (void*) &layer.probability);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.scale), (void*) &layer.scale);
    check_error(cl);

    const size_t global_size[] = {size};

    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}
#endif
