#include "cost_layer.h"
#include "utils.h"
#include "mini_blas.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

COST_TYPE get_cost_type(char *s)
{
    if (strcmp(s, "sse")==0) return SSE;
    if (strcmp(s, "detection")==0) return DETECTION;
    fprintf(stderr, "Couldn't find activation function %s, going with SSE\n", s);
    return SSE;
}

char *get_cost_string(COST_TYPE a)
{
    switch(a){
        case SSE:
            return "sse";
        case DETECTION:
            return "detection";
    }
    return "sse";
}

cost_layer *make_cost_layer(int batch, int inputs, COST_TYPE type)
{
    fprintf(stderr, "Cost Layer: %d inputs\n", inputs);
    cost_layer *layer = calloc(1, sizeof(cost_layer));
    layer->batch = batch;
    layer->inputs = inputs;
    layer->type = type;
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
    if(layer.type == DETECTION){
        int i;
        for(i = 0; i < layer.batch*layer.inputs; ++i){
            if((i%5) && !truth[(i/5)*5]) layer.delta[i] = 0;
        }
    }
    *(layer.output) = dot_cpu(layer.batch*layer.inputs, layer.delta, 1, layer.delta, 1);
    //printf("cost: %f\n", *layer.output);
}

void backward_cost_layer(const cost_layer layer, float *input, float *delta)
{
    copy_cpu(layer.batch*layer.inputs, layer.delta, 1, delta, 1);
}

#ifdef GPU

cl_kernel get_mask_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/axpy.cl", "mask", 0);
        init = 1;
    }
    return kernel;
}

void mask_ongpu(int n, cl_mem x, cl_mem mask, int mod)
{
    cl_setup();
    cl_kernel kernel = get_mask_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(x), (void*) &x);
    cl.error = clSetKernelArg(kernel, i++, sizeof(mask), (void*) &mask);
    cl.error = clSetKernelArg(kernel, i++, sizeof(mod), (void*) &mod);
    check_error(cl);

    const size_t global_size[] = {n};

    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);

}

void forward_cost_layer_gpu(cost_layer layer, cl_mem input, cl_mem truth)
{
    if (!truth) return;

    copy_ongpu(layer.batch*layer.inputs, truth, 1, layer.delta_cl, 1);
    axpy_ongpu(layer.batch*layer.inputs, -1, input, 1, layer.delta_cl, 1);

    if(layer.type==DETECTION){
        mask_ongpu(layer.inputs*layer.batch, layer.delta_cl, truth, 5);
    }

    cl_read_array(layer.delta_cl, layer.delta, layer.batch*layer.inputs);
    *(layer.output) = dot_cpu(layer.batch*layer.inputs, layer.delta, 1, layer.delta, 1);
    //printf("cost: %f\n", *layer.output);
}

void backward_cost_layer_gpu(const cost_layer layer, cl_mem input, cl_mem delta)
{
    copy_ongpu(layer.batch*layer.inputs, layer.delta_cl, 1, delta, 1);
}
#endif

