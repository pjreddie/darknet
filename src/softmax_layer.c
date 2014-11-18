#include "softmax_layer.h"
#include "mini_blas.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

softmax_layer *make_softmax_layer(int batch, int inputs)
{
    fprintf(stderr, "Softmax Layer: %d inputs\n", inputs);
    softmax_layer *layer = calloc(1, sizeof(softmax_layer));
    layer->batch = batch;
    layer->inputs = inputs;
    layer->output = calloc(inputs*batch, sizeof(float));
    layer->delta = calloc(inputs*batch, sizeof(float));
    layer->jacobian = calloc(inputs*inputs*batch, sizeof(float));
    #ifdef GPU
    layer->output_cl = cl_make_array(layer->output, inputs*batch); 
    layer->delta_cl = cl_make_array(layer->delta, inputs*batch); 
    #endif
    return layer;
}

void forward_softmax_layer(const softmax_layer layer, float *input)
{
    int i,b;
    for(b = 0; b < layer.batch; ++b){
        float sum = 0;
        float largest = -FLT_MAX;
        for(i = 0; i < layer.inputs; ++i){
            if(input[i+b*layer.inputs] > largest) largest = input[i+b*layer.inputs];
        }
        for(i = 0; i < layer.inputs; ++i){
            sum += exp(input[i+b*layer.inputs]-largest);
        }
        if(sum) sum = largest+log(sum);
        else sum = largest-100;
        for(i = 0; i < layer.inputs; ++i){
            layer.output[i+b*layer.inputs] = exp(input[i+b*layer.inputs]-sum);
        }
    }
}

void backward_softmax_layer(const softmax_layer layer, float *delta)
{
    int i;
    for(i = 0; i < layer.inputs*layer.batch; ++i){
        delta[i] = layer.delta[i];
    }
}

#ifdef GPU

void pull_softmax_layer_output(const softmax_layer layer)
{
    cl_read_array(layer.output_cl, layer.output, layer.inputs*layer.batch);
}

cl_kernel get_softmax_forward_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/softmax_layer.cl", "forward", 0);
        init = 1;
    }
    return kernel;
}

void forward_softmax_layer_gpu(const softmax_layer layer, cl_mem input)
{
    cl_setup();
    cl_kernel kernel = get_softmax_forward_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.inputs), (void*) &layer.inputs);
    cl.error = clSetKernelArg(kernel, i++, sizeof(input), (void*) &input);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.output_cl), (void*) &layer.output_cl);
    check_error(cl);

    const size_t global_size[] = {layer.batch};

    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);

    /*
    cl_read_array(layer.output_cl, layer.output, layer.inputs*layer.batch);
    int z;
    for(z = 0; z < layer.inputs*layer.batch; ++z) printf("%f,",layer.output[z]);
    */
}

void backward_softmax_layer_gpu(const softmax_layer layer, cl_mem delta)
{
    copy_ongpu(layer.batch*layer.inputs, layer.delta_cl, 1, delta, 1);
}
#endif

/* This is if you want softmax w/o log-loss classification. You probably don't.
   int i,j,b;
   for(b = 0; b < layer.batch; ++b){
   for(i = 0; i < layer.inputs; ++i){
   for(j = 0; j < layer.inputs; ++j){
   int d = (i==j);
   layer.jacobian[b*layer.inputs*layer.inputs + i*layer.inputs + j] = 
   layer.output[b*layer.inputs + i] * (d - layer.output[b*layer.inputs + j]);
   }
   }
   }
   for(b = 0; b < layer.batch; ++b){
   int M = layer.inputs;
   int N = 1;
   int K = layer.inputs;
   float *A = layer.jacobian + b*layer.inputs*layer.inputs;
   float *B = layer.delta + b*layer.inputs;
   float *C = delta + b*layer.inputs;
   gemm(0,0,M,N,K,1,A,K,B,N,0,C,N);
   }
 */
