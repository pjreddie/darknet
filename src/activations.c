#include "activations.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *get_activation_string(ACTIVATION a)
{
    switch(a){
        case SIGMOID:
            return "sigmoid";
        case RELU:
            return "relu";
        case RAMP:
            return "ramp";
        case LINEAR:
            return "linear";
        case TANH:
            return "tanh";
        default:
            break;
    }
    return "relu";
}

ACTIVATION get_activation(char *s)
{
    if (strcmp(s, "sigmoid")==0) return SIGMOID;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "linear")==0) return LINEAR;
    if (strcmp(s, "ramp")==0) return RAMP;
    if (strcmp(s, "tanh")==0) return TANH;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

float linear_activate(float x){return x;}
float sigmoid_activate(float x){return 1./(1. + exp(-x));}
float relu_activate(float x){return x*(x>0);}
float ramp_activate(float x){return x*(x>0)+.1*x;}
float tanh_activate(float x){return (exp(2*x)-1)/(exp(2*x)+1);}

float activate(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate(x);
        case SIGMOID:
            return sigmoid_activate(x);
        case RELU:
            return relu_activate(x);
        case RAMP:
            return ramp_activate(x);
        case TANH:
            return tanh_activate(x);
    }
    return 0;
}

void activate_array(float *x, const int n, const ACTIVATION a)
{
    int i;
    for(i = 0; i < n; ++i){
        x[i] = activate(x[i], a);
    }
}


float gradient(float x, ACTIVATION a){
    switch(a){
        case LINEAR:
            return 1;
        case SIGMOID:
            return (1.-x)*x;
        case RELU:
            return (x>0);
        case RAMP:
            return (x>0) + .1;
        case TANH:
            return 1-x*x;
    }
    return 0;
}

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        delta[i] *= gradient(x[i], a);
    }
} 

#ifdef GPU

#include "opencl.h"
#include <math.h>

cl_kernel get_activation_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/activations.cl", "activate_array", 0);
        init = 1;
    }
    return kernel;
}


void activate_array_ongpu(cl_mem x, int n, ACTIVATION a) 
{
    cl_setup();
    cl_kernel kernel = get_activation_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(x), (void*) &x);
    cl.error = clSetKernelArg(kernel, i++, sizeof(n), (void*) &n);
    cl.error = clSetKernelArg(kernel, i++, sizeof(a), (void*) &a);
    check_error(cl);

    size_t gsize = n;

    clEnqueueNDRangeKernel(queue, kernel, 1, 0, &gsize, 0, 0, 0, 0);
    check_error(cl);
}
#endif
