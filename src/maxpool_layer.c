#include "maxpool_layer.h"
#include <stdio.h>

image get_maxpool_image(maxpool_layer layer)
{
    int h = (layer.h-1)/layer.stride + 1;
    int w = (layer.w-1)/layer.stride + 1;
    int c = layer.c;
    return float_to_image(h,w,c,layer.output);
}

image get_maxpool_delta(maxpool_layer layer)
{
    int h = (layer.h-1)/layer.stride + 1;
    int w = (layer.w-1)/layer.stride + 1;
    int c = layer.c;
    return float_to_image(h,w,c,layer.delta);
}

maxpool_layer *make_maxpool_layer(int batch, int h, int w, int c, int size, int stride)
{
    fprintf(stderr, "Maxpool Layer: %d x %d x %d image, %d size, %d stride\n", h,w,c,size,stride);
    maxpool_layer *layer = calloc(1, sizeof(maxpool_layer));
    layer->batch = batch;
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->size = size;
    layer->stride = stride;
    int output_size = ((h-1)/stride+1) * ((w-1)/stride+1) * c * batch;
    layer->indexes = calloc(output_size, sizeof(int));
    layer->output =  calloc(output_size, sizeof(float));
    layer->delta =   calloc(output_size, sizeof(float));
    #ifdef GPU
    layer->indexes_cl = cl_make_int_array(layer->indexes, output_size);
    layer->output_cl  = cl_make_array(layer->output, output_size);
    layer->delta_cl   = cl_make_array(layer->delta, output_size);
    #endif
    return layer;
}

void resize_maxpool_layer(maxpool_layer *layer, int h, int w, int c)
{
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->output = realloc(layer->output, ((h-1)/layer->stride+1) * ((w-1)/layer->stride+1) * c * layer->batch* sizeof(float));
    layer->delta = realloc(layer->delta, ((h-1)/layer->stride+1) * ((w-1)/layer->stride+1) * c * layer->batch*sizeof(float));
}

void forward_maxpool_layer(const maxpool_layer layer, float *input)
{
    int b,i,j,k,l,m;
    int w_offset = (-layer.size-1)/2 + 1;
    int h_offset = (-layer.size-1)/2 + 1;

    int h = (layer.h-1)/layer.stride + 1;
    int w = (layer.w-1)/layer.stride + 1;
    int c = layer.c;

    for(b = 0; b < layer.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for(l = 0; l < layer.size; ++l){
                        for(m = 0; m < layer.size; ++m){
                            int cur_h = h_offset + i*layer.stride + l;
                            int cur_w = w_offset + j*layer.stride + m;
                            int index = cur_w + layer.w*(cur_h + layer.h*(k + b*layer.c));
                            int valid = (cur_h >= 0 && cur_h < layer.h &&
                                         cur_w >= 0 && cur_w < layer.w);
                            float val = (valid != 0) ? input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    layer.output[out_index] = max;
                    layer.indexes[out_index] = max_i;
                }
            }
        }
    }
}

void backward_maxpool_layer(const maxpool_layer layer, float *delta)
{
    int i;
    int h = (layer.h-1)/layer.stride + 1;
    int w = (layer.w-1)/layer.stride + 1;
    int c = layer.c;
    memset(delta, 0, layer.batch*layer.h*layer.w*layer.c*sizeof(float));
    for(i = 0; i < h*w*c*layer.batch; ++i){
        int index = layer.indexes[i];
        delta[index] += layer.delta[i];
    }
}

#ifdef GPU
cl_kernel get_forward_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/maxpool_layer.cl", "forward", 0);
        init = 1;
    }
    return kernel;
}

void forward_maxpool_layer_gpu(maxpool_layer layer, cl_mem input)
{
    int h = (layer.h-1)/layer.stride + 1;
    int w = (layer.w-1)/layer.stride + 1;
    int c = layer.c;
    cl_setup();
    cl_kernel kernel = get_forward_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.h), (void*) &layer.h);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.w), (void*) &layer.w);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.c), (void*) &layer.c);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.stride), (void*) &layer.stride);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.size), (void*) &layer.size);
    cl.error = clSetKernelArg(kernel, i++, sizeof(input), (void*) &input);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.output_cl), (void*) &layer.output_cl);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.indexes_cl), (void*) &layer.indexes_cl);
    check_error(cl);

    const size_t global_size[] = {h*w*c*layer.batch};

    clEnqueueNDRangeKernel(queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}

cl_kernel get_backward_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/maxpool_layer.cl", "backward", 0);
        init = 1;
    }
    return kernel;
}

void backward_maxpool_layer_gpu(maxpool_layer layer, cl_mem delta)
{
    cl_setup();
    cl_kernel kernel = get_backward_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.h), (void*) &layer.h);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.w), (void*) &layer.w);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.c), (void*) &layer.c);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.stride), (void*) &layer.stride);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.size), (void*) &layer.size);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.delta_cl), (void*) &layer.delta_cl);
    cl.error = clSetKernelArg(kernel, i++, sizeof(delta), (void*) &delta);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.indexes_cl), (void*) &layer.indexes_cl);
    check_error(cl);

    const size_t global_size[] = {layer.h*layer.w*layer.c*layer.batch};

    clEnqueueNDRangeKernel(queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}

#endif
