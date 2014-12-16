#include "crop_layer.h"
#include <stdio.h>

image get_crop_image(crop_layer layer)
{
    int h = layer.crop_height;
    int w = layer.crop_width;
    int c = layer.c;
    return float_to_image(h,w,c,layer.output);
}

crop_layer *make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip)
{
    fprintf(stderr, "Crop Layer: %d x %d -> %d x %d x %d image\n", h,w,crop_height,crop_width,c);
    crop_layer *layer = calloc(1, sizeof(crop_layer));
    layer->batch = batch;
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->flip = flip;
    layer->crop_width = crop_width;
    layer->crop_height = crop_height;
    layer->output = calloc(crop_width*crop_height * c*batch, sizeof(float));
    #ifdef GPU
    layer->output_cl = cl_make_array(layer->output, crop_width*crop_height*c*batch);
    #endif
    return layer;
}

void forward_crop_layer(const crop_layer layer, float *input)
{
    int i,j,c,b,row,col;
    int index;
    int count = 0;
    int flip = (layer.flip && rand()%2);
    int dh = rand()%(layer.h - layer.crop_height);
    int dw = rand()%(layer.w - layer.crop_width);
    for(b = 0; b < layer.batch; ++b){
        for(c = 0; c < layer.c; ++c){
            for(i = 0; i < layer.crop_height; ++i){
                for(j = 0; j < layer.crop_width; ++j){
                    if(flip){
                        col = layer.w - dw - j - 1;    
                    }else{
                        col = j + dw;
                    }
                    row = i + dh;
                    index = col+layer.w*(row+layer.h*(c + layer.c*b)); 
                    layer.output[count++] = input[index];
                }
            }
        }
    }
}

#ifdef GPU
cl_kernel get_crop_kernel()
{
    static int init = 0;
    static cl_kernel kernel;
    if(!init){
        kernel = get_kernel("src/crop_layer.cl", "forward", 0);
        init = 1;
    }
    return kernel;
}

void forward_crop_layer_gpu(crop_layer layer, cl_mem input)
{
    int flip = (layer.flip && rand()%2);
    int dh = rand()%(layer.h - layer.crop_height);
    int dw = rand()%(layer.w - layer.crop_width);
    int size = layer.batch*layer.c*layer.crop_width*layer.crop_height;

    cl_kernel kernel = get_crop_kernel();
    cl_command_queue queue = cl.queue;

    cl_uint i = 0;
    cl.error = clSetKernelArg(kernel, i++, sizeof(input), (void*) &input);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.c), (void*) &layer.c);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.h), (void*) &layer.h);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.w), (void*) &layer.w);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.crop_height), (void*) &layer.crop_height);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.crop_width), (void*) &layer.crop_width);
    cl.error = clSetKernelArg(kernel, i++, sizeof(dh), (void*) &dh);
    cl.error = clSetKernelArg(kernel, i++, sizeof(dw), (void*) &dw);
    cl.error = clSetKernelArg(kernel, i++, sizeof(flip), (void*) &flip);
    cl.error = clSetKernelArg(kernel, i++, sizeof(layer.output_cl), (void*) &layer.output_cl);
    check_error(cl);

    const size_t global_size[] = {size};

    cl.error = clEnqueueNDRangeKernel(queue, kernel, 1, 0, global_size, 0, 0, 0, 0);
    check_error(cl);
}

#endif
