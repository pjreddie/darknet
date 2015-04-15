#include "crop_layer.h"
#include "cuda.h"
#include <stdio.h>

image get_crop_image(crop_layer layer)
{
    int h = layer.crop_height;
    int w = layer.crop_width;
    int c = layer.c;
    return float_to_image(w,h,c,layer.output);
}

crop_layer *make_crop_layer(int batch, int h, int w, int c, int crop_height, int crop_width, int flip, float angle)
{
    fprintf(stderr, "Crop Layer: %d x %d -> %d x %d x %d image\n", h,w,crop_height,crop_width,c);
    crop_layer *layer = calloc(1, sizeof(crop_layer));
    layer->batch = batch;
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->flip = flip;
    layer->angle = angle;
    layer->crop_width = crop_width;
    layer->crop_height = crop_height;
    layer->output = calloc(crop_width*crop_height * c*batch, sizeof(float));
    #ifdef GPU
    layer->output_gpu = cuda_make_array(layer->output, crop_width*crop_height*c*batch);
    #endif
    return layer;
}

void forward_crop_layer(const crop_layer layer, network_state state)
{
    int i,j,c,b,row,col;
    int index;
    int count = 0;
    int flip = (layer.flip && rand()%2);
    int dh = rand()%(layer.h - layer.crop_height + 1);
    int dw = rand()%(layer.w - layer.crop_width + 1);
    float scale = 2;
    float trans = -1;
    if(!state.train){
        flip = 0;
        dh = (layer.h - layer.crop_height)/2;
        dw = (layer.w - layer.crop_width)/2;
    }
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
                    layer.output[count++] = state.input[index]*scale + trans;
                }
            }
        }
    }
}

