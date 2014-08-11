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
    layer->delta = calloc(crop_width*crop_height * c*batch, sizeof(float));
    return layer;
}
void forward_crop_layer(const crop_layer layer, float *input)
{
    int i,j,c,b;
    int dh = rand()%(layer.h - layer.crop_height);
    int dw = rand()%(layer.w - layer.crop_width);
    int count = 0;
    if(layer.flip && rand()%2){
        for(b = 0; b < layer.batch; ++b){
            for(c = 0; c < layer.c; ++c){
                for(i = dh; i < dh+layer.crop_height; ++i){
                    for(j = dw+layer.crop_width-1; j >= dw; --j){
                        int index = j+layer.w*(i+layer.h*(c + layer.c*b));
                        layer.output[count++] = input[index];
                    }
                }
            }
        }
    }else{
        for(b = 0; b < layer.batch; ++b){
            for(c = 0; c < layer.c; ++c){
                for(i = dh; i < dh+layer.crop_height; ++i){
                    for(j = dw; j < dw+layer.crop_width; ++j){
                        int index = j+layer.w*(i+layer.h*(c + layer.c*b));
                        layer.output[count++] = input[index];
                    }
                }
            }
        }
    }
}

