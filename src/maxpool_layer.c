#include "maxpool_layer.h"

maxpool_layer *make_maxpool_layer(int h, int w, int c, int stride)
{
    maxpool_layer *layer = calloc(1, sizeof(maxpool_layer));
    layer->stride = stride;
    layer->output = make_image((h-1)/stride+1, (w-1)/stride+1, c);
    return layer;
}

void run_maxpool_layer(const image input, const maxpool_layer layer)
{
    int i,j,k;
    for(i = 0; i < layer.output.h*layer.output.w*layer.output.c; ++i) layer.output.data[i] = -DBL_MAX;
    for(i = 0; i < input.h; ++i){
        for(j = 0; j < input.w; ++j){
            for(k = 0; k < input.c; ++k){
                double val = get_pixel(input, i, j, k);
                double cur = get_pixel(layer.output, i/layer.stride, j/layer.stride, k);
                if(val > cur) set_pixel(layer.output, i/layer.stride, j/layer.stride, k, val);
            }
        }
    }
}
