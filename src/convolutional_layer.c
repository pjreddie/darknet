#include "convolutional_layer.h"
#include "utils.h"
#include "mini_blas.h"
#include <stdio.h>

int convolutional_out_height(convolutional_layer layer)
{
    return (layer.h-layer.size)/layer.stride + 1;
}

int convolutional_out_width(convolutional_layer layer)
{
    return (layer.w-layer.size)/layer.stride + 1;
}

image get_convolutional_image(convolutional_layer layer)
{
    int h,w,c;
    h = convolutional_out_height(layer);
    w = convolutional_out_width(layer);
    c = layer.n;
    return float_to_image(h,w,c,layer.output);
}

image get_convolutional_delta(convolutional_layer layer)
{
    int h,w,c;
    h = convolutional_out_height(layer);
    w = convolutional_out_width(layer);
    c = layer.n;
    return float_to_image(h,w,c,layer.delta);
}

convolutional_layer *make_convolutional_layer(int h, int w, int c, int n, int size, int stride, ACTIVATION activation)
{
    int i;
    size = 2*(size/2)+1; //HA! And you thought you'd use an even sized filter...
    convolutional_layer *layer = calloc(1, sizeof(convolutional_layer));
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->n = n;
    layer->stride = stride;
    layer->size = size;

    layer->filters = calloc(c*n*size*size, sizeof(float));
    layer->filter_updates = calloc(c*n*size*size, sizeof(float));
    layer->filter_momentum = calloc(c*n*size*size, sizeof(float));

    layer->biases = calloc(n, sizeof(float));
    layer->bias_updates = calloc(n, sizeof(float));
    layer->bias_momentum = calloc(n, sizeof(float));
    float scale = 1./(size*size*c);
    for(i = 0; i < c*n*size*size; ++i) layer->filters[i] = scale*(rand_uniform());
    for(i = 0; i < n; ++i){
        //layer->biases[i] = rand_normal()*scale + scale;
        layer->biases[i] = 0;
    }
    int out_h = (h-size)/stride + 1;
    int out_w = (w-size)/stride + 1;

    layer->col_image = calloc(out_h*out_w*size*size*c, sizeof(float));
    layer->output = calloc(out_h * out_w * n, sizeof(float));
    layer->delta  = calloc(out_h * out_w * n, sizeof(float));
    layer->activation = activation;

    fprintf(stderr, "Convolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);
    srand(0);

    return layer;
}

void forward_convolutional_layer(const convolutional_layer layer, float *in)
{
    int i;
    int m = layer.n;
    int k = layer.size*layer.size*layer.c;
    int n = ((layer.h-layer.size)/layer.stride + 1)*
            ((layer.w-layer.size)/layer.stride + 1);

    memset(layer.output, 0, m*n*sizeof(float));

    float *a = layer.filters;
    float *b = layer.col_image;
    float *c = layer.output;

    im2col_cpu(in,  layer.c,  layer.h,  layer.w,  layer.size,  layer.stride, b);
    gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);

    for(i = 0; i < m*n; ++i){
        layer.output[i] = activate(layer.output[i], layer.activation);
    }
    //for(i = 0; i < m*n; ++i) if(i%(m*n/10+1)==0) printf("%f, ", layer.output[i]); printf("\n");

}

void gradient_delta_convolutional_layer(convolutional_layer layer)
{
    int i;
    int size = convolutional_out_height(layer)
                *convolutional_out_width(layer)
                *layer.n;
    for(i = 0; i < size; ++i){
        layer.delta[i] *= gradient(layer.output[i], layer.activation);
    }
}

void learn_bias_convolutional_layer(convolutional_layer layer)
{
    int i,j;
    int size = convolutional_out_height(layer)
                *convolutional_out_width(layer);
    for(i = 0; i < layer.n; ++i){
        float sum = 0;
        for(j = 0; j < size; ++j){
            sum += layer.delta[j+i*size];
        }
        layer.bias_updates[i] += sum/size;
    }
}

void learn_convolutional_layer(convolutional_layer layer)
{
    gradient_delta_convolutional_layer(layer);
    learn_bias_convolutional_layer(layer);
    int m = layer.n;
    int n = layer.size*layer.size*layer.c;
    int k = ((layer.h-layer.size)/layer.stride + 1)*
            ((layer.w-layer.size)/layer.stride + 1);

    float *a = layer.delta;
    float *b = layer.col_image;
    float *c = layer.filter_updates;

    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
}

void backward_convolutional_layer(convolutional_layer layer, float *delta)
{
    int m = layer.size*layer.size*layer.c;
    int k = layer.n;
    int n = ((layer.h-layer.size)/layer.stride + 1)*
            ((layer.w-layer.size)/layer.stride + 1);

    float *a = layer.filters;
    float *b = layer.delta;
    float *c = layer.col_image;


    memset(c, 0, m*n*sizeof(float));
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

    memset(delta, 0, layer.h*layer.w*layer.c*sizeof(float));
    col2im_cpu(c,  layer.c,  layer.h,  layer.w,  layer.size,  layer.stride, delta);
}

void update_convolutional_layer(convolutional_layer layer, float step, float momentum, float decay)
{
    int i;
    int size = layer.size*layer.size*layer.c*layer.n;
    for(i = 0; i < layer.n; ++i){
        layer.biases[i] += step*layer.bias_updates[i];
        layer.bias_updates[i] *= momentum;
    }
    for(i = 0; i < size; ++i){
        layer.filters[i] += step*(layer.filter_updates[i] - decay*layer.filters[i]);
        layer.filter_updates[i] *= momentum;
    }
}
/*

void backward_convolutional_layer2(convolutional_layer layer, float *input, float *delta)
{
    image in_delta = float_to_image(layer.h, layer.w, layer.c, delta);
    image out_delta = get_convolutional_delta(layer);
    int i,j;
    for(i = 0; i < layer.n; ++i){
        rotate_image(layer.kernels[i]);
    }

    zero_image(in_delta);
    upsample_image(out_delta, layer.stride, layer.upsampled);
    for(j = 0; j < in_delta.c; ++j){
        for(i = 0; i < layer.n; ++i){
            two_d_convolve(layer.upsampled, i, layer.kernels[i], j, 1, in_delta, j, layer.edge);
        }
    }

    for(i = 0; i < layer.n; ++i){
        rotate_image(layer.kernels[i]);
    }
}


void learn_convolutional_layer(convolutional_layer layer, float *input)
{
    int i;
    image in_image = float_to_image(layer.h, layer.w, layer.c, input);
    image out_delta = get_convolutional_delta(layer);
    gradient_delta_convolutional_layer(layer);
    for(i = 0; i < layer.n; ++i){
        kernel_update(in_image, layer.kernel_updates[i], layer.stride, i, out_delta, layer.edge);
        layer.bias_updates[i] += avg_image_layer(out_delta, i);
    }
}

void update_convolutional_layer(convolutional_layer layer, float step, float momentum, float decay)
{
    int i,j;
    for(i = 0; i < layer.n; ++i){
        layer.bias_momentum[i] = step*(layer.bias_updates[i]) 
                                + momentum*layer.bias_momentum[i];
        layer.biases[i] += layer.bias_momentum[i];
        layer.bias_updates[i] = 0;
        int pixels = layer.kernels[i].h*layer.kernels[i].w*layer.kernels[i].c;
        for(j = 0; j < pixels; ++j){
            layer.kernel_momentum[i].data[j] = step*(layer.kernel_updates[i].data[j] - decay*layer.kernels[i].data[j]) 
                                                + momentum*layer.kernel_momentum[i].data[j];
            layer.kernels[i].data[j] += layer.kernel_momentum[i].data[j];
        }
        zero_image(layer.kernel_updates[i]);
    }
}
*/

void test_convolutional_layer()
{
    convolutional_layer l = *make_convolutional_layer(4,4,1,1,3,1,LINEAR);
    float input[] =    {1,2,3,4,
                        5,6,7,8,
                        9,10,11,12,
                        13,14,15,16};
    float filter[] =   {.5, 0, .3,
                        0  , 1,  0,
                        .2 , 0,  1};
    float delta[] =    {1, 2,
                        3,  4};
    float in_delta[] = {.5,1,.3,.6,
                        5,6,7,8,
                        9,10,11,12,
                        13,14,15,16};
    l.filters = filter;
    forward_convolutional_layer(l, input);
    l.delta = delta;
    learn_convolutional_layer(l);
    image filter_updates = float_to_image(3,3,1,l.filter_updates);
    print_image(filter_updates);
    printf("Delta:\n");
    backward_convolutional_layer(l, in_delta);
    pm(4,4,in_delta);
}

image get_convolutional_filter(convolutional_layer layer, int i)
{
    int h = layer.size;
    int w = layer.size;
    int c = layer.c;
    return float_to_image(h,w,c,layer.filters+i*h*w*c);
}

void visualize_convolutional_layer(convolutional_layer layer, char *window)
{
    int color = 1;
    int border = 1;
    int h,w,c;
    int size = layer.size;
    h = size;
    w = (size + border) * layer.n - border;
    c = layer.c;
    if(c != 3 || !color){
        h = (h+border)*c - border;
        c = 1;
    }

    image filters = make_image(h,w,c);
    int i,j;
    for(i = 0; i < layer.n; ++i){
        int w_offset = i*(size+border);
        image k = get_convolutional_filter(layer, i);
        //printf("%f ** ", layer.biases[i]);
        //print_image(k);
        image copy = copy_image(k);
        normalize_image(copy);
        for(j = 0; j < k.c; ++j){
            //set_pixel(copy,0,0,j,layer.biases[i]);
        }
        if(c == 3 && color){
            embed_image(copy, filters, 0, w_offset);
        }
        else{
            for(j = 0; j < k.c; ++j){
                int h_offset = j*(size+border);
                image layer = get_image_layer(k, j);
                embed_image(layer, filters, h_offset, w_offset);
                free_image(layer);
            }
        }
        free_image(copy);
    }
    image delta = get_convolutional_delta(layer);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Delta", window);
    show_image(dc, buff);
    free_image(dc);
    show_image(filters, window);
    free_image(filters);
}

