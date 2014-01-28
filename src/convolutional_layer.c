#include "convolutional_layer.h"
#include "utils.h"
#include "mini_blas.h"
#include <stdio.h>

image get_convolutional_image(convolutional_layer layer)
{
    int h,w,c;
    h = layer.out_h;
    w = layer.out_w;
    c = layer.n;
    return double_to_image(h,w,c,layer.output);
}

image get_convolutional_delta(convolutional_layer layer)
{
    int h,w,c;
    h = layer.out_h;
    w = layer.out_w;
    c = layer.n;
    return double_to_image(h,w,c,layer.delta);
}

convolutional_layer *make_convolutional_layer(int h, int w, int c, int n, int size, int stride, ACTIVATION activation)
{
    int i;
    int out_h,out_w;
    size = 2*(size/2)+1; //HA! And you thought you'd use an even sized filter...
    convolutional_layer *layer = calloc(1, sizeof(convolutional_layer));
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->n = n;
    layer->stride = stride;
    layer->size = size;

    layer->filters = calloc(c*n*size*size, sizeof(double));
    layer->filter_updates = calloc(c*n*size*size, sizeof(double));
    layer->filter_momentum = calloc(c*n*size*size, sizeof(double));

    layer->biases = calloc(n, sizeof(double));
    layer->bias_updates = calloc(n, sizeof(double));
    layer->bias_momentum = calloc(n, sizeof(double));
    double scale = 2./(size*size);
    for(i = 0; i < c*n*size*size; ++i) layer->filters[i] = rand_normal()*scale;
    for(i = 0; i < n; ++i){
        //layer->biases[i] = rand_normal()*scale + scale;
        layer->biases[i] = 0;
    }
    out_h = (h-size)/stride + 1;
    out_w = (w-size)/stride + 1;

    layer->col_image = calloc(out_h*out_w*size*size*c, sizeof(double));
    layer->output = calloc(out_h * out_w * n, sizeof(double));
    layer->delta  = calloc(out_h * out_w * n, sizeof(double));
    layer->activation = activation;
    layer->out_h = out_h;
    layer->out_w = out_w;

    fprintf(stderr, "Convolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);
    srand(0);

    return layer;
}

void forward_convolutional_layer(const convolutional_layer layer, double *in)
{
    int m = layer.n;
    int k = layer.size*layer.size*layer.c;
    int n = ((layer.h-layer.size)/layer.stride + 1)*
            ((layer.w-layer.size)/layer.stride + 1);

    memset(layer.output, 0, m*n*sizeof(double));

    double *a = layer.filters;
    double *b = layer.col_image;
    double *c = layer.output;

    im2col_cpu(in,  layer.c,  layer.h,  layer.w,  layer.size,  layer.stride, b);
    gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);

}

void gradient_delta_convolutional_layer(convolutional_layer layer)
{
    int i;
    for(i = 0; i < layer.out_h*layer.out_w*layer.n; ++i){
        layer.delta[i] *= gradient(layer.output[i], layer.activation);
    }
}

void learn_bias_convolutional_layer(convolutional_layer layer)
{
    int i,j;
    int size = layer.out_h*layer.out_w;
    for(i = 0; i < layer.n; ++i){
        double sum = 0;
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

    double *a = layer.delta;
    double *b = layer.col_image;
    double *c = layer.filter_updates;

    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
}

void update_convolutional_layer(convolutional_layer layer, double step, double momentum, double decay)
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

void backward_convolutional_layer2(convolutional_layer layer, double *input, double *delta)
{
    image in_delta = double_to_image(layer.h, layer.w, layer.c, delta);
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


void learn_convolutional_layer(convolutional_layer layer, double *input)
{
    int i;
    image in_image = double_to_image(layer.h, layer.w, layer.c, input);
    image out_delta = get_convolutional_delta(layer);
    gradient_delta_convolutional_layer(layer);
    for(i = 0; i < layer.n; ++i){
        kernel_update(in_image, layer.kernel_updates[i], layer.stride, i, out_delta, layer.edge);
        layer.bias_updates[i] += avg_image_layer(out_delta, i);
    }
}

void update_convolutional_layer(convolutional_layer layer, double step, double momentum, double decay)
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
    double input[] =    {1,2,3,4,
                        5,6,7,8,
                        9,10,11,12,
                        13,14,15,16};
    double filter[] =   {.5, 0, .3,
                        0  , 1,  0,
                        .2 , 0,  1};
    double delta[] =    {1, 2,
                        3,  4};
    l.filters = filter;
    forward_convolutional_layer(l, input);
    l.delta = delta;
    learn_convolutional_layer(l);
    image filter_updates = double_to_image(3,3,1,l.filter_updates);
    print_image(filter_updates);
}

image get_convolutional_filter(convolutional_layer layer, int i)
{
    int h = layer.size;
    int w = layer.size;
    int c = layer.c;
    return double_to_image(h,w,c,layer.filters+i*h*w*c);
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

