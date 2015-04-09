#include "convolutional_layer.h"
#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

int convolutional_out_height(convolutional_layer layer)
{
    int h = layer.h;
    if (!layer.pad) h -= layer.size;
    else h -= 1;
    return h/layer.stride + 1;
}

int convolutional_out_width(convolutional_layer layer)
{
    int w = layer.w;
    if (!layer.pad) w -= layer.size;
    else w -= 1;
    return w/layer.stride + 1;
}

image get_convolutional_image(convolutional_layer layer)
{
    int h,w,c;
    h = convolutional_out_height(layer);
    w = convolutional_out_width(layer);
    c = layer.n;
    return float_to_image(w,h,c,layer.output);
}

image get_convolutional_delta(convolutional_layer layer)
{
    int h,w,c;
    h = convolutional_out_height(layer);
    w = convolutional_out_width(layer);
    c = layer.n;
    return float_to_image(w,h,c,layer.delta);
}

convolutional_layer *make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation)
{
    int i;
    convolutional_layer *layer = calloc(1, sizeof(convolutional_layer));

    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->n = n;
    layer->batch = batch;
    layer->stride = stride;
    layer->size = size;
    layer->pad = pad;

    layer->filters = calloc(c*n*size*size, sizeof(float));
    layer->filter_updates = calloc(c*n*size*size, sizeof(float));

    layer->biases = calloc(n, sizeof(float));
    layer->bias_updates = calloc(n, sizeof(float));
    float scale = 1./sqrt(size*size*c);
    for(i = 0; i < c*n*size*size; ++i) layer->filters[i] = scale*rand_normal();
    for(i = 0; i < n; ++i){
        layer->biases[i] = scale;
    }
    int out_h = convolutional_out_height(*layer);
    int out_w = convolutional_out_width(*layer);

    layer->col_image = calloc(out_h*out_w*size*size*c, sizeof(float));
    layer->output = calloc(layer->batch*out_h * out_w * n, sizeof(float));
    layer->delta  = calloc(layer->batch*out_h * out_w * n, sizeof(float));

    #ifdef GPU
    layer->filters_gpu = cuda_make_array(layer->filters, c*n*size*size);
    layer->filter_updates_gpu = cuda_make_array(layer->filter_updates, c*n*size*size);

    layer->biases_gpu = cuda_make_array(layer->biases, n);
    layer->bias_updates_gpu = cuda_make_array(layer->bias_updates, n);

    layer->col_image_gpu = cuda_make_array(layer->col_image, out_h*out_w*size*size*c);
    layer->delta_gpu = cuda_make_array(layer->delta, layer->batch*out_h*out_w*n);
    layer->output_gpu = cuda_make_array(layer->output, layer->batch*out_h*out_w*n);
    #endif
    layer->activation = activation;

    fprintf(stderr, "Convolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);

    return layer;
}

void resize_convolutional_layer(convolutional_layer *layer, int h, int w)
{
    layer->h = h;
    layer->w = w;
    int out_h = convolutional_out_height(*layer);
    int out_w = convolutional_out_width(*layer);

    layer->col_image = realloc(layer->col_image,
                                out_h*out_w*layer->size*layer->size*layer->c*sizeof(float));
    layer->output = realloc(layer->output,
                                layer->batch*out_h * out_w * layer->n*sizeof(float));
    layer->delta  = realloc(layer->delta,
                                layer->batch*out_h * out_w * layer->n*sizeof(float));

    #ifdef GPU
    cuda_free(layer->col_image_gpu);
    cuda_free(layer->delta_gpu);
    cuda_free(layer->output_gpu);

    layer->col_image_gpu = cuda_make_array(layer->col_image, out_h*out_w*layer->size*layer->size*layer->c);
    layer->delta_gpu = cuda_make_array(layer->delta, layer->batch*out_h*out_w*layer->n);
    layer->output_gpu = cuda_make_array(layer->output, layer->batch*out_h*out_w*layer->n);
    #endif
}

void bias_output(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] = biases[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}


void forward_convolutional_layer(const convolutional_layer layer, network_state state)
{
    int out_h = convolutional_out_height(layer);
    int out_w = convolutional_out_width(layer);
    int i;

    bias_output(layer.output, layer.biases, layer.batch, layer.n, out_h*out_w);

    int m = layer.n;
    int k = layer.size*layer.size*layer.c;
    int n = out_h*out_w;

    float *a = layer.filters;
    float *b = layer.col_image;
    float *c = layer.output;

    for(i = 0; i < layer.batch; ++i){
        im2col_cpu(state.input, layer.c, layer.h, layer.w, 
            layer.size, layer.stride, layer.pad, b);
        gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        c += n*m;
        state.input += layer.c*layer.h*layer.w;
    }
    activate_array(layer.output, m*n*layer.batch, layer.activation);
}

void backward_convolutional_layer(convolutional_layer layer, network_state state)
{
    int i;
    int m = layer.n;
    int n = layer.size*layer.size*layer.c;
    int k = convolutional_out_height(layer)*
        convolutional_out_width(layer);

    gradient_array(layer.output, m*k*layer.batch, layer.activation, layer.delta);
    backward_bias(layer.bias_updates, layer.delta, layer.batch, layer.n, k);

    if(state.delta) memset(state.delta, 0, layer.batch*layer.h*layer.w*layer.c*sizeof(float));

    for(i = 0; i < layer.batch; ++i){
        float *a = layer.delta + i*m*k;
        float *b = layer.col_image;
        float *c = layer.filter_updates;

        float *im = state.input+i*layer.c*layer.h*layer.w;

        im2col_cpu(im, layer.c, layer.h, layer.w, 
                layer.size, layer.stride, layer.pad, b);
        gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if(state.delta){
            a = layer.filters;
            b = layer.delta + i*m*k;
            c = layer.col_image;

            gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

            col2im_cpu(layer.col_image, layer.c,  layer.h,  layer.w,  layer.size,  layer.stride, layer.pad, state.delta+i*layer.c*layer.h*layer.w);
        }
    }
}

void update_convolutional_layer(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay)
{
    int size = layer.size*layer.size*layer.c*layer.n;
    axpy_cpu(layer.n, learning_rate/batch, layer.bias_updates, 1, layer.biases, 1);
    scal_cpu(layer.n, momentum, layer.bias_updates, 1);

    axpy_cpu(size, -decay*batch, layer.filters, 1, layer.filter_updates, 1);
    axpy_cpu(size, learning_rate/batch, layer.filter_updates, 1, layer.filters, 1);
    scal_cpu(size, momentum, layer.filter_updates, 1);
}


image get_convolutional_filter(convolutional_layer layer, int i)
{
    int h = layer.size;
    int w = layer.size;
    int c = layer.c;
    return float_to_image(w,h,c,layer.filters+i*h*w*c);
}

image *get_filters(convolutional_layer layer)
{
    image *filters = calloc(layer.n, sizeof(image));
    int i;
    for(i = 0; i < layer.n; ++i){
        filters[i] = copy_image(get_convolutional_filter(layer, i));
    }
    return filters;
}

image *visualize_convolutional_layer(convolutional_layer layer, char *window, image *prev_filters)
{
    image *single_filters = get_filters(layer);
    show_images(single_filters, layer.n, window);

    image delta = get_convolutional_image(layer);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_filters;
}

