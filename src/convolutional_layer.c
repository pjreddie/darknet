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

convolutional_layer *make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation, float learning_rate, float momentum, float decay)
{
    int i;
    size = 2*(size/2)+1; //HA! And you thought you'd use an even sized filter...
    convolutional_layer *layer = calloc(1, sizeof(convolutional_layer));

    layer->learning_rate = learning_rate;
    layer->momentum = momentum;
    layer->decay = decay;

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
    //scale = .01;
    for(i = 0; i < c*n*size*size; ++i) layer->filters[i] = scale*rand_normal();
    for(i = 0; i < n; ++i){
        //layer->biases[i] = rand_normal()*scale + scale;
        layer->biases[i] = scale;
        //layer->biases[i] = 1;
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

void resize_convolutional_layer(convolutional_layer *layer, int h, int w, int c)
{
    layer->h = h;
    layer->w = w;
    layer->c = c;
    int out_h = convolutional_out_height(*layer);
    int out_w = convolutional_out_width(*layer);

    layer->col_image = realloc(layer->col_image,
                                out_h*out_w*layer->size*layer->size*layer->c*sizeof(float));
    layer->output = realloc(layer->output,
                                layer->batch*out_h * out_w * layer->n*sizeof(float));
    layer->delta  = realloc(layer->delta,
                                layer->batch*out_h * out_w * layer->n*sizeof(float));
}

void bias_output(const convolutional_layer layer)
{
    int i,j,b;
    int out_h = convolutional_out_height(layer);
    int out_w = convolutional_out_width(layer);
    for(b = 0; b < layer.batch; ++b){
        for(i = 0; i < layer.n; ++i){
            for(j = 0; j < out_h*out_w; ++j){
                layer.output[(b*layer.n + i)*out_h*out_w + j] = layer.biases[i];
            }
        }
    }
}

void forward_convolutional_layer(const convolutional_layer layer, float *in)
{
    int out_h = convolutional_out_height(layer);
    int out_w = convolutional_out_width(layer);
    int i;

    bias_output(layer);

    int m = layer.n;
    int k = layer.size*layer.size*layer.c;
    int n = out_h*out_w;

    float *a = layer.filters;
    float *b = layer.col_image;
    float *c = layer.output;

    for(i = 0; i < layer.batch; ++i){
        im2col_cpu(in, layer.c, layer.h, layer.w, 
            layer.size, layer.stride, layer.pad, b);
        gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        c += n*m;
        in += layer.c*layer.h*layer.w;
    }
    activate_array(layer.output, m*n*layer.batch, layer.activation);
}

void learn_bias_convolutional_layer(convolutional_layer layer)
{
    float alpha = 1./layer.batch;
    int i,b;
    int size = convolutional_out_height(layer)
        *convolutional_out_width(layer);
    for(b = 0; b < layer.batch; ++b){
        for(i = 0; i < layer.n; ++i){
            layer.bias_updates[i] += alpha * sum_array(layer.delta+size*(i+b*layer.n), size);
        }
    }
}

void backward_convolutional_layer(convolutional_layer layer, float *in, float *delta)
{
    float alpha = 1./layer.batch;
    int i;
    int m = layer.n;
    int n = layer.size*layer.size*layer.c;
    int k = convolutional_out_height(layer)*
        convolutional_out_width(layer);

    gradient_array(layer.output, m*k*layer.batch, layer.activation, layer.delta);

    learn_bias_convolutional_layer(layer);

    if(delta) memset(delta, 0, layer.batch*layer.h*layer.w*layer.c*sizeof(float));

    for(i = 0; i < layer.batch; ++i){
        float *a = layer.delta + i*m*k;
        float *b = layer.col_image;
        float *c = layer.filter_updates;

        float *im = in+i*layer.c*layer.h*layer.w;

        im2col_cpu(im, layer.c, layer.h, layer.w, 
                layer.size, layer.stride, layer.pad, b);
        gemm(0,1,m,n,k,alpha,a,k,b,k,1,c,n);

        if(delta){
            a = layer.filters;
            b = layer.delta + i*m*k;
            c = layer.col_image;

            gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

            col2im_cpu(layer.col_image, layer.c,  layer.h,  layer.w,  layer.size,  layer.stride, layer.pad, delta+i*layer.c*layer.h*layer.w);
        }
    }
}

void update_convolutional_layer(convolutional_layer layer)
{
    int size = layer.size*layer.size*layer.c*layer.n;
    axpy_cpu(layer.n, layer.learning_rate, layer.bias_updates, 1, layer.biases, 1);
    scal_cpu(layer.n, layer.momentum, layer.bias_updates, 1);

    axpy_cpu(size, -layer.decay, layer.filters, 1, layer.filter_updates, 1);
    axpy_cpu(size, layer.learning_rate, layer.filter_updates, 1, layer.filters, 1);
    scal_cpu(size, layer.momentum, layer.filter_updates, 1);
}


image get_convolutional_filter(convolutional_layer layer, int i)
{
    int h = layer.size;
    int w = layer.size;
    int c = layer.c;
    return float_to_image(h,w,c,layer.filters+i*h*w*c);
}

image *weighted_sum_filters(convolutional_layer layer, image *prev_filters)
{
    image *filters = calloc(layer.n, sizeof(image));
    int i,j,k,c;
    if(!prev_filters){
        for(i = 0; i < layer.n; ++i){
            filters[i] = copy_image(get_convolutional_filter(layer, i));
        }
    }
    else{
        image base = prev_filters[0];
        for(i = 0; i < layer.n; ++i){
            image filter = get_convolutional_filter(layer, i);
            filters[i] = make_image(base.h, base.w, base.c);
            for(j = 0; j < layer.size; ++j){
                for(k = 0; k < layer.size; ++k){
                    for(c = 0; c < layer.c; ++c){
                        float weight = get_pixel(filter, j, k, c);
                        image prev_filter = copy_image(prev_filters[c]);
                        scale_image(prev_filter, weight);
                        add_into_image(prev_filter, filters[i], 0,0);
                        free_image(prev_filter);
                    }
                }
            }
        }
    }
    return filters;
}

image *visualize_convolutional_layer(convolutional_layer layer, char *window, image *prev_filters)
{
    image *single_filters = weighted_sum_filters(layer, 0);
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

