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

convolutional_layer *make_convolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, ACTIVATION activation)
{
    int i;
    size = 2*(size/2)+1; //HA! And you thought you'd use an even sized filter...
    convolutional_layer *layer = calloc(1, sizeof(convolutional_layer));
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->n = n;
    layer->batch = batch;
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
    int out_h = convolutional_out_height(*layer);
    int out_w = convolutional_out_width(*layer);

    layer->col_image = calloc(layer->batch*out_h*out_w*size*size*c, sizeof(float));
    layer->output = calloc(layer->batch*out_h * out_w * n, sizeof(float));
    layer->delta  = calloc(layer->batch*out_h * out_w * n, sizeof(float));
    layer->activation = activation;

    fprintf(stderr, "Convolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);
    srand(0);

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
                                layer->batch*out_h*out_w*layer->size*layer->size*layer->c*sizeof(float));
    layer->output = realloc(layer->output,
                                layer->batch*out_h * out_w * layer->n*sizeof(float));
    layer->delta  = realloc(layer->delta,
                                layer->batch*out_h * out_w * layer->n*sizeof(float));
}

void forward_convolutional_layer(const convolutional_layer layer, float *in)
{
    int i;
    int m = layer.n;
    int k = layer.size*layer.size*layer.c;
    int n = convolutional_out_height(layer)*
            convolutional_out_width(layer)*
            layer.batch;

    memset(layer.output, 0, m*n*sizeof(float));

    float *a = layer.filters;
    float *b = layer.col_image;
    float *c = layer.output;
    for(i = 0; i < layer.batch; ++i){
        im2col_cpu(in+i*(n/layer.batch),  layer.c,  layer.h,  layer.w,  layer.size,  layer.stride, b+i*(n/layer.batch));
    }
    gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);

    for(i = 0; i < m*n; ++i){
        layer.output[i] = activate(layer.output[i], layer.activation);
    }
    //for(i = 0; i < m*n; ++i) if(i%(m*n/10+1)==0) printf("%f, ", layer.output[i]); printf("\n");

}

void gradient_delta_convolutional_layer(convolutional_layer layer)
{
    int i;
    int size = convolutional_out_height(layer)*
                convolutional_out_width(layer)*
                layer.n*
                layer.batch;
    for(i = 0; i < size; ++i){
        layer.delta[i] *= gradient(layer.output[i], layer.activation);
    }
}

void learn_bias_convolutional_layer(convolutional_layer layer)
{
    int i,j,b;
    int size = convolutional_out_height(layer)
                *convolutional_out_width(layer);
    for(b = 0; b < layer.batch; ++b){
        for(i = 0; i < layer.n; ++i){
            float sum = 0;
            for(j = 0; j < size; ++j){
                sum += layer.delta[j+size*(i+b*layer.n)];
            }
            layer.bias_updates[i] += sum/size;
        }
    }
}

void learn_convolutional_layer(convolutional_layer layer)
{
    gradient_delta_convolutional_layer(layer);
    learn_bias_convolutional_layer(layer);
    int m = layer.n;
    int n = layer.size*layer.size*layer.c;
    int k = convolutional_out_height(layer)*
            convolutional_out_width(layer)*
            layer.batch;

    float *a = layer.delta;
    float *b = layer.col_image;
    float *c = layer.filter_updates;

    gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);
}

void backward_convolutional_layer(convolutional_layer layer, float *delta)
{
    int i;
    int m = layer.size*layer.size*layer.c;
    int k = layer.n;
    int n = convolutional_out_height(layer)*
            convolutional_out_width(layer)*
            layer.batch;

    float *a = layer.filters;
    float *b = layer.delta;
    float *c = layer.col_image;


    memset(c, 0, m*n*sizeof(float));
    gemm(1,0,m,n,k,1,a,m,b,n,1,c,n);

    memset(delta, 0, layer.batch*layer.h*layer.w*layer.c*sizeof(float));
    for(i = 0; i < layer.batch; ++i){
        col2im_cpu(c+i*n/layer.batch,  layer.c,  layer.h,  layer.w,  layer.size,  layer.stride, delta+i*n/layer.batch);
    }
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
    convolutional_layer l = *make_convolutional_layer(1,4,4,1,1,3,1,LINEAR);
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
    show_image(dc, buff);
    save_image(dc, buff);
    free_image(dc);
    return single_filters;
}

