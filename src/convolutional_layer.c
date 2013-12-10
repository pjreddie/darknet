#include "convolutional_layer.h"
#include "utils.h"
#include <stdio.h>

image get_convolutional_image(convolutional_layer layer)
{
    int h,w,c;
    if(layer.edge){
        h = (layer.h-1)/layer.stride + 1;
        w = (layer.w-1)/layer.stride + 1;
    }else{
        h = (layer.h - layer.size)/layer.stride+1;
        w = (layer.h - layer.size)/layer.stride+1;
    }
    c = layer.n;
    return double_to_image(h,w,c,layer.output);
}

image get_convolutional_delta(convolutional_layer layer)
{
    int h,w,c;
    if(layer.edge){
        h = (layer.h-1)/layer.stride + 1;
        w = (layer.w-1)/layer.stride + 1;
    }else{
        h = (layer.h - layer.size)/layer.stride+1;
        w = (layer.h - layer.size)/layer.stride+1;
    }
    c = layer.n;
    return double_to_image(h,w,c,layer.delta);
}

convolutional_layer *make_convolutional_layer(int h, int w, int c, int n, int size, int stride, ACTIVATION activation)
{
    int i;
    int out_h,out_w;
    convolutional_layer *layer = calloc(1, sizeof(convolutional_layer));
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->n = n;
    layer->edge = 0;
    layer->stride = stride;
    layer->kernels = calloc(n, sizeof(image));
    layer->kernel_updates = calloc(n, sizeof(image));
    layer->kernel_momentum = calloc(n, sizeof(image));
    layer->biases = calloc(n, sizeof(double));
    layer->bias_updates = calloc(n, sizeof(double));
    layer->bias_momentum = calloc(n, sizeof(double));
    double scale = 2./(size*size);
    for(i = 0; i < n; ++i){
        //layer->biases[i] = rand_normal()*scale + scale;
        layer->biases[i] = 0;
        layer->kernels[i] = make_random_kernel(size, c, scale);
        layer->kernel_updates[i] = make_random_kernel(size, c, 0);
        layer->kernel_momentum[i] = make_random_kernel(size, c, 0);
    }
    layer->size = 2*(size/2)+1;
    if(layer->edge){
        out_h = (layer->h-1)/layer->stride + 1;
        out_w = (layer->w-1)/layer->stride + 1;
    }else{
        out_h = (layer->h - layer->size)/layer->stride+1;
        out_w = (layer->h - layer->size)/layer->stride+1;
    }
    fprintf(stderr, "Convolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);
    layer->output = calloc(out_h * out_w * n, sizeof(double));
    layer->delta  = calloc(out_h * out_w * n, sizeof(double));
    layer->upsampled = make_image(h,w,n);
    layer->activation = activation;

    return layer;
}

void forward_convolutional_layer(const convolutional_layer layer, double *in)
{
    image input = double_to_image(layer.h, layer.w, layer.c, in);
    image output = get_convolutional_image(layer);
    int i,j;
    for(i = 0; i < layer.n; ++i){
        convolve(input, layer.kernels[i], layer.stride, i, output, layer.edge);
    }
    for(i = 0; i < output.c; ++i){
        for(j = 0; j < output.h*output.w; ++j){
            int index = i*output.h*output.w + j;
            output.data[index] += layer.biases[i];
            output.data[index] = activate(output.data[index], layer.activation);
        }
    }
}

void backward_convolutional_layer(convolutional_layer layer, double *input, double *delta)
{
    int i;

    image in_delta = double_to_image(layer.h, layer.w, layer.c, delta);
    image out_delta = get_convolutional_delta(layer);
    zero_image(in_delta);

    for(i = 0; i < layer.n; ++i){
        back_convolve(in_delta, layer.kernels[i], layer.stride, i, out_delta, layer.edge);
    }
}

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

void gradient_delta_convolutional_layer(convolutional_layer layer)
{
    int i;
    image out_delta = get_convolutional_delta(layer);
    image out_image = get_convolutional_image(layer);
    for(i = 0; i < out_image.h*out_image.w*out_image.c; ++i){
        out_delta.data[i] *= gradient(out_image.data[i], layer.activation);
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

void visualize_convolutional_filters(convolutional_layer layer, char *window)
{
    int color = 1;
    int border = 1;
    int h,w,c;
    int size = layer.size;
    h = size;
    w = (size + border) * layer.n - border;
    c = layer.kernels[0].c;
    if(c != 3 || !color){
        h = (h+border)*c - border;
        c = 1;
    }

    image filters = make_image(h,w,c);
    int i,j;
    for(i = 0; i < layer.n; ++i){
        int w_offset = i*(size+border);
        image k = layer.kernels[i];
        image copy = copy_image(k);
        normalize_image(copy);
        for(j = 0; j < k.c; ++j){
            set_pixel(copy,0,0,j,layer.biases[i]);
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

void visualize_convolutional_layer(convolutional_layer layer)
{
    int i;
    char buff[256];
    for(i = 0; i < layer.n; ++i){
        image k = layer.kernels[i];
        sprintf(buff, "Kernel %d", i);
        if(k.c <= 3) show_image(k, buff);
        else show_image_layers(k, buff);
    }
}

