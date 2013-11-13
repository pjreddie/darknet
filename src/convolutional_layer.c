#include "convolutional_layer.h"
#include <stdio.h>

image get_convolutional_image(convolutional_layer layer)
{
    int h = (layer.h-1)/layer.stride + 1;
    int w = (layer.w-1)/layer.stride + 1;
    int c = layer.n;
    return double_to_image(h,w,c,layer.output);
}

image get_convolutional_delta(convolutional_layer layer)
{
    int h = (layer.h-1)/layer.stride + 1;
    int w = (layer.w-1)/layer.stride + 1;
    int c = layer.n;
    return double_to_image(h,w,c,layer.delta);
}

convolutional_layer *make_convolutional_layer(int h, int w, int c, int n, int size, int stride, ACTIVATION activator)
{
    printf("Convolutional Layer: %d x %d x %d image, %d filters\n", h,w,c,n);
    int i;
    convolutional_layer *layer = calloc(1, sizeof(convolutional_layer));
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->n = n;
    layer->stride = stride;
    layer->kernels = calloc(n, sizeof(image));
    layer->kernel_updates = calloc(n, sizeof(image));
    layer->biases = calloc(n, sizeof(double));
    layer->bias_updates = calloc(n, sizeof(double));
    for(i = 0; i < n; ++i){
        layer->biases[i] = .005;
        layer->kernels[i] = make_random_kernel(size, c);
        layer->kernel_updates[i] = make_random_kernel(size, c);
    }
    layer->output = calloc(((h-1)/stride+1) * ((w-1)/stride+1) * n, sizeof(double));
    layer->delta  = calloc(((h-1)/stride+1) * ((w-1)/stride+1) * n, sizeof(double));
    layer->upsampled = make_image(h,w,n);

    if(activator == SIGMOID){
        layer->activation = sigmoid_activation;
        layer->gradient = sigmoid_gradient;
    }else if(activator == RELU){
        layer->activation = relu_activation;
        layer->gradient = relu_gradient;
    }else if(activator == IDENTITY){
        layer->activation = identity_activation;
        layer->gradient = identity_gradient;
    }
    return layer;
}

void forward_convolutional_layer(const convolutional_layer layer, double *in)
{
    image input = double_to_image(layer.h, layer.w, layer.c, in);
    image output = get_convolutional_image(layer);
    int i,j;
    for(i = 0; i < layer.n; ++i){
        convolve(input, layer.kernels[i], layer.stride, i, output);
    }
    for(i = 0; i < output.c; ++i){
        for(j = 0; j < output.h*output.w; ++j){
            int index = i*output.h*output.w + j;
            output.data[index] += layer.biases[i];
            output.data[index] = layer.activation(output.data[index]);
        }
    }
}

void backward_convolutional_layer(convolutional_layer layer, double *input, double *delta)
{
    int i;

    image in_image = double_to_image(layer.h, layer.w, layer.c, input);
    image in_delta = double_to_image(layer.h, layer.w, layer.c, delta);
    image out_delta = get_convolutional_delta(layer);
    zero_image(in_delta);

    for(i = 0; i < layer.n; ++i){
        back_convolve(in_delta, layer.kernels[i], layer.stride, i, out_delta);
    }
    for(i = 0; i < layer.h*layer.w*layer.c; ++i){
        in_delta.data[i] *= layer.gradient(in_image.data[i]);
    }
}

/*
void backpropagate_convolutional_layer_convolve(image input, convolutional_layer layer)
{
    int i,j;
    for(i = 0; i < layer.n; ++i){
        rotate_image(layer.kernels[i]);
    }

    zero_image(input);
    upsample_image(layer.output, layer.stride, layer.upsampled);
    for(j = 0; j < input.c; ++j){
        for(i = 0; i < layer.n; ++i){
            two_d_convolve(layer.upsampled, i, layer.kernels[i], j, 1, input, j);
        }
    }

    for(i = 0; i < layer.n; ++i){
        rotate_image(layer.kernels[i]);
    }
}
*/

void learn_convolutional_layer(convolutional_layer layer, double *input)
{
    int i;
    image in_image = double_to_image(layer.h, layer.w, layer.c, input);
    image out_delta = get_convolutional_delta(layer);
    for(i = 0; i < layer.n; ++i){
        kernel_update(in_image, layer.kernel_updates[i], layer.stride, i, out_delta);
        layer.bias_updates[i] += avg_image_layer(out_delta, i);
    }
}

void update_convolutional_layer(convolutional_layer layer, double step)
{
    return;
    int i,j;
    for(i = 0; i < layer.n; ++i){
        layer.biases[i] += step*layer.bias_updates[i];
        layer.bias_updates[i] = 0;
        int pixels = layer.kernels[i].h*layer.kernels[i].w*layer.kernels[i].c;
        for(j = 0; j < pixels; ++j){
            layer.kernels[i].data[j] += step*layer.kernel_updates[i].data[j];
        }
        zero_image(layer.kernel_updates[i]);
    }
}

void visualize_convolutional_layer(convolutional_layer layer)
{
    int i;
    char buff[256];
    //image vis = make_image(layer.n*layer.size, layer.size*layer.kernels[0].c, 3);
    for(i = 0; i < layer.n; ++i){
        image k = layer.kernels[i];
        sprintf(buff, "Kernel %d", i);
        if(k.c <= 3) show_image(k, buff);
        else show_image_layers(k, buff);
    }
}

