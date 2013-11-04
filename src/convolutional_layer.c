#include "convolutional_layer.h"

double convolution_activation(double x)
{
    return x*(x>0);
}

double convolution_gradient(double x)
{
    return (x>=0);
}

convolutional_layer make_convolutional_layer(int h, int w, int c, int n, int size, int stride)
{
    int i;
    convolutional_layer layer;
    layer.n = n;
    layer.stride = stride;
    layer.kernels = calloc(n, sizeof(image));
    layer.kernel_updates = calloc(n, sizeof(image));
    for(i = 0; i < n; ++i){
        layer.kernels[i] = make_random_kernel(size, c);
        layer.kernel_updates[i] = make_random_kernel(size, c);
    }
    layer.output = make_image((h-1)/stride+1, (w-1)/stride+1, n);
    layer.upsampled = make_image(h,w,n);
    return layer;
}

void run_convolutional_layer(const image input, const convolutional_layer layer)
{
    int i;
    for(i = 0; i < layer.n; ++i){
        convolve(input, layer.kernels[i], layer.stride, i, layer.output);
    }
    for(i = 0; i < input.h*input.w*input.c; ++i){
        input.data[i] = convolution_activation(input.data[i]);
    }
}

void backpropagate_layer(image input, convolutional_layer layer)
{
    int i;
    zero_image(input);
    for(i = 0; i < layer.n; ++i){
        back_convolve(input, layer.kernels[i], layer.stride, i, layer.output);
    }
}

void backpropagate_layer_convolve(image input, convolutional_layer layer)
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

void error_convolutional_layer(image input, convolutional_layer layer)
{
    int i;
    for(i = 0; i < layer.n; ++i){
        kernel_update(input, layer.kernel_updates[i], layer.stride, i, layer.output);
    }
    image old_input = copy_image(input);
    zero_image(input);
    for(i = 0; i < layer.n; ++i){
        back_convolve(input, layer.kernels[i], layer.stride, i, layer.output);
    }
    for(i = 0; i < input.h*input.w*input.c; ++i){
        input.data[i] = input.data[i]*convolution_gradient(input.data[i]);
    }
    free_image(old_input);
}

