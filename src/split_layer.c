//#include "darknet.h"
#include "split_layer.h"

#include <stdio.h>
#include <assert.h>

split_layer make_split_layer(network *net, int batch, int w, int h, int c, int input_layer_index, int tensor_index)
{
    layer *input_layer;
    tensor *input_tensor;
    layer l = {0};
    l.type = SPLIT;
    l.batch = 1;

    input_layer = &net->layers[input_layer_index];

    if (input_layer->type != ODLA) {
        fprintf(stderr, "ERROR: incorrect input layer type, expected ODLA\n");
    }

    input_tensor = &input_layer->output_tensors[tensor_index];

    l.w = input_tensor->w;
    l.h = input_tensor->h;
    l.c = input_tensor->c;

    l.out_w = input_tensor->w;
    l.out_h = input_tensor->h;
    l.out_c = input_tensor->c;
    l.outputs = input_tensor->size;
    l.output_i8 = input_tensor->buffer;

    l.forward = forward_split_layer;
    l.backward = backward_split_layer;
    fprintf(stderr, "split          tensor %d %4d x%4d x%4d   ->  %4d x%4d x%4d\n", tensor_index, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_split_layer(layer *l, int w, int h)
{
    fprintf(stderr, "resize_split_layer\n");
}

void forward_split_layer(const layer l, network net)
{
    fprintf(stderr, "forward_split_layer\n");
}

void backward_split_layer(const layer l, network net)
{
    fprintf(stderr, "backward_split_layer\n");
}
