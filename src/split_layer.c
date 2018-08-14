#include "split_layer.h"

#include <stdio.h>
#include <assert.h>

split_layer make_split_layer(network *net, int batch, int w, int h, int c, int input_layer, int tensor)
{
    layer l = {0};
    l.type = SPLIT;
    l.batch = 1;

    l.input_layer = &net->layers[input_layer];
    l.input_tensor = &l.input_layer->output_tensors[tensor];

    l.w = l.input_tensor->w;
    l.h = l.input_tensor->h;
    l.c = l.input_tensor->c;

    l.out_w = l.input_tensor->w;
    l.out_h = l.input_tensor->h;
    l.out_c = l.input_tensor->c;
    l.outputs = l.out_w*l.out_h*l.out_c;
    l.output = calloc(l.outputs*batch, sizeof(float));

    l.forward = forward_split_layer;
    l.backward = backward_split_layer;
    fprintf(stderr, "split          tensor %d %4d x%4d x%4d   ->  %4d x%4d x%4d\n", tensor, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_split_layer(layer *l, int w, int h)
{
    fprintf(stderr, "resize_split_layer\n");
}


void forward_split_layer(const layer l, network net)
{
    fprintf(stderr, "forward_split_layer\n");
    copy_cpu(l.outputs*l.batch, l.input_tensor->buffer, 1, l.output, 1);
}

void backward_split_layer(const layer l, network net)
{
    fprintf(stderr, "backward_split_layer\n");
}
