#include "caffe_layer.h"
#include <string.h>
#include <stdio.h>
#include <assert.h>

#if CAFFE
extern void *load_caffe_model(const char *cfg, const char *weights);
extern void run_caffe_model(void *net, float *input);
extern float *get_output(void *net, int index);
extern int num_inputs(void *net);
extern int num_outputs(void *net);
extern int input_channels(void *net, int index);
extern int input_height(void *net, int index);
extern int input_width(void *net, int index);
extern int output_channels(void *net, int index);
extern int output_height(void *net, int index);
extern int output_height(void *net, int index);
extern int output_width(void *net, int index);
#endif

caffe_layer make_caffe_layer(int batch, int w, int h, int c, const char *cfg, const char *weights)
{
    layer l = {0};

    //Unused
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w*h*c;
    l.output = calloc(l.outputs*batch, sizeof(float));

#if CAFFE
    int i = 0;
    l.caffe_net = load_caffe_model(cfg, weights);
    l.num_output = num_outputs(l.caffe_net);
    l.output_tensors = calloc(num_outputs(l.caffe_net), sizeof(tensor));

    for (i = 0; i < l.num_output; i++) {
        l.output_tensors[i].w = output_width(l.caffe_net, i);
        l.output_tensors[i].h = output_height(l.caffe_net, i);
        l.output_tensors[i].c = output_channels(l.caffe_net, i);

        l.output_tensors[i].size = (output_channels(l.caffe_net, i) * output_height(l.caffe_net, i) * output_width(l.caffe_net, i));
        l.output_tensors[i].data = calloc(l.output_tensors[i].size, sizeof(float));
        fprintf(stderr, "caffe          tensor %d %4d x%4d x%4d   ->  %4d x%4d x%4d\n", i, w, h, c, l.output_tensors[i].w, l.output_tensors[i].h, l.output_tensors[i].c);
    }
#endif

    l.forward = forward_caffe_layer;
    l.backward = backward_caffe_layer;
    l.type = CAFFE;
    l.batch = batch;
    return l;
}

void resize_caffe_layer(layer *l, int w, int h)
{
}

void forward_caffe_layer(const layer l, network net)
{
#if CAFFE
    int  i =0;

    run_caffe_model((void *)l.caffe_net, (float *)net.input);

    for (i = 0; i < l.num_output; i++) {
        float *output;
        output = get_output((void *)l.caffe_net, i);
        copy_cpu(l.output_tensors[i].size, output, 1, l.output_tensors[i].data, 1);
    }
#endif
    copy_cpu(l.outputs, net.input, 1, l.output, 1);
}

void backward_caffe_layer(const layer l, network net)
{
}
