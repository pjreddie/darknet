#include "odla_layer.h"

#include <stdio.h>
#include <assert.h>

extern void *odla_create_runtime(void);
extern void odla_load_loadable(void *runtime, const char *loadable, int instance);
extern void *odla_get_output(void *runtime, int index);
extern void odla_execute(void *runtime, int instance);
extern int odla_num_input(void *runtime);
extern int odla_num_output(void *runtime);
extern void odla_alloc_input_tensor(void *runtime, void *buffer, int index);
extern void odla_alloc_output_tensor(void *runtime, void *buffer, int index);
extern int odla_input_channel(void *runtime, int index);
extern int odla_input_width(void *runtime, int index);
extern int odla_input_height(void *runtime, int index);
extern int odla_output_channel(void *runtime, int index);
extern int odla_output_width(void *runtime, int index);
extern int odla_output_height(void *runtime, int index);
extern void odla_copy_input(float *input, unsigned int size, void *buffer);
extern void odla_copy_output(void *buffer, unsigned int size, float *output);

odla_layer make_odla_layer(int batch, int h, int w, int c, int instance, const char *loadable)
{
    int i = 0;

    layer l = {0};
    l.type = ODLA;
    l.batch = batch;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w*h*c;

    l.odla_instance = instance;
    l.output = calloc(l.outputs*batch, sizeof(float));

    //create runtime instance
    l.odla_runtime = odla_create_runtime();

    //load loadable in runtime
    odla_load_loadable(l.odla_runtime, loadable, l.odla_instance);

    //setup input tensor
    l.num_input = odla_num_input(l.odla_runtime);
    l.input_tensors = calloc(l.num_input, sizeof(tensor));
    for (i = 0; i < l.num_input; i++) {
        l.input_tensors[i].w = odla_input_width(l.odla_runtime, i);
        l.input_tensors[i].h = odla_input_height(l.odla_runtime, i);
        l.input_tensors[i].c = odla_input_channel(l.odla_runtime, i);

        l.input_tensors[i].size = (l.input_tensors[i].w * l.input_tensors[i].h * l.input_tensors[i].c);
        l.input_tensors[i].data = calloc(l.input_tensors[i].size, sizeof(float));
        l.input_tensors[i].buffer = NULL;
        odla_alloc_input_tensor(l.odla_runtime, &l.input_tensors[i].buffer, i);
    }

    //setup output tensors
    l.num_output = odla_num_output(l.odla_runtime);
    l.output_tensors = calloc(l.num_output, sizeof(tensor));
    for (i = 0; i < l.num_output; i++) {
        l.output_tensors[i].w = odla_output_width(l.odla_runtime, i);
        l.output_tensors[i].h = odla_output_height(l.odla_runtime, i);
        l.output_tensors[i].c = odla_output_channel(l.odla_runtime, i);

        l.output_tensors[i].size = (l.output_tensors[i].w * l.output_tensors[i].h * l.output_tensors[i].c);
        l.output_tensors[i].data = calloc(l.output_tensors[i].size, sizeof(float));
        odla_alloc_output_tensor(l.odla_runtime, &l.output_tensors[i].buffer, i);
        fprintf(stderr, "odla          tensor %d %4d x%4d x%4d   ->  %4d x%4d x%4d\n", i, w, h, c, l.output_tensors[i].w, l.output_tensors[i].h, l.output_tensors[i].c);
    }

    l.forward = forward_odla_layer;
    l.backward = backward_odla_layer;
    return l;
}

void resize_odla_layer(layer *l, int w, int h)
{
    fprintf(stderr, "resize_odla_layer\n");
}


void forward_odla_layer(const layer l, network net)
{
    int i = 0;

    fprintf(stderr, "forward_odla_layer\n");

    fprintf(stderr, "%s %d input tensor size %d\n", __func__, __LINE__, l.input_tensors[0].size);
    //copy input to dla tensor
//    copy_cpu(l.input_tensors[0].size, net.input, 1, l.input_tensors[0].buffer, 1);
//    odla_copy_input(net.input, l.input_tensors[0].size, l.input_tensors[0].buffer);

    //run network
    odla_execute(l.odla_runtime, l.odla_instance);

    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    //copy output from dla tensor
#if 0
    for (i = 0; i < l.num_output; i++) {
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
        odla_copy_output(l.output_tensors[i].buffer, l.output_tensors[i].size, l.output_tensors[i].data);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    }
#endif
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
}

void backward_odla_layer(const layer l, network net)
{
    fprintf(stderr, "backward_odla_layer\n");
}
