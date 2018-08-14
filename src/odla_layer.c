#include "odla_layer.h"

#include <stdio.h>
#include <assert.h>

#define ODLA 1

#if ODLA
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
extern int odla_input_size(void *runtime, int index);
extern int odla_output_size(void *runtime, int index);
#endif

odla_layer make_odla_layer(int batch, int h, int w, int c, int instance, const char *loadable)
{
    int i = 0;

    layer l = {0};
    l.w = w;
    l.h = h;
    l.c = c;
    l.type = ODLA;
    l.odla_instance = instance;

#if ODLA
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

        l.input_tensors[i].size = odla_input_size(l.odla_runtime, i);
        l.input_tensors[i].data = calloc(l.input_tensors[i].size, sizeof(uint8_t));
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

        l.output_tensors[i].size = odla_output_size(l.odla_runtime, i);
        l.output_tensors[i].data = calloc(l.output_tensors[i].size, sizeof(int8_t));
        fprintf(stderr, "odla          tensor %d %4d x%4d x%4d   ->  %4d x%4d x%4d\n", i, w, h, c, l.output_tensors[i].w, l.output_tensors[i].h, l.output_tensors[i].c);
        odla_alloc_output_tensor(l.odla_runtime, &l.output_tensors[i].buffer, i);
    }


    /**
     * Update the output dimension using output tensors
     * TODO: Make assertion check for #output_tensor > 0
     * NOTE: Currently supports input of uint8 and output of int8
     **/
    l.batch = batch;
    l.out_w = l.output_tensors[0].w;
    l.out_h = l.output_tensors[0].h;
    l.out_c = l.output_tensors[0].c;
    l.outputs = l.output_tensors[0].size; // TODO: Fixit

    l.output_i8 = calloc(l.outputs*batch, sizeof(int8_t));

    l.output = calloc(l.outputs*batch, sizeof(float));
    l.output_u8 = calloc(l.outputs*batch, sizeof(uint8_t));
#endif

    l.forward = forward_odla_layer;
    l.backward = backward_odla_layer;
    return l;
}

void resize_odla_layer(layer *l, int w, int h)
{
    fprintf(stderr, "resize_odla_layer\n");
}

static int32_t align_to(int32_t number, int32_t multiple)
{
    if (number % multiple == 0)
        return number;

    return number + multiple - number % multiple;
}

void forward_odla_layer(const layer l, network net)
{
    int h = 0;

    fprintf(stderr, "%s %d input tensor size %d\n",
            __func__, __LINE__, l.input_tensors[0].size);

    /**
     * Copy line by line from net.input to input tensor buffer
     * Assumption: input to be in uint8 and NHWC format.
     **/
    uint8_t *src = (uint8_t*)net.input_u8;
    uint8_t *dst = (uint8_t*)l.input_tensors[0].buffer;
    unsigned size = l.w * l.c * sizeof(uint8_t);
    for (h = 0; h < l.h; h++) {
        unsigned srcOffset = l.w * l.c * h;
        unsigned dstOffset = align_to(l.w * l.c, 32) * h;

        memcpy(dst + dstOffset, src + srcOffset, size);
    }

    //run network
#if ODLA
    fprintf(stderr, "%s %d: Executing in ODLA... \n", __func__, __LINE__);
    odla_execute(l.odla_runtime, l.odla_instance);
#endif

    /**
     * Copy output line by line from output tensor to layer output.
     * Again expected to be in feature format. Copy to the output appropriately
     **/
    if (l.num_output == 1) {
        fprintf(stderr, "%s %d: Copying output \n", __func__, __LINE__);
        memcpy(l.output_i8, l.output_tensors[0].buffer,
                l.outputs * l.batch * sizeof(int8_t));
        fprintf(stderr, "%s %d: Forward CPU DONE\n", __func__, __LINE__);
    }
}

void backward_odla_layer(const layer l, network net)
{
    fprintf(stderr, "backward_odla_layer\n");
}
