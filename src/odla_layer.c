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

odla_layer make_odla_layer(int batch, int h, int w, int c,
                            odla_params params)
{
    int i = 0;

    int instance = params.instance;
    const char *loadable = params.loadable;

    layer l = {0};
    l.w = w;
    l.h = h;
    l.c = c;
    l.type = ODLA;
    l.odla_instance = instance;
    l.dla_params = params;

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

/**
 * COPYING RULES to input tensor:
 * [1] Copy from output buffer, if layer != ODLA
 * [2] Copy from output tensor, otherwise
 **/
static void copy_to_input_tensor(tensor *to, layer *from_layer, tensor *from)
{
    if (from_layer == NULL)
        return;

    if (from_layer->type == ODLA && from == NULL)
        return;

    if (from_layer->type == ODLA) {
        memcpy(to->buffer, from->buffer, to->size);
    }
    else {
        int8_t *dst = (int8_t *)to->buffer;

        /* Copy line*/
        int w = from_layer->out_w;
        int h = from_layer->out_h;
        int c = from_layer->out_c;

        /* WAR: decide src either i8 or u8 output buffer */
        if ((w * c) % 32 != 0) {
            int8_t *src = (int8_t *)from_layer->output_u8;
            int hh;
            for (hh = 0; hh < h; hh++) {
                unsigned srcOffset = w * c * hh;
                unsigned dstOffset = align_to(w * c, 32) * hh;

                memcpy(dst + dstOffset,
                        src + srcOffset,
                        w * c * sizeof(uint8_t));
            }
        }
        else {
            uint8_t *src = (uint8_t *)from_layer->output_i8;
            memcpy(dst, src, to->size);
        }
    }
}

void forward_odla_layer(const layer l, network net)
{
    int i;
    fprintf(stderr, "%s %d input tensor size %d\n",
            __func__, __LINE__, l.input_tensors[0].size);

    /**
     * Copy line by line from net.input to input tensor buffer
     * Assumption: input to be in uint8 and NHWC format.
     **/
    odla_params params = l.dla_params;
    for (i = 0; i < params.n_inputs; i++) {
        int layer_index = params.input_layer_index[i];
        int tensor_index = params.input_tensor_index[i];
        layer *from_layer = NULL;
        tensor *from = NULL;
        tensor *to = &l.input_tensors[i];

        /*TODO: have some constrain checks*/
        if (layer_index >= 0 && layer_index < net.n)
            from_layer = &net.layers[layer_index];

        if (from_layer != NULL) {
            if (from_layer->type == ODLA &&
                tensor_index >= 0 && tensor_index < from_layer->num_output) {
                from = &from_layer->output_tensors[tensor_index];
            }
        }

        copy_to_input_tensor(to, from_layer, from);
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
