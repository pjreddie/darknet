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
extern int odla_input_size(void *runtime, int index);
extern int odla_output_size(void *runtime, int index);

static void odla_dump_data(const char *filename, int8_t *data, int w, int h, int c)
{
#if 0
    FILE *fp;

    fp = fopen(filename, "w");

    unsigned int line_stride = w * 32;
    unsigned int surface_stride = line_stride * h;

    fprintf(fp, "blobs {\n");
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < w; k++) {
                int surface_index = i / 32;
                fprintf(fp, "  double_data: %d\n", data[surface_stride*surface_index + line_stride*j + 32*k + i%32]);
            }
        }
    }
    fprintf(fp, "  shape {\n");
    fprintf(fp, "    dim: 1\n");
    fprintf(fp, "    dim: %d\n", c);
    fprintf(fp, "    dim: %d\n", h);
    fprintf(fp, "    dim: %d\n", w);
    fprintf(fp, "  }\n");
    fprintf(fp, "}\n");

    fclose(fp);
#endif
}

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
    l.input_tensor = params.input_tensor;

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
        fprintf(stderr, "odla          tensor %d %4d x%4d x%4d   ->  %4d x%4d x%4d\n", i, w, h, c, l.output_tensors[i].w, l.output_tensors[i].h, l.output_tensors[i].c);
        odla_alloc_output_tensor(l.odla_runtime, &l.output_tensors[i].buffer, i);
    }

    l.batch = batch;
    l.out_w = l.output_tensors[0].w;
    l.out_h = l.output_tensors[0].h;
    l.out_c = l.output_tensors[0].c;

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
    int8_t *input = (int8_t *)l.input_tensors[l.input_tensor].buffer;

    fprintf(stderr, "copying input to tensor index %d\n", l.input_tensor);

   //copy from network i8 output to one of input tensors
    //it is assumed that another input is from upsample layer which will
    //update upsampled output directly in tensor buffer to avoid one
    //more memcpy
    memcpy(input, net.input_i8, l.input_tensors[l.input_tensor].size);

    if (l.num_input > 1) {
      for (int i = 0; i < l.num_input; i++) {
        char filename[80];
        snprintf(filename, sizeof(filename), "input_%02d_%02d.dimg", l.layer_index, i);
        odla_dump_data(filename, (int8_t *)l.input_tensors[i].buffer, l.input_tensors[i].w, l.input_tensors[i].h, l.input_tensors[i].c);
      }
    }

    //run network
    fprintf(stderr, "%s %d: Executing in ODLA... \n", __func__, __LINE__);
    odla_execute(l.odla_runtime, l.odla_instance);

    for (int i = 0; i < l.num_output; i++) {
        char filename[80];
        snprintf(filename, sizeof(filename), "output_%02d_%02d.dimg", l.layer_index, i);
        odla_dump_data(filename, (int8_t *)l.output_tensors[i].buffer, l.output_tensors[i].w, l.output_tensors[i].h, l.output_tensors[i].c);
    }
}

void backward_odla_layer(const layer l, network net)
{
    fprintf(stderr, "backward_odla_layer\n");
}
