#include "scale_channels_layer.h"
#include "dark_cuda.h"
#include "blas.h"
#include <stdio.h>
#include <assert.h>

layer make_scale_channels_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
    fprintf(stderr,"scale Layer: %d\n", index);
    layer l = { (LAYER_TYPE)0 };
    l.type = SCALE_CHANNELS;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    assert(w == 1 && h == 1);

    l.out_w = w2;
    l.out_h = h2;
    l.out_c = c2;
    assert(l.out_c == l.c);

    l.outputs = l.out_w*l.out_h*l.out_c;
    l.inputs = l.outputs;
    l.index = index;

    l.delta = (float*)calloc(l.outputs * batch, sizeof(float));
    l.output = (float*)calloc(l.outputs * batch, sizeof(float));

    l.forward = forward_scale_channels_layer;
    l.backward = backward_scale_channels_layer;
#ifdef GPU
    l.forward_gpu = forward_scale_channels_layer_gpu;
    l.backward_gpu = backward_scale_channels_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
#endif
    return l;
}

void resize_scale_channels_layer(layer *l, int w, int h)
{
    l->out_w = w;
    l->out_h = h;
    l->outputs = l->out_w*l->out_h*l->out_c;
    l->inputs = l->outputs;
    l->delta = (float*)realloc(l->delta, l->outputs * l->batch * sizeof(float));
    l->output = (float*)realloc(l->output, l->outputs * l->batch * sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu = cuda_make_array(l->delta, l->outputs*l->batch);
#endif

}

void forward_scale_channels_layer(const layer l, network_state state)
{
    int size = l.batch * l.out_c * l.out_w * l.out_h;
    int channel_size = l.out_w * l.out_h;
    float *from_output = state.net.layers[l.index].output;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) {
        l.output[i] = state.input[i / channel_size] * from_output[i];
    }

    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_scale_channels_layer(const layer l, network_state state)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    //axpy_cpu(l.outputs*l.batch, 1, l.delta, 1, state.delta, 1);
    //scale_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, state.net.layers[l.index].delta);

    int size = l.batch * l.out_c * l.out_w * l.out_h;
    int channel_size = l.out_w * l.out_h;
    float *from_output = state.net.layers[l.index].output;
    float *from_delta = state.net.layers[l.index].delta;

    int i;
    #pragma omp parallel for
    for (i = 0; i < size; ++i) {
        state.delta[i / channel_size] += l.delta[i] * from_output[i] / channel_size; // l.delta * from  (should be divided by channel_size?)

        from_delta[i] += state.input[i / channel_size] * l.delta[i]; // input * l.delta
    }
}

#ifdef GPU
void forward_scale_channels_layer_gpu(const layer l, network_state state)
{
    int size = l.batch * l.out_c * l.out_w * l.out_h;
    int channel_size = l.out_w * l.out_h;

    scale_channels_gpu(state.net.layers[l.index].output_gpu, size, channel_size, state.input, l.output_gpu);

    activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void backward_scale_channels_layer_gpu(const layer l, network_state state)
{
    gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    int size = l.batch * l.out_c * l.out_w * l.out_h;
    int channel_size = l.out_w * l.out_h;
    float *from_output = state.net.layers[l.index].output_gpu;
    float *from_delta = state.net.layers[l.index].delta_gpu;

    backward_scale_channels_gpu(l.delta_gpu, size, channel_size, state.input, from_delta, from_output, state.delta);
}
#endif
