#include "shortcut_layer.h"
#include "cuda.h"
#include "blas.h"
#include <stdio.h>
#include <assert.h>

layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2)
{
    fprintf(stderr,"Shortcut Layer: %d\n", index);
    layer l = {0};
    l.type = SHORTCUT;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w*h*c;
    l.inputs = w*h*c;
    int stride = w2 / w;

    assert(stride * w == w2);
    assert(stride * h == h2);
    assert(c >= c2);

    l.stride = stride;
    l.n = c2;
    l.index = index;

    l.delta =  calloc(l.outputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));;
    #ifdef GPU
    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    #endif
    return l;
}

void forward_shortcut_layer(const layer l, network_state state)
{
    copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    shortcut_cpu(l.output, l.w, l.h, l.c, l.batch, 1, state.net.layers[l.index].output, l.stride, l.n);
}

void backward_shortcut_layer(const layer l, network_state state)
{
    copy_cpu(l.outputs*l.batch, l.delta, 1, state.delta, 1);
    shortcut_cpu(state.net.layers[l.index].delta, l.w*l.stride, l.h*l.stride, l.n, l.batch, l.stride, l.delta, 1, l.c);
}

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network_state state)
{
    copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);
    shortcut_gpu(l.output_gpu, l.w, l.h, l.c, l.batch, 1, state.net.layers[l.index].output_gpu, l.stride, l.n);
}

void backward_shortcut_layer_gpu(const layer l, network_state state)
{
    copy_ongpu(l.outputs*l.batch, l.delta_gpu, 1, state.delta, 1);
    shortcut_gpu(state.net.layers[l.index].delta_gpu, l.w*l.stride, l.h*l.stride, l.n, l.batch, l.stride, l.delta_gpu, 1, l.c);
}
#endif
