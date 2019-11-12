#include "shortcut_layer.h"
#include "convolutional_layer.h"
#include "dark_cuda.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <assert.h>

layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2, int assisted_excitation, int train)
{
    if(assisted_excitation) fprintf(stderr, "Shortcut Layer - AE: %d\n", index);
    else fprintf(stderr,"Shortcut Layer: %d\n", index);
    layer l = { (LAYER_TYPE)0 };
    l.train = train;
    l.type = SHORTCUT;
    l.batch = batch;
    l.w = w2;
    l.h = h2;
    l.c = c2;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w*h*c;
    l.inputs = l.outputs;

    l.assisted_excitation = assisted_excitation;

    if(w != w2 || h != h2 || c != c2) fprintf(stderr, " w = %d, w2 = %d, h = %d, h2 = %d, c = %d, c2 = %d \n", w, w2, h, h2, c, c2);

    l.index = index;

    if (train) l.delta = (float*)calloc(l.outputs * batch, sizeof(float));
    l.output = (float*)calloc(l.outputs * batch, sizeof(float));

    l.forward = forward_shortcut_layer;
    l.backward = backward_shortcut_layer;

    if (l.activation == SWISH || l.activation == MISH) l.activation_input = (float*)calloc(l.batch*l.outputs, sizeof(float));
#ifdef GPU
    if (l.activation == SWISH || l.activation == MISH) l.activation_input_gpu = cuda_make_array(l.activation_input, l.batch*l.outputs);

    l.forward_gpu = forward_shortcut_layer_gpu;
    l.backward_gpu = backward_shortcut_layer_gpu;

    if (train) l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    if (l.assisted_excitation)
    {
        const int size = l.out_w * l.out_h * l.batch;
        l.gt_gpu = cuda_make_array(NULL, size);
        l.a_avg_gpu = cuda_make_array(NULL, size);
    }
#endif  // GPU
    return l;
}

void resize_shortcut_layer(layer *l, int w, int h)
{
    //assert(l->w == l->out_w);
    //assert(l->h == l->out_h);
    l->w = l->out_w = w;
    l->h = l->out_h = h;
    l->outputs = w*h*l->out_c;
    l->inputs = l->outputs;
    if (l->train) l->delta = (float*)realloc(l->delta, l->outputs * l->batch * sizeof(float));
    l->output = (float*)realloc(l->output, l->outputs * l->batch * sizeof(float));

#ifdef GPU
    cuda_free(l->output_gpu);
    l->output_gpu = cuda_make_array(l->output, l->outputs*l->batch);

    if (l->train) {
        cuda_free(l->delta_gpu);
        l->delta_gpu = cuda_make_array(l->delta, l->outputs*l->batch);
    }
#endif

}

void forward_shortcut_layer(const layer l, network_state state)
{
    if (l.w == l.out_w && l.h == l.out_h && l.c == l.out_c) {
        int size = l.batch * l.w * l.h * l.c;
        int i;
        #pragma omp parallel for
        for(i = 0; i < size; ++i)
            l.output[i] = state.input[i] + state.net.layers[l.index].output[i];
    }
    else {
        copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
        shortcut_cpu(l.batch, l.w, l.h, l.c, state.net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.output);
    }

    //activate_array(l.output, l.outputs*l.batch, l.activation);
    if (l.activation == SWISH) activate_array_swish(l.output, l.outputs*l.batch, l.activation_input, l.output);
    else if (l.activation == MISH) activate_array_mish(l.output, l.outputs*l.batch, l.activation_input, l.output);
    else activate_array_cpu_custom(l.output, l.outputs*l.batch, l.activation);

    if (l.assisted_excitation && state.train) assisted_excitation_forward(l, state);
}

void backward_shortcut_layer(const layer l, network_state state)
{
    if (l.activation == SWISH) gradient_array_swish(l.output, l.outputs*l.batch, l.activation_input, l.delta);
    else if (l.activation == MISH) gradient_array_mish(l.outputs*l.batch, l.activation_input, l.delta);
    else gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    axpy_cpu(l.outputs*l.batch, 1, l.delta, 1, state.delta, 1);
    shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, state.net.layers[l.index].delta);
}

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network_state state)
{
    //copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);
    //simple_copy_ongpu(l.outputs*l.batch, state.input, l.output_gpu);
    //shortcut_gpu(l.batch, l.w, l.h, l.c, state.net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.output_gpu);
    input_shortcut_gpu(state.input, l.batch, l.w, l.h, l.c, state.net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.output_gpu);

    if (l.activation == SWISH) activate_array_swish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
    else if (l.activation == MISH) activate_array_mish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
    else activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);

    if (l.assisted_excitation && state.train) assisted_excitation_forward_gpu(l, state);
}

void backward_shortcut_layer_gpu(const layer l, network_state state)
{
    if (l.activation == SWISH) gradient_array_swish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.delta_gpu);
    else if (l.activation == MISH) gradient_array_mish_ongpu(l.outputs*l.batch, l.activation_input_gpu, l.delta_gpu);
    else gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    axpy_ongpu(l.outputs*l.batch, 1, l.delta_gpu, 1, state.delta, 1);
    shortcut_gpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta_gpu, l.w, l.h, l.c, state.net.layers[l.index].delta_gpu);
}
#endif
