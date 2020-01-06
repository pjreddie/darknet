#include "shortcut_layer.h"
#include "convolutional_layer.h"
#include "dark_cuda.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <assert.h>

layer make_shortcut_layer(int batch, int n, int *input_layers, int* input_sizes, int w, int h, int c,
    float **layers_output, float **layers_delta, float **layers_output_gpu, float **layers_delta_gpu, ACTIVATION activation, int train)
{
    fprintf(stderr, "Shortcut Layer: ");
    int i;
    for(i = 0; i < n; ++i) fprintf(stderr, "%d, ", input_layers[i]);

    layer l = { (LAYER_TYPE)0 };
    l.train = train;
    l.type = SHORTCUT;
    l.batch = batch;
    l.activation = activation;
    l.n = n;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    l.layers_output = layers_output;
    l.layers_delta = layers_delta;

    //l.w = w2;
    //l.h = h2;
    //l.c = c2;
    l.w = l.out_w = w;
    l.h = l.out_h = h;
    l.c = l.out_c = c;
    l.outputs = w*h*c;
    l.inputs = l.outputs;

    //if(w != w2 || h != h2 || c != c2) fprintf(stderr, " w = %d, w2 = %d, h = %d, h2 = %d, c = %d, c2 = %d \n", w, w2, h, h2, c, c2);

    l.index = l.input_layers[0];

    if (train) l.delta = (float*)calloc(l.outputs * batch, sizeof(float));
    l.output = (float*)calloc(l.outputs * batch, sizeof(float));

    l.forward = forward_shortcut_layer;
    l.backward = backward_shortcut_layer;
#ifndef GPU
    if (l.activation == SWISH || l.activation == MISH) l.activation_input = (float*)calloc(l.batch*l.outputs, sizeof(float));
#endif // GPU

#ifdef GPU
    if (l.activation == SWISH || l.activation == MISH) l.activation_input_gpu = cuda_make_array(l.activation_input, l.batch*l.outputs);

    l.forward_gpu = forward_shortcut_layer_gpu;
    l.backward_gpu = backward_shortcut_layer_gpu;

    if (train) l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);

    l.input_sizes_gpu = cuda_make_int_array_new_api(input_sizes, l.n);
    l.layers_output_gpu = cuda_make_array_pointers((void**)layers_output_gpu, l.n);
    l.layers_delta_gpu = cuda_make_array_pointers((void**)layers_delta_gpu, l.n);
#endif  // GPU

    l.bflops = l.out_w * l.out_h * l.out_c * l.n / 1000000000.;
    fprintf(stderr, " outputs:%4d x%4d x%4d %5.3f BF\n", l.out_w, l.out_h, l.out_c, l.bflops);
    return l;
}

void resize_shortcut_layer(layer *l, int w, int h, network *net)
{
    //assert(l->w == l->out_w);
    //assert(l->h == l->out_h);
    l->w = l->out_w = w;
    l->h = l->out_h = h;
    l->outputs = w*h*l->out_c;
    l->inputs = l->outputs;
    if (l->train) l->delta = (float*)realloc(l->delta, l->outputs * l->batch * sizeof(float));
    l->output = (float*)realloc(l->output, l->outputs * l->batch * sizeof(float));

    int i;
    for (i = 0; i < l->n; ++i) {
        int index = l->input_layers[i];
        l->input_sizes[i] = net->layers[index].outputs;
        l->layers_output[i] = net->layers[index].output;
        l->layers_delta[i] = net->layers[index].delta;

        assert(l->w == net->layers[index].out_w && l->h == net->layers[index].out_h);
    }

#ifdef GPU
    cuda_free(l->output_gpu);
    l->output_gpu = cuda_make_array(l->output, l->outputs*l->batch);

    if (l->train) {
        cuda_free(l->delta_gpu);
        l->delta_gpu = cuda_make_array(l->delta, l->outputs*l->batch);
    }

    memcpy_ongpu(l->input_sizes_gpu, l->input_sizes, l->n * sizeof(int));
    memcpy_ongpu(l->layers_output_gpu, l->layers_output, l->n * sizeof(float*));
    memcpy_ongpu(l->layers_delta_gpu, l->layers_delta, l->n * sizeof(float*));
#endif

}

void forward_shortcut_layer(const layer l, network_state state)
{
    int from_w = state.net.layers[l.index].w;
    int from_h = state.net.layers[l.index].h;
    int from_c = state.net.layers[l.index].c;

    if (l.n == 1 && from_w == l.w && from_h == l.h && from_c == l.c) {
        int size = l.batch * l.w * l.h * l.c;
        int i;
        #pragma omp parallel for
        for(i = 0; i < size; ++i)
            l.output[i] = state.input[i] + state.net.layers[l.index].output[i];
    }
    else {
        shortcut_multilayer_cpu(l.outputs * l.batch, l.outputs, l.batch, l.n, l.input_sizes, l.layers_output, l.output, state.input);
    }

    //copy_cpu(l.outputs*l.batch, state.input, 1, l.output, 1);
    //shortcut_cpu(l.batch, from_w, from_h, from_c, state.net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.output);

    //activate_array(l.output, l.outputs*l.batch, l.activation);
    if (l.activation == SWISH) activate_array_swish(l.output, l.outputs*l.batch, l.activation_input, l.output);
    else if (l.activation == MISH) activate_array_mish(l.output, l.outputs*l.batch, l.activation_input, l.output);
    else activate_array_cpu_custom(l.output, l.outputs*l.batch, l.activation);
}

void backward_shortcut_layer(const layer l, network_state state)
{
    if (l.activation == SWISH) gradient_array_swish(l.output, l.outputs*l.batch, l.activation_input, l.delta);
    else if (l.activation == MISH) gradient_array_mish(l.outputs*l.batch, l.activation_input, l.delta);
    else gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    backward_shortcut_multilayer_cpu(l.outputs * l.batch, l.outputs, l.batch, l.n, l.input_sizes,
        l.layers_delta, state.delta, l.delta);

    //axpy_cpu(l.outputs*l.batch, 1, l.delta, 1, state.delta, 1);
    //shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, state.net.layers[l.index].delta);
}

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network_state state)
{
    //copy_ongpu(l.outputs*l.batch, state.input, 1, l.output_gpu, 1);
    //simple_copy_ongpu(l.outputs*l.batch, state.input, l.output_gpu);
    //shortcut_gpu(l.batch, l.w, l.h, l.c, state.net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.output_gpu);

    //input_shortcut_gpu(state.input, l.batch, l.w, l.h, l.c, state.net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.output_gpu);

    //-----------
    //if (l.outputs == l.input_sizes[0])
    //if(l.n == 1)
    //{
    //    input_shortcut_gpu(state.input, l.batch, state.net.layers[l.index].w, state.net.layers[l.index].h, state.net.layers[l.index].c,
    //        state.net.layers[l.index].output_gpu, l.out_w, l.out_h, l.out_c, l.output_gpu);
    //}
    //else
    {
        shortcut_multilayer_gpu(l.outputs, l.batch, l.n, l.input_sizes_gpu, l.layers_output_gpu, l.output_gpu, state.input);
    }

    if (l.activation == SWISH) activate_array_swish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
    else if (l.activation == MISH) activate_array_mish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
    else activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);

}

void backward_shortcut_layer_gpu(const layer l, network_state state)
{
    if (l.activation == SWISH) gradient_array_swish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.delta_gpu);
    else if (l.activation == MISH) gradient_array_mish_ongpu(l.outputs*l.batch, l.activation_input_gpu, l.delta_gpu);
    else gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

    backward_shortcut_multilayer_gpu(l.outputs, l.batch, l.n, l.input_sizes_gpu, l.layers_delta_gpu, state.delta, l.delta_gpu);

    //axpy_ongpu(l.outputs*l.batch, 1, l.delta_gpu, 1, state.delta, 1);
    //shortcut_gpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta_gpu, l.w, l.h, l.c, state.net.layers[l.index].delta_gpu);
}
#endif
