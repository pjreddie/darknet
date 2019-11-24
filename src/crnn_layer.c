#include "crnn_layer.h"
#include "convolutional_layer.h"
#include "utils.h"
#include "dark_cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void increment_layer(layer *l, int steps)
{
    int num = l->outputs*l->batch*steps;
    l->output += num;
    l->delta += num;
    l->x += num;
    l->x_norm += num;

#ifdef GPU
    l->output_gpu += num;
    l->delta_gpu += num;
    l->x_gpu += num;
    l->x_norm_gpu += num;
#endif
}

layer make_crnn_layer(int batch, int h, int w, int c, int hidden_filters, int output_filters, int groups, int steps, int size, int stride, int dilation, int pad, ACTIVATION activation, int batch_normalize, int xnor, int train)
{
    fprintf(stderr, "CRNN Layer: %d x %d x %d image, %d filters\n", h,w,c,output_filters);
    batch = batch / steps;
    layer l = { (LAYER_TYPE)0 };
    l.train = train;
    l.batch = batch;
    l.type = CRNN;
    l.steps = steps;
    l.size = size;
    l.stride = stride;
    l.dilation = dilation;
    l.pad = pad;
    l.h = h;
    l.w = w;
    l.c = c;
    l.groups = groups;
    l.out_c = output_filters;
    l.inputs = h * w * c;
    l.hidden = h * w * hidden_filters;
    l.xnor = xnor;

    l.state = (float*)calloc(l.hidden * l.batch * (l.steps + 1), sizeof(float));

    l.input_layer = (layer*)calloc(1, sizeof(layer));
    *(l.input_layer) = make_convolutional_layer(batch, steps, h, w, c, hidden_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, train);
    l.input_layer->batch = batch;
    if (l.workspace_size < l.input_layer->workspace_size) l.workspace_size = l.input_layer->workspace_size;

    l.self_layer = (layer*)calloc(1, sizeof(layer));
    *(l.self_layer) = make_convolutional_layer(batch, steps, h, w, hidden_filters, hidden_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, train);
    l.self_layer->batch = batch;
    if (l.workspace_size < l.self_layer->workspace_size) l.workspace_size = l.self_layer->workspace_size;

    l.output_layer = (layer*)calloc(1, sizeof(layer));
    *(l.output_layer) = make_convolutional_layer(batch, steps, h, w, hidden_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, train);
    l.output_layer->batch = batch;
    if (l.workspace_size < l.output_layer->workspace_size) l.workspace_size = l.output_layer->workspace_size;

    l.out_h = l.output_layer->out_h;
    l.out_w = l.output_layer->out_w;
    l.outputs = l.output_layer->outputs;

    assert(l.input_layer->outputs == l.self_layer->outputs);
    assert(l.input_layer->outputs == l.output_layer->inputs);

    l.output = l.output_layer->output;
    l.delta = l.output_layer->delta;

    l.forward = forward_crnn_layer;
    l.backward = backward_crnn_layer;
    l.update = update_crnn_layer;

#ifdef GPU
    l.forward_gpu = forward_crnn_layer_gpu;
    l.backward_gpu = backward_crnn_layer_gpu;
    l.update_gpu = update_crnn_layer_gpu;
    l.state_gpu = cuda_make_array(l.state, l.batch*l.hidden*(l.steps + 1));
    l.output_gpu = l.output_layer->output_gpu;
    l.delta_gpu = l.output_layer->delta_gpu;
#endif

    l.bflops = l.input_layer->bflops + l.self_layer->bflops + l.output_layer->bflops;

    return l;
}

void resize_crnn_layer(layer *l, int w, int h)
{
    resize_convolutional_layer(l->input_layer, w, h);
    if (l->workspace_size < l->input_layer->workspace_size) l->workspace_size = l->input_layer->workspace_size;

    resize_convolutional_layer(l->self_layer, w, h);
    if (l->workspace_size < l->self_layer->workspace_size) l->workspace_size = l->self_layer->workspace_size;

    resize_convolutional_layer(l->output_layer, w, h);
    if (l->workspace_size < l->output_layer->workspace_size) l->workspace_size = l->output_layer->workspace_size;

    l->output = l->output_layer->output;
    l->delta = l->output_layer->delta;

    int hidden_filters = l->self_layer->c;
    l->w = w;
    l->h = h;
    l->inputs = h * w * l->c;
    l->hidden = h * w * hidden_filters;

    l->out_h = l->output_layer->out_h;
    l->out_w = l->output_layer->out_w;
    l->outputs = l->output_layer->outputs;

    assert(l->input_layer->inputs == l->inputs);
    assert(l->self_layer->inputs == l->hidden);
    assert(l->input_layer->outputs == l->self_layer->outputs);
    assert(l->input_layer->outputs == l->output_layer->inputs);

    l->state = (float*)realloc(l->state, l->batch*l->hidden*(l->steps + 1)*sizeof(float));

#ifdef GPU
    if (l->state_gpu) cudaFree(l->state_gpu);
    l->state_gpu = cuda_make_array(l->state, l->batch*l->hidden*(l->steps + 1));

    l->output_gpu = l->output_layer->output_gpu;
    l->delta_gpu = l->output_layer->delta_gpu;
#endif
}

void free_state_crnn(layer l)
{
    int i;
    for (i = 0; i < l.outputs * l.batch; ++i) l.self_layer->output[i] = rand_uniform(-1, 1);

#ifdef GPU
    cuda_push_array(l.self_layer->output_gpu, l.self_layer->output, l.outputs * l.batch);
#endif  // GPU
}

void update_crnn_layer(layer l, int batch, float learning_rate, float momentum, float decay)
{
    update_convolutional_layer(*(l.input_layer), batch, learning_rate, momentum, decay);
    update_convolutional_layer(*(l.self_layer), batch, learning_rate, momentum, decay);
    update_convolutional_layer(*(l.output_layer), batch, learning_rate, momentum, decay);
}

void forward_crnn_layer(layer l, network_state state)
{
    network_state s = {0};
    s.train = state.train;
    s.workspace = state.workspace;
    s.net = state.net;
    //s.index = state.index;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);

    if (state.train) {
        fill_cpu(l.outputs * l.batch * l.steps, 0, output_layer.delta, 1);
        fill_cpu(l.hidden * l.batch * l.steps, 0, self_layer.delta, 1);
        fill_cpu(l.hidden * l.batch * l.steps, 0, input_layer.delta, 1);
        fill_cpu(l.hidden * l.batch, 0, l.state, 1);
    }

    for (i = 0; i < l.steps; ++i) {
        s.input = state.input;
        forward_convolutional_layer(input_layer, s);

        s.input = l.state;
        forward_convolutional_layer(self_layer, s);

        float *old_state = l.state;
        if(state.train) l.state += l.hidden*l.batch;
        if(l.shortcut){
            copy_cpu(l.hidden * l.batch, old_state, 1, l.state, 1);
        }else{
            fill_cpu(l.hidden * l.batch, 0, l.state, 1);
        }
        axpy_cpu(l.hidden * l.batch, 1, input_layer.output, 1, l.state, 1);
        axpy_cpu(l.hidden * l.batch, 1, self_layer.output, 1, l.state, 1);

        s.input = l.state;
        forward_convolutional_layer(output_layer, s);

        state.input += l.inputs*l.batch;
        increment_layer(&input_layer, 1);
        increment_layer(&self_layer, 1);
        increment_layer(&output_layer, 1);
    }
}

void backward_crnn_layer(layer l, network_state state)
{
    network_state s = {0};
    s.train = state.train;
    s.workspace = state.workspace;
    s.net = state.net;
    //s.index = state.index;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);

    increment_layer(&input_layer, l.steps-1);
    increment_layer(&self_layer, l.steps-1);
    increment_layer(&output_layer, l.steps-1);

    l.state += l.hidden*l.batch*l.steps;
    for (i = l.steps-1; i >= 0; --i) {
        copy_cpu(l.hidden * l.batch, input_layer.output, 1, l.state, 1);
        axpy_cpu(l.hidden * l.batch, 1, self_layer.output, 1, l.state, 1);

        s.input = l.state;
        s.delta = self_layer.delta;
        backward_convolutional_layer(output_layer, s);

        l.state -= l.hidden*l.batch;
        /*
           if(i > 0){
           copy_cpu(l.hidden * l.batch, input_layer.output - l.hidden*l.batch, 1, l.state, 1);
           axpy_cpu(l.hidden * l.batch, 1, self_layer.output - l.hidden*l.batch, 1, l.state, 1);
           }else{
           fill_cpu(l.hidden * l.batch, 0, l.state, 1);
           }
         */

        s.input = l.state;
        s.delta = self_layer.delta - l.hidden*l.batch;
        if (i == 0) s.delta = 0;
        backward_convolutional_layer(self_layer, s);

        copy_cpu(l.hidden*l.batch, self_layer.delta, 1, input_layer.delta, 1);
        if (i > 0 && l.shortcut) axpy_cpu(l.hidden*l.batch, 1, self_layer.delta, 1, self_layer.delta - l.hidden*l.batch, 1);
        s.input = state.input + i*l.inputs*l.batch;
        if(state.delta) s.delta = state.delta + i*l.inputs*l.batch;
        else s.delta = 0;
        backward_convolutional_layer(input_layer, s);

        increment_layer(&input_layer, -1);
        increment_layer(&self_layer, -1);
        increment_layer(&output_layer, -1);
    }
}

#ifdef GPU

void pull_crnn_layer(layer l)
{
    pull_convolutional_layer(*(l.input_layer));
    pull_convolutional_layer(*(l.self_layer));
    pull_convolutional_layer(*(l.output_layer));
}

void push_crnn_layer(layer l)
{
    push_convolutional_layer(*(l.input_layer));
    push_convolutional_layer(*(l.self_layer));
    push_convolutional_layer(*(l.output_layer));
}

void update_crnn_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay)
{
    update_convolutional_layer_gpu(*(l.input_layer), batch, learning_rate, momentum, decay);
    update_convolutional_layer_gpu(*(l.self_layer), batch, learning_rate, momentum, decay);
    update_convolutional_layer_gpu(*(l.output_layer), batch, learning_rate, momentum, decay);
}

void forward_crnn_layer_gpu(layer l, network_state state)
{
    network_state s = {0};
    s.train = state.train;
    s.workspace = state.workspace;
    s.net = state.net;
    if(!state.train) s.index = state.index;  // don't use TC for training (especially without cuda_convert_f32_to_f16() )
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);

/*
#ifdef CUDNN_HALF   // slow and bad for training
    if (!state.train && state.net.cudnn_half) {
        s.index = state.index;
        cuda_convert_f32_to_f16(input_layer.weights_gpu, input_layer.c*input_layer.n*input_layer.size*input_layer.size, input_layer.weights_gpu16);
        cuda_convert_f32_to_f16(self_layer.weights_gpu, self_layer.c*self_layer.n*self_layer.size*self_layer.size, self_layer.weights_gpu16);
        cuda_convert_f32_to_f16(output_layer.weights_gpu, output_layer.c*output_layer.n*output_layer.size*output_layer.size, output_layer.weights_gpu16);
    }
#endif  //CUDNN_HALF
*/

    if (state.train) {
        fill_ongpu(l.outputs * l.batch * l.steps, 0, output_layer.delta_gpu, 1);
        fill_ongpu(l.hidden * l.batch * l.steps, 0, self_layer.delta_gpu, 1);
        fill_ongpu(l.hidden * l.batch * l.steps, 0, input_layer.delta_gpu, 1);
        fill_ongpu(l.hidden * l.batch, 0, l.state_gpu, 1);
    }

    for (i = 0; i < l.steps; ++i) {
        s.input = state.input;
        forward_convolutional_layer_gpu(input_layer, s);

        s.input = l.state_gpu;
        forward_convolutional_layer_gpu(self_layer, s);

        float *old_state = l.state_gpu;
        if(state.train) l.state_gpu += l.hidden*l.batch;
        if(l.shortcut){
            copy_ongpu(l.hidden * l.batch, old_state, 1, l.state_gpu, 1);
        }else{
            fill_ongpu(l.hidden * l.batch, 0, l.state_gpu, 1);
        }
        axpy_ongpu(l.hidden * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
        axpy_ongpu(l.hidden * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);

        s.input = l.state_gpu;
        forward_convolutional_layer_gpu(output_layer, s);

        state.input += l.inputs*l.batch;
        increment_layer(&input_layer, 1);
        increment_layer(&self_layer, 1);
        increment_layer(&output_layer, 1);
    }
}

void backward_crnn_layer_gpu(layer l, network_state state)
{
    network_state s = {0};
    s.train = state.train;
    s.workspace = state.workspace;
    s.net = state.net;
    //s.index = state.index;
    int i;
    layer input_layer = *(l.input_layer);
    layer self_layer = *(l.self_layer);
    layer output_layer = *(l.output_layer);
    increment_layer(&input_layer,  l.steps - 1);
    increment_layer(&self_layer,   l.steps - 1);
    increment_layer(&output_layer, l.steps - 1);
    float *init_state_gpu = l.state_gpu;
    l.state_gpu += l.hidden*l.batch*l.steps;
    for (i = l.steps-1; i >= 0; --i) {
        //copy_ongpu(l.hidden * l.batch, input_layer.output_gpu, 1, l.state_gpu, 1);   // commented in RNN
        //axpy_ongpu(l.hidden * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1); // commented in RNN

        s.input = l.state_gpu;
        s.delta = self_layer.delta_gpu;
        backward_convolutional_layer_gpu(output_layer, s);

        l.state_gpu -= l.hidden*l.batch;

        copy_ongpu(l.hidden*l.batch, self_layer.delta_gpu, 1, input_layer.delta_gpu, 1);

        s.input = l.state_gpu;
        s.delta = self_layer.delta_gpu - l.hidden*l.batch;
        if (i == 0) s.delta = 0;
        backward_convolutional_layer_gpu(self_layer, s);

        if (i > 0 && l.shortcut) axpy_ongpu(l.hidden*l.batch, 1, self_layer.delta_gpu, 1, self_layer.delta_gpu - l.hidden*l.batch, 1);
        s.input = state.input + i*l.inputs*l.batch;
        if(state.delta) s.delta = state.delta + i*l.inputs*l.batch;
        else s.delta = 0;
        backward_convolutional_layer_gpu(input_layer, s);

        if (state.net.try_fix_nan) {
            fix_nan_and_inf(output_layer.delta_gpu, output_layer.inputs * output_layer.batch);
            fix_nan_and_inf(self_layer.delta_gpu, self_layer.inputs * self_layer.batch);
            fix_nan_and_inf(input_layer.delta_gpu, input_layer.inputs * input_layer.batch);
        }

        increment_layer(&input_layer,  -1);
        increment_layer(&self_layer,   -1);
        increment_layer(&output_layer, -1);
    }
    fill_ongpu(l.hidden * l.batch, 0, init_state_gpu, 1); //clean l.state_gpu
}
#endif
