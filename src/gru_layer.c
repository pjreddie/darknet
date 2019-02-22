#include "gru_layer.h"
#include "connected_layer.h"
#include "utils.h"
#include "cuda.h"
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

layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize)
{
    fprintf(stderr, "GRU Layer: %d inputs, %d outputs\n", inputs, outputs);
    batch = batch / steps;
    layer l = { (LAYER_TYPE)0 };
    l.batch = batch;
    l.type = GRU;
    l.steps = steps;
    l.inputs = inputs;

    l.input_z_layer = (layer*)malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.input_z_layer) = make_connected_layer(batch, steps, inputs, outputs, LINEAR, batch_normalize);
    l.input_z_layer->batch = batch;

    l.state_z_layer = (layer*)malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.state_z_layer) = make_connected_layer(batch, steps, outputs, outputs, LINEAR, batch_normalize);
    l.state_z_layer->batch = batch;



    l.input_r_layer = (layer*)malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.input_r_layer) = make_connected_layer(batch, steps, inputs, outputs, LINEAR, batch_normalize);
    l.input_r_layer->batch = batch;

    l.state_r_layer = (layer*)malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.state_r_layer) = make_connected_layer(batch, steps, outputs, outputs, LINEAR, batch_normalize);
    l.state_r_layer->batch = batch;



    l.input_h_layer = (layer*)malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.input_h_layer) = make_connected_layer(batch, steps, inputs, outputs, LINEAR, batch_normalize);
    l.input_h_layer->batch = batch;

    l.state_h_layer = (layer*)malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.state_h_layer) = make_connected_layer(batch, steps, outputs, outputs, LINEAR, batch_normalize);
    l.state_h_layer->batch = batch;

    l.batch_normalize = batch_normalize;


    l.outputs = outputs;
    l.output = (float*)calloc(outputs * batch * steps, sizeof(float));
    l.delta = (float*)calloc(outputs * batch * steps, sizeof(float));
    l.state = (float*)calloc(outputs * batch, sizeof(float));
    l.prev_state = (float*)calloc(outputs * batch, sizeof(float));
    l.forgot_state = (float*)calloc(outputs * batch, sizeof(float));
    l.forgot_delta = (float*)calloc(outputs * batch, sizeof(float));

    l.r_cpu = (float*)calloc(outputs * batch, sizeof(float));
    l.z_cpu = (float*)calloc(outputs * batch, sizeof(float));
    l.h_cpu = (float*)calloc(outputs * batch, sizeof(float));

    l.forward = forward_gru_layer;
    l.backward = backward_gru_layer;
    l.update = update_gru_layer;

#ifdef GPU
    l.forward_gpu = forward_gru_layer_gpu;
    l.backward_gpu = backward_gru_layer_gpu;
    l.update_gpu = update_gru_layer_gpu;

    l.forgot_state_gpu = cuda_make_array(l.output, batch*outputs);
    l.forgot_delta_gpu = cuda_make_array(l.output, batch*outputs);
    l.prev_state_gpu = cuda_make_array(l.output, batch*outputs);
    l.state_gpu = cuda_make_array(l.output, batch*outputs);
    l.output_gpu = cuda_make_array(l.output, batch*outputs*steps);
    l.delta_gpu = cuda_make_array(l.delta, batch*outputs*steps);
    l.r_gpu = cuda_make_array(l.output_gpu, batch*outputs);
    l.z_gpu = cuda_make_array(l.output_gpu, batch*outputs);
    l.h_gpu = cuda_make_array(l.output_gpu, batch*outputs);
#endif

    return l;
}

void update_gru_layer(layer l, int batch, float learning_rate, float momentum, float decay)
{
    update_connected_layer(*(l.input_layer), batch, learning_rate, momentum, decay);
    update_connected_layer(*(l.self_layer), batch, learning_rate, momentum, decay);
    update_connected_layer(*(l.output_layer), batch, learning_rate, momentum, decay);
}

void forward_gru_layer(layer l, network_state state)
{
    network_state s = {0};
    s.train = state.train;
    s.workspace = state.workspace;
    int i;
    layer input_z_layer = *(l.input_z_layer);
    layer input_r_layer = *(l.input_r_layer);
    layer input_h_layer = *(l.input_h_layer);

    layer state_z_layer = *(l.state_z_layer);
    layer state_r_layer = *(l.state_r_layer);
    layer state_h_layer = *(l.state_h_layer);

    fill_cpu(l.outputs * l.batch * l.steps, 0, input_z_layer.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, input_r_layer.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, input_h_layer.delta, 1);

    fill_cpu(l.outputs * l.batch * l.steps, 0, state_z_layer.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, state_r_layer.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, state_h_layer.delta, 1);
    if(state.train) {
        fill_cpu(l.outputs * l.batch * l.steps, 0, l.delta, 1);
        copy_cpu(l.outputs*l.batch, l.state, 1, l.prev_state, 1);
    }

    for (i = 0; i < l.steps; ++i) {
        s.input = l.state;
        forward_connected_layer(state_z_layer, s);
        forward_connected_layer(state_r_layer, s);

        s.input = state.input;
        forward_connected_layer(input_z_layer, s);
        forward_connected_layer(input_r_layer, s);
        forward_connected_layer(input_h_layer, s);


        copy_cpu(l.outputs*l.batch, input_z_layer.output, 1, l.z_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, state_z_layer.output, 1, l.z_cpu, 1);

        copy_cpu(l.outputs*l.batch, input_r_layer.output, 1, l.r_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, state_r_layer.output, 1, l.r_cpu, 1);

        activate_array(l.z_cpu, l.outputs*l.batch, LOGISTIC);
        activate_array(l.r_cpu, l.outputs*l.batch, LOGISTIC);

        copy_cpu(l.outputs*l.batch, l.state, 1, l.forgot_state, 1);
        mul_cpu(l.outputs*l.batch, l.r_cpu, 1, l.forgot_state, 1);

        s.input = l.forgot_state;
        forward_connected_layer(state_h_layer, s);

        copy_cpu(l.outputs*l.batch, input_h_layer.output, 1, l.h_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, state_h_layer.output, 1, l.h_cpu, 1);

        #ifdef USET
        activate_array(l.h_cpu, l.outputs*l.batch, TANH);
        #else
        activate_array(l.h_cpu, l.outputs*l.batch, LOGISTIC);
        #endif

        weighted_sum_cpu(l.state, l.h_cpu, l.z_cpu, l.outputs*l.batch, l.output);

        copy_cpu(l.outputs*l.batch, l.output, 1, l.state, 1);

        state.input += l.inputs*l.batch;
        l.output += l.outputs*l.batch;
        increment_layer(&input_z_layer, 1);
        increment_layer(&input_r_layer, 1);
        increment_layer(&input_h_layer, 1);

        increment_layer(&state_z_layer, 1);
        increment_layer(&state_r_layer, 1);
        increment_layer(&state_h_layer, 1);
    }
}

void backward_gru_layer(layer l, network_state state)
{
}

#ifdef GPU

void pull_gru_layer(layer l)
{
}

void push_gru_layer(layer l)
{
}

void update_gru_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay)
{
    update_connected_layer_gpu(*(l.input_r_layer), batch, learning_rate, momentum, decay);
    update_connected_layer_gpu(*(l.input_z_layer), batch, learning_rate, momentum, decay);
    update_connected_layer_gpu(*(l.input_h_layer), batch, learning_rate, momentum, decay);
    update_connected_layer_gpu(*(l.state_r_layer), batch, learning_rate, momentum, decay);
    update_connected_layer_gpu(*(l.state_z_layer), batch, learning_rate, momentum, decay);
    update_connected_layer_gpu(*(l.state_h_layer), batch, learning_rate, momentum, decay);
}

void forward_gru_layer_gpu(layer l, network_state state)
{
    network_state s = {0};
    s.train = state.train;
    s.workspace = state.workspace;
    int i;
    layer input_z_layer = *(l.input_z_layer);
    layer input_r_layer = *(l.input_r_layer);
    layer input_h_layer = *(l.input_h_layer);

    layer state_z_layer = *(l.state_z_layer);
    layer state_r_layer = *(l.state_r_layer);
    layer state_h_layer = *(l.state_h_layer);

    fill_ongpu(l.outputs * l.batch * l.steps, 0, input_z_layer.delta_gpu, 1);
    fill_ongpu(l.outputs * l.batch * l.steps, 0, input_r_layer.delta_gpu, 1);
    fill_ongpu(l.outputs * l.batch * l.steps, 0, input_h_layer.delta_gpu, 1);

    fill_ongpu(l.outputs * l.batch * l.steps, 0, state_z_layer.delta_gpu, 1);
    fill_ongpu(l.outputs * l.batch * l.steps, 0, state_r_layer.delta_gpu, 1);
    fill_ongpu(l.outputs * l.batch * l.steps, 0, state_h_layer.delta_gpu, 1);
    if(state.train) {
        fill_ongpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
        copy_ongpu(l.outputs*l.batch, l.state_gpu, 1, l.prev_state_gpu, 1);
    }

    for (i = 0; i < l.steps; ++i) {
        s.input = l.state_gpu;
        forward_connected_layer_gpu(state_z_layer, s);
        forward_connected_layer_gpu(state_r_layer, s);

        s.input = state.input;
        forward_connected_layer_gpu(input_z_layer, s);
        forward_connected_layer_gpu(input_r_layer, s);
        forward_connected_layer_gpu(input_h_layer, s);


        copy_ongpu(l.outputs*l.batch, input_z_layer.output_gpu, 1, l.z_gpu, 1);
        axpy_ongpu(l.outputs*l.batch, 1, state_z_layer.output_gpu, 1, l.z_gpu, 1);

        copy_ongpu(l.outputs*l.batch, input_r_layer.output_gpu, 1, l.r_gpu, 1);
        axpy_ongpu(l.outputs*l.batch, 1, state_r_layer.output_gpu, 1, l.r_gpu, 1);

        activate_array_ongpu(l.z_gpu, l.outputs*l.batch, LOGISTIC);
        activate_array_ongpu(l.r_gpu, l.outputs*l.batch, LOGISTIC);

        copy_ongpu(l.outputs*l.batch, l.state_gpu, 1, l.forgot_state_gpu, 1);
        mul_ongpu(l.outputs*l.batch, l.r_gpu, 1, l.forgot_state_gpu, 1);

        s.input = l.forgot_state_gpu;
        forward_connected_layer_gpu(state_h_layer, s);

        copy_ongpu(l.outputs*l.batch, input_h_layer.output_gpu, 1, l.h_gpu, 1);
        axpy_ongpu(l.outputs*l.batch, 1, state_h_layer.output_gpu, 1, l.h_gpu, 1);

        #ifdef USET
        activate_array_ongpu(l.h_gpu, l.outputs*l.batch, TANH);
        #else
        activate_array_ongpu(l.h_gpu, l.outputs*l.batch, LOGISTIC);
        #endif

        weighted_sum_gpu(l.state_gpu, l.h_gpu, l.z_gpu, l.outputs*l.batch, l.output_gpu);

        copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.state_gpu, 1);

        state.input += l.inputs*l.batch;
        l.output_gpu += l.outputs*l.batch;
        increment_layer(&input_z_layer, 1);
        increment_layer(&input_r_layer, 1);
        increment_layer(&input_h_layer, 1);

        increment_layer(&state_z_layer, 1);
        increment_layer(&state_r_layer, 1);
        increment_layer(&state_h_layer, 1);
    }
}

void backward_gru_layer_gpu(layer l, network_state state)
{
    network_state s = {0};
    s.train = state.train;
    s.workspace = state.workspace;
    int i;
    layer input_z_layer = *(l.input_z_layer);
    layer input_r_layer = *(l.input_r_layer);
    layer input_h_layer = *(l.input_h_layer);

    layer state_z_layer = *(l.state_z_layer);
    layer state_r_layer = *(l.state_r_layer);
    layer state_h_layer = *(l.state_h_layer);

    increment_layer(&input_z_layer, l.steps - 1);
    increment_layer(&input_r_layer, l.steps - 1);
    increment_layer(&input_h_layer, l.steps - 1);

    increment_layer(&state_z_layer, l.steps - 1);
    increment_layer(&state_r_layer, l.steps - 1);
    increment_layer(&state_h_layer, l.steps - 1);

    state.input += l.inputs*l.batch*(l.steps-1);
    if(state.delta) state.delta += l.inputs*l.batch*(l.steps-1);
    l.output_gpu += l.outputs*l.batch*(l.steps-1);
    l.delta_gpu += l.outputs*l.batch*(l.steps-1);
    for (i = l.steps-1; i >= 0; --i) {
        if(i != 0) copy_ongpu(l.outputs*l.batch, l.output_gpu - l.outputs*l.batch, 1, l.prev_state_gpu, 1);
        float *prev_delta_gpu = (i == 0) ? 0 : l.delta_gpu - l.outputs*l.batch;

        copy_ongpu(l.outputs*l.batch, input_z_layer.output_gpu, 1, l.z_gpu, 1);
        axpy_ongpu(l.outputs*l.batch, 1, state_z_layer.output_gpu, 1, l.z_gpu, 1);

        copy_ongpu(l.outputs*l.batch, input_r_layer.output_gpu, 1, l.r_gpu, 1);
        axpy_ongpu(l.outputs*l.batch, 1, state_r_layer.output_gpu, 1, l.r_gpu, 1);

        activate_array_ongpu(l.z_gpu, l.outputs*l.batch, LOGISTIC);
        activate_array_ongpu(l.r_gpu, l.outputs*l.batch, LOGISTIC);

        copy_ongpu(l.outputs*l.batch, input_h_layer.output_gpu, 1, l.h_gpu, 1);
        axpy_ongpu(l.outputs*l.batch, 1, state_h_layer.output_gpu, 1, l.h_gpu, 1);

        #ifdef USET
        activate_array_ongpu(l.h_gpu, l.outputs*l.batch, TANH);
        #else
        activate_array_ongpu(l.h_gpu, l.outputs*l.batch, LOGISTIC);
        #endif

        weighted_delta_gpu(l.prev_state_gpu, l.h_gpu, l.z_gpu, prev_delta_gpu, input_h_layer.delta_gpu, input_z_layer.delta_gpu, l.outputs*l.batch, l.delta_gpu);

        #ifdef USET
        gradient_array_ongpu(l.h_gpu, l.outputs*l.batch, TANH, input_h_layer.delta_gpu);
        #else
        gradient_array_ongpu(l.h_gpu, l.outputs*l.batch, LOGISTIC, input_h_layer.delta_gpu);
        #endif

        copy_ongpu(l.outputs*l.batch, input_h_layer.delta_gpu, 1, state_h_layer.delta_gpu, 1);

        copy_ongpu(l.outputs*l.batch, l.prev_state_gpu, 1, l.forgot_state_gpu, 1);
        mul_ongpu(l.outputs*l.batch, l.r_gpu, 1, l.forgot_state_gpu, 1);
        fill_ongpu(l.outputs*l.batch, 0, l.forgot_delta_gpu, 1);

        s.input = l.forgot_state_gpu;
        s.delta = l.forgot_delta_gpu;

        backward_connected_layer_gpu(state_h_layer, s);
        if(prev_delta_gpu) mult_add_into_gpu(l.outputs*l.batch, l.forgot_delta_gpu, l.r_gpu, prev_delta_gpu);
        mult_add_into_gpu(l.outputs*l.batch, l.forgot_delta_gpu, l.prev_state_gpu, input_r_layer.delta_gpu);

        gradient_array_ongpu(l.r_gpu, l.outputs*l.batch, LOGISTIC, input_r_layer.delta_gpu);
        copy_ongpu(l.outputs*l.batch, input_r_layer.delta_gpu, 1, state_r_layer.delta_gpu, 1);

        gradient_array_ongpu(l.z_gpu, l.outputs*l.batch, LOGISTIC, input_z_layer.delta_gpu);
        copy_ongpu(l.outputs*l.batch, input_z_layer.delta_gpu, 1, state_z_layer.delta_gpu, 1);

        s.input = l.prev_state_gpu;
        s.delta = prev_delta_gpu;

        backward_connected_layer_gpu(state_r_layer, s);
        backward_connected_layer_gpu(state_z_layer, s);

        s.input = state.input;
        s.delta = state.delta;

        backward_connected_layer_gpu(input_h_layer, s);
        backward_connected_layer_gpu(input_r_layer, s);
        backward_connected_layer_gpu(input_z_layer, s);


        state.input -= l.inputs*l.batch;
        if(state.delta) state.delta -= l.inputs*l.batch;
        l.output_gpu -= l.outputs*l.batch;
        l.delta_gpu -= l.outputs*l.batch;
        increment_layer(&input_z_layer, -1);
        increment_layer(&input_r_layer, -1);
        increment_layer(&input_h_layer, -1);

        increment_layer(&state_z_layer, -1);
        increment_layer(&state_r_layer, -1);
        increment_layer(&state_h_layer, -1);
    }
}
#endif
