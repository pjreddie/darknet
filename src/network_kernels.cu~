#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "parser.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "rnn_layer.h"
#include "crnn_layer.h"
#include "detection_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "deconvolutional_layer.h"
#include "maxpool_layer.h"
#include "avgpool_layer.h"
#include "normalization_layer.h"
#include "cost_layer.h"
#include "local_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "blas.h"
}

float * get_network_output_gpu_layer(network net, int i);
float * get_network_delta_gpu_layer(network net, int i);
float * get_network_output_gpu(network net);

void forward_network_gpu(network net, network_state state)
{
    int i;
    for(i = 0; i < net.n; ++i){
        state.index = i;
        layer l = net.layers[i];
        if(l.delta_gpu){
            fill_ongpu(l.outputs * l.batch, 0, l.delta_gpu, 1);
        }
        if(l.type == CONVOLUTIONAL){
            forward_convolutional_layer_gpu(l, state);
        } else if(l.type == DECONVOLUTIONAL){
            forward_deconvolutional_layer_gpu(l, state);
        } else if(l.type == ACTIVE){
            forward_activation_layer_gpu(l, state);
        } else if(l.type == LOCAL){
            forward_local_layer_gpu(l, state);
        } else if(l.type == DETECTION){
            forward_detection_layer_gpu(l, state);
        } else if(l.type == CONNECTED){
            forward_connected_layer_gpu(l, state);
        } else if(l.type == RNN){
            forward_rnn_layer_gpu(l, state);
        } else if(l.type == CRNN){
            forward_crnn_layer_gpu(l, state);
        } else if(l.type == CROP){
            forward_crop_layer_gpu(l, state);
        } else if(l.type == COST){
            forward_cost_layer_gpu(l, state);
        } else if(l.type == SOFTMAX){
            forward_softmax_layer_gpu(l, state);
        } else if(l.type == NORMALIZATION){
            forward_normalization_layer_gpu(l, state);
        } else if(l.type == MAXPOOL){
            forward_maxpool_layer_gpu(l, state);
        } else if(l.type == AVGPOOL){
            forward_avgpool_layer_gpu(l, state);
        } else if(l.type == DROPOUT){
            forward_dropout_layer_gpu(l, state);
        } else if(l.type == ROUTE){
            forward_route_layer_gpu(l, net);
        } else if(l.type == SHORTCUT){
            forward_shortcut_layer_gpu(l, state);
        }
        state.input = l.output_gpu;
    }
}

void backward_network_gpu(network net, network_state state)
{
    int i;
    float * original_input = state.input;
    float * original_delta = state.delta;
    for(i = net.n-1; i >= 0; --i){
        state.index = i;
        layer l = net.layers[i];
        if(i == 0){
            state.input = original_input;
            state.delta = original_delta;
        }else{
            layer prev = net.layers[i-1];
            state.input = prev.output_gpu;
            state.delta = prev.delta_gpu;
        }
        if(l.type == CONVOLUTIONAL){
            backward_convolutional_layer_gpu(l, state);
        } else if(l.type == DECONVOLUTIONAL){
            backward_deconvolutional_layer_gpu(l, state);
        } else if(l.type == ACTIVE){
            backward_activation_layer_gpu(l, state);
        } else if(l.type == LOCAL){
            backward_local_layer_gpu(l, state);
        } else if(l.type == MAXPOOL){
            if(i != 0) backward_maxpool_layer_gpu(l, state);
        } else if(l.type == AVGPOOL){
            if(i != 0) backward_avgpool_layer_gpu(l, state);
        } else if(l.type == DROPOUT){
            backward_dropout_layer_gpu(l, state);
        } else if(l.type == DETECTION){
            backward_detection_layer_gpu(l, state);
        } else if(l.type == NORMALIZATION){
            backward_normalization_layer_gpu(l, state);
        } else if(l.type == SOFTMAX){
            if(i != 0) backward_softmax_layer_gpu(l, state);
        } else if(l.type == CONNECTED){
            backward_connected_layer_gpu(l, state);
        } else if(l.type == RNN){
            backward_rnn_layer_gpu(l, state);
        } else if(l.type == CRNN){
            backward_crnn_layer_gpu(l, state);
        } else if(l.type == COST){
            backward_cost_layer_gpu(l, state);
        } else if(l.type == ROUTE){
            backward_route_layer_gpu(l, net);
        } else if(l.type == SHORTCUT){
            backward_shortcut_layer_gpu(l, state);
        }
    }
}

void update_network_gpu(network net)
{
    int i;
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            update_convolutional_layer_gpu(l, update_batch, rate, net.momentum, net.decay);
        } else if(l.type == DECONVOLUTIONAL){
            update_deconvolutional_layer_gpu(l, rate, net.momentum, net.decay);
        } else if(l.type == CONNECTED){
            update_connected_layer_gpu(l, update_batch, rate, net.momentum, net.decay);
        } else if(l.type == RNN){
            update_rnn_layer_gpu(l, update_batch, rate, net.momentum, net.decay);
        } else if(l.type == CRNN){
            update_crnn_layer_gpu(l, update_batch, rate, net.momentum, net.decay);
        } else if(l.type == LOCAL){
            update_local_layer_gpu(l, update_batch, rate, net.momentum, net.decay);
        }
    }
}

float train_network_datum_gpu(network net, float *x, float *y)
{
    network_state state;
    state.index = 0;
    state.net = net;
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = get_network_output_size(net)*net.batch;
    if(net.layers[net.n-1].type == DETECTION) y_size = net.layers[net.n-1].truths*net.batch;
    if(!*net.input_gpu){
        *net.input_gpu = cuda_make_array(x, x_size);
        *net.truth_gpu = cuda_make_array(y, y_size);
    }else{
        cuda_push_array(*net.input_gpu, x, x_size);
        cuda_push_array(*net.truth_gpu, y, y_size);
    }
    state.input = *net.input_gpu;
    state.delta = 0;
    state.truth = *net.truth_gpu;
    state.train = 1;
    forward_network_gpu(net, state);
    backward_network_gpu(net, state);
    float error = get_network_cost(net);
    if (((*net.seen) / net.batch) % net.subdivisions == 0) update_network_gpu(net);

    return error;
}

float *get_network_output_layer_gpu(network net, int i)
{
    layer l = net.layers[i];
    cuda_pull_array(l.output_gpu, l.output, l.outputs*l.batch);
    return l.output;
}

float *get_network_output_gpu(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return get_network_output_layer_gpu(net, i);
}

float *network_predict_gpu(network net, float *input)
{
    int size = get_network_input_size(net) * net.batch;
    network_state state;
    state.index = 0;
    state.net = net;
    state.input = cuda_make_array(input, size);
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    forward_network_gpu(net, state);
    float *out = get_network_output_gpu(net);
    cuda_free(state.input);
    return out;
}

