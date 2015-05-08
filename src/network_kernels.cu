extern "C" {
#include <stdio.h>
#include <time.h>

#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "params.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "detection_layer.h"
#include "convolutional_layer.h"
#include "deconvolutional_layer.h"
#include "maxpool_layer.h"
#include "cost_layer.h"
#include "normalization_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
}

float * get_network_output_gpu_layer(network net, int i);
float * get_network_delta_gpu_layer(network net, int i);
float * get_network_output_gpu(network net);

void forward_network_gpu(network net, network_state state)
{
    int i;
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            forward_convolutional_layer_gpu(*(convolutional_layer *)net.layers[i], state);
        }
        else if(net.types[i] == DECONVOLUTIONAL){
            forward_deconvolutional_layer_gpu(*(deconvolutional_layer *)net.layers[i], state);
        }
        else if(net.types[i] == COST){
            forward_cost_layer_gpu(*(cost_layer *)net.layers[i], state);
        }
        else if(net.types[i] == CONNECTED){
            forward_connected_layer_gpu(*(connected_layer *)net.layers[i], state);
        }
        else if(net.types[i] == DETECTION){
            forward_detection_layer_gpu(*(detection_layer *)net.layers[i], state);
        }
        else if(net.types[i] == MAXPOOL){
            forward_maxpool_layer_gpu(*(maxpool_layer *)net.layers[i], state);
        }
        else if(net.types[i] == SOFTMAX){
            forward_softmax_layer_gpu(*(softmax_layer *)net.layers[i], state);
        }
        else if(net.types[i] == DROPOUT){
            forward_dropout_layer_gpu(*(dropout_layer *)net.layers[i], state);
        }
        else if(net.types[i] == CROP){
            forward_crop_layer_gpu(*(crop_layer *)net.layers[i], state);
        }
        else if(net.types[i] == ROUTE){
            forward_route_layer_gpu(*(route_layer *)net.layers[i], net);
        }
        state.input = get_network_output_gpu_layer(net, i);
    }
}

void backward_network_gpu(network net, network_state state)
{
    int i;
    float * original_input = state.input;
    for(i = net.n-1; i >= 0; --i){
        if(i == 0){
            state.input = original_input;
            state.delta = 0;
        }else{
            state.input = get_network_output_gpu_layer(net, i-1);
            state.delta = get_network_delta_gpu_layer(net, i-1);
        }

        if(net.types[i] == CONVOLUTIONAL){
            backward_convolutional_layer_gpu(*(convolutional_layer *)net.layers[i], state);
        }
        else if(net.types[i] == DECONVOLUTIONAL){
            backward_deconvolutional_layer_gpu(*(deconvolutional_layer *)net.layers[i], state);
        }
        else if(net.types[i] == COST){
            backward_cost_layer_gpu(*(cost_layer *)net.layers[i], state);
        }
        else if(net.types[i] == CONNECTED){
            backward_connected_layer_gpu(*(connected_layer *)net.layers[i], state);
        }
        else if(net.types[i] == DETECTION){
            backward_detection_layer_gpu(*(detection_layer *)net.layers[i], state);
        }
        else if(net.types[i] == MAXPOOL){
            backward_maxpool_layer_gpu(*(maxpool_layer *)net.layers[i], state);
        }
        else if(net.types[i] == DROPOUT){
            backward_dropout_layer_gpu(*(dropout_layer *)net.layers[i], state);
        }
        else if(net.types[i] == SOFTMAX){
            backward_softmax_layer_gpu(*(softmax_layer *)net.layers[i], state);
        }
        else if(net.types[i] == ROUTE){
            backward_route_layer_gpu(*(route_layer *)net.layers[i], net);
        }
    }
}

void update_network_gpu(network net)
{
    int i;
    int update_batch = net.batch*net.subdivisions;
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            update_convolutional_layer_gpu(layer, update_batch, net.learning_rate, net.momentum, net.decay);
        }
        else if(net.types[i] == DECONVOLUTIONAL){
            deconvolutional_layer layer = *(deconvolutional_layer *)net.layers[i];
            update_deconvolutional_layer_gpu(layer, net.learning_rate, net.momentum, net.decay);
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            update_connected_layer_gpu(layer, update_batch, net.learning_rate, net.momentum, net.decay);
        }
    }
}

float * get_network_output_gpu_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        return ((convolutional_layer *)net.layers[i]) -> output_gpu;
    }
    else if(net.types[i] == DECONVOLUTIONAL){
        return ((deconvolutional_layer *)net.layers[i]) -> output_gpu;
    }
    else if(net.types[i] == DETECTION){
        return ((detection_layer *)net.layers[i]) -> output_gpu;
    }
    else if(net.types[i] == CONNECTED){
        return ((connected_layer *)net.layers[i]) -> output_gpu;
    }
    else if(net.types[i] == MAXPOOL){
        return ((maxpool_layer *)net.layers[i]) -> output_gpu;
    }
    else if(net.types[i] == CROP){
        return ((crop_layer *)net.layers[i]) -> output_gpu;
    }
    else if(net.types[i] == SOFTMAX){
        return ((softmax_layer *)net.layers[i]) -> output_gpu;
    }
    else if(net.types[i] == ROUTE){
        return ((route_layer *)net.layers[i]) -> output_gpu;
    }
    else if(net.types[i] == DROPOUT){
        return get_network_output_gpu_layer(net, i-1);
    }
    return 0;
}

float * get_network_delta_gpu_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return layer.delta_gpu;
    }
    else if(net.types[i] == DETECTION){
        detection_layer layer = *(detection_layer *)net.layers[i];
        return layer.delta_gpu;
    }
    else if(net.types[i] == DECONVOLUTIONAL){
        deconvolutional_layer layer = *(deconvolutional_layer *)net.layers[i];
        return layer.delta_gpu;
    }
    else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.delta_gpu;
    }
    else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        return layer.delta_gpu;
    }
    else if(net.types[i] == ROUTE){
        route_layer layer = *(route_layer *)net.layers[i];
        return layer.delta_gpu;
    }
    else if(net.types[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net.layers[i];
        return layer.delta_gpu;
    } else if(net.types[i] == DROPOUT){
        if(i == 0) return 0;
        return get_network_delta_gpu_layer(net, i-1);
    }
    return 0;
}

float train_network_datum_gpu(network net, float *x, float *y)
{
    network_state state;
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = get_network_output_size(net)*net.batch;
    if(!*net.input_gpu){
        *net.input_gpu = cuda_make_array(x, x_size);
        *net.truth_gpu = cuda_make_array(y, y_size);
    }else{
        cuda_push_array(*net.input_gpu, x, x_size);
        cuda_push_array(*net.truth_gpu, y, y_size);
    }
    state.input = *net.input_gpu;
    state.truth = *net.truth_gpu;
    state.train = 1;
    forward_network_gpu(net, state);
    backward_network_gpu(net, state);
    float error = get_network_cost(net);
    if ((net.seen / net.batch) % net.subdivisions == 0) update_network_gpu(net);

    return error;
}

float *get_network_output_layer_gpu(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return layer.output;
    }
    else if(net.types[i] == DECONVOLUTIONAL){
        deconvolutional_layer layer = *(deconvolutional_layer *)net.layers[i];
        return layer.output;
    }
    else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        cuda_pull_array(layer.output_gpu, layer.output, layer.outputs*layer.batch);
        return layer.output;
    }
    else if(net.types[i] == DETECTION){
        detection_layer layer = *(detection_layer *)net.layers[i];
        int outputs = get_detection_layer_output_size(layer);
        cuda_pull_array(layer.output_gpu, layer.output, outputs*layer.batch);
        return layer.output;
    }
    else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        return layer.output;
    }
    else if(net.types[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net.layers[i];
        pull_softmax_layer_output(layer);
        return layer.output;
    }
    return 0;
}

float *get_network_output_gpu(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.types[i] != COST) break;
    return get_network_output_layer_gpu(net, i);
}

float *network_predict_gpu(network net, float *input)
{
    int size = get_network_input_size(net) * net.batch;
    network_state state;
    state.input = cuda_make_array(input, size);
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    forward_network_gpu(net, state);
    float *out = get_network_output_gpu(net);
    cuda_free(state.input);
    return out;
}

