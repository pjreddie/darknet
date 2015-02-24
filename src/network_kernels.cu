extern "C" {
#include <stdio.h>
#include <time.h>

#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "convolutional_layer.h"
#include "deconvolutional_layer.h"
#include "maxpool_layer.h"
#include "cost_layer.h"
#include "normalization_layer.h"
#include "freeweight_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
}

extern "C" float * get_network_output_gpu_layer(network net, int i);
extern "C" float * get_network_delta_gpu_layer(network net, int i);
float *get_network_output_gpu(network net);

void forward_network_gpu(network net, float * input, float * truth, int train)
{
    int i;
    for(i = 0; i < net.n; ++i){
        //clock_t time = clock();
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            forward_convolutional_layer_gpu(layer, input);
            input = layer.output_gpu;
        }
        else if(net.types[i] == DECONVOLUTIONAL){
            deconvolutional_layer layer = *(deconvolutional_layer *)net.layers[i];
            forward_deconvolutional_layer_gpu(layer, input);
            input = layer.output_gpu;
        }
        else if(net.types[i] == COST){
            cost_layer layer = *(cost_layer *)net.layers[i];
            forward_cost_layer_gpu(layer, input, truth);
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            forward_connected_layer_gpu(layer, input);
            input = layer.output_gpu;
        }
        else if(net.types[i] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net.layers[i];
            forward_maxpool_layer_gpu(layer, input);
            input = layer.output_gpu;
        }
        else if(net.types[i] == SOFTMAX){
            softmax_layer layer = *(softmax_layer *)net.layers[i];
            forward_softmax_layer_gpu(layer, input);
            input = layer.output_gpu;
        }
        else if(net.types[i] == DROPOUT){
            if(!train) continue;
            dropout_layer layer = *(dropout_layer *)net.layers[i];
            forward_dropout_layer_gpu(layer, input);
            input = layer.output_gpu;
        }
        else if(net.types[i] == CROP){
            crop_layer layer = *(crop_layer *)net.layers[i];
            forward_crop_layer_gpu(layer, train, input);
            input = layer.output_gpu;
        }
        //cudaDeviceSynchronize();
        //printf("Forward %d %s %f\n", i, get_layer_string(net.types[i]), sec(clock() - time));
    }
}

void backward_network_gpu(network net, float * input)
{
    int i;
    float * prev_input;
    float * prev_delta;
    for(i = net.n-1; i >= 0; --i){
        //clock_t time = clock();
        if(i == 0){
            prev_input = input;
            prev_delta = 0;
        }else{
            prev_input = get_network_output_gpu_layer(net, i-1);
            prev_delta = get_network_delta_gpu_layer(net, i-1);
        }
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            backward_convolutional_layer_gpu(layer, prev_input, prev_delta);
        }
        else if(net.types[i] == DECONVOLUTIONAL){
            deconvolutional_layer layer = *(deconvolutional_layer *)net.layers[i];
            backward_deconvolutional_layer_gpu(layer, prev_input, prev_delta);
        }
        else if(net.types[i] == COST){
            cost_layer layer = *(cost_layer *)net.layers[i];
            backward_cost_layer_gpu(layer, prev_input, prev_delta);
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            backward_connected_layer_gpu(layer, prev_input, prev_delta);
        }
        else if(net.types[i] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net.layers[i];
            backward_maxpool_layer_gpu(layer, prev_delta);
        }
        else if(net.types[i] == DROPOUT){
            dropout_layer layer = *(dropout_layer *)net.layers[i];
            backward_dropout_layer_gpu(layer, prev_delta);
        }
        else if(net.types[i] == SOFTMAX){
            softmax_layer layer = *(softmax_layer *)net.layers[i];
            backward_softmax_layer_gpu(layer, prev_delta);
        }
        //printf("Backward %d %s %f\n", i, get_layer_string(net.types[i]), sec(clock() - time));
    }
}

void update_network_gpu(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            update_convolutional_layer_gpu(layer);
        }
        else if(net.types[i] == DECONVOLUTIONAL){
            deconvolutional_layer layer = *(deconvolutional_layer *)net.layers[i];
            update_deconvolutional_layer_gpu(layer);
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            update_connected_layer_gpu(layer);
        }
    }
}

float * get_network_output_gpu_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return layer.output_gpu;
    }
    else if(net.types[i] == DECONVOLUTIONAL){
        deconvolutional_layer layer = *(deconvolutional_layer *)net.layers[i];
        return layer.output_gpu;
    }
    else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.output_gpu;
    }
    else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        return layer.output_gpu;
    }
    else if(net.types[i] == CROP){
        crop_layer layer = *(crop_layer *)net.layers[i];
        return layer.output_gpu;
    }
    else if(net.types[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net.layers[i];
        return layer.output_gpu;
    } else if(net.types[i] == DROPOUT){
        dropout_layer layer = *(dropout_layer *)net.layers[i];
        return layer.output_gpu;
    }
    return 0;
}

float * get_network_delta_gpu_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
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
  //clock_t time = clock();
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = get_network_output_size(net)*net.batch;
    if(!*net.input_gpu){
        *net.input_gpu = cuda_make_array(x, x_size);
        *net.truth_gpu = cuda_make_array(y, y_size);
    }else{
        cuda_push_array(*net.input_gpu, x, x_size);
        cuda_push_array(*net.truth_gpu, y, y_size);
    }
  //printf("trans %f\n", sec(clock() - time));
  //time = clock();
    forward_network_gpu(net, *net.input_gpu, *net.truth_gpu, 1);
  //printf("forw %f\n", sec(clock() - time));
  //time = clock();
    backward_network_gpu(net, *net.input_gpu);
  //printf("back %f\n", sec(clock() - time));
  //time = clock();
    update_network_gpu(net);
    float error = get_network_cost(net);

    //print_letters(y, 50);
    //float *out = get_network_output_gpu(net);
    //print_letters(out, 50);
  //printf("updt %f\n", sec(clock() - time));
  //time = clock();
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
    float * input_gpu = cuda_make_array(input, size);
    forward_network_gpu(net, input_gpu, 0, 0);
    float *out = get_network_output_gpu(net);
    cuda_free(input_gpu);
    return out;
}

