#include <stdio.h>
#include <time.h>

#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "cost_layer.h"
#include "normalization_layer.h"
#include "freeweight_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"

#ifdef GPU

void forward_network_gpu(network net, cl_mem input, cl_mem truth, int train)
{
    //printf("start\n");
    int i;
   // printf("Truth: %f\n", cl_checksum(truth, 1000*net.batch));
    for(i = 0; i < net.n; ++i){
        //printf("Truth %i: %f\n", i, cl_checksum(truth, 1000*net.batch));
        //clock_t time = clock();
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            forward_convolutional_layer_gpu(layer, input);
            input = layer.output_cl;
        }
        else if(net.types[i] == COST){
            cost_layer layer = *(cost_layer *)net.layers[i];
            forward_cost_layer_gpu(layer, input, truth);
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            forward_connected_layer_gpu(layer, input);
            input = layer.output_cl;
        }
        else if(net.types[i] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net.layers[i];
            forward_maxpool_layer_gpu(layer, input);
            input = layer.output_cl;
        }
        else if(net.types[i] == SOFTMAX){
            softmax_layer layer = *(softmax_layer *)net.layers[i];
            forward_softmax_layer_gpu(layer, input);
            input = layer.output_cl;
        }
        else if(net.types[i] == DROPOUT){
            if(!train) continue;
            dropout_layer layer = *(dropout_layer *)net.layers[i];
            forward_dropout_layer_gpu(layer, input);
        }
        else if(net.types[i] == CROP){
            crop_layer layer = *(crop_layer *)net.layers[i];
            forward_crop_layer_gpu(layer, input);
            input = layer.output_cl;
        }
        //printf("%d %f\n", i, sec(clock()-time));
        /*
           else if(net.types[i] == CROP){
           crop_layer layer = *(crop_layer *)net.layers[i];
           forward_crop_layer(layer, input);
           input = layer.output;
           }
           else if(net.types[i] == NORMALIZATION){
           normalization_layer layer = *(normalization_layer *)net.layers[i];
           forward_normalization_layer(layer, input);
           input = layer.output;
           }
         */
    }
}

void backward_network_gpu(network net, cl_mem input)
{
    int i;
    cl_mem prev_input;
    cl_mem prev_delta;
    for(i = net.n-1; i >= 0; --i){
        //clock_t time = clock();
        if(i == 0){
            prev_input = input;
            prev_delta = 0;
        }else{
            prev_input = get_network_output_cl_layer(net, i-1);
            prev_delta = get_network_delta_cl_layer(net, i-1);
        }
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            backward_convolutional_layer_gpu(layer, prev_input, prev_delta);
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
        //printf("back: %d %f\n", i, sec(clock()-time));
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
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            update_connected_layer_gpu(layer);
        }
    }
}

cl_mem get_network_output_cl_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return layer.output_cl;
    }
    else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.output_cl;
    }
    else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        return layer.output_cl;
    }
    else if(net.types[i] == CROP){
        crop_layer layer = *(crop_layer *)net.layers[i];
        return layer.output_cl;
    }
    else if(net.types[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net.layers[i];
        return layer.output_cl;
    } else if(net.types[i] == DROPOUT){
        return get_network_output_cl_layer(net, i-1);
    }
    return 0;
}

cl_mem get_network_delta_cl_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return layer.delta_cl;
    }
    else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.delta_cl;
    }
    else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        return layer.delta_cl;
    }
    else if(net.types[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net.layers[i];
        return layer.delta_cl;
    } else if(net.types[i] == DROPOUT){
        return get_network_delta_cl_layer(net, i-1);
    }
    return 0;
}

float train_network_datum_gpu(network net, float *x, float *y)
{
    int x_size = get_network_input_size(net)*net.batch;
    int y_size = get_network_output_size(net)*net.batch;
    //clock_t time = clock();
    if(!*net.input_cl){
        *net.input_cl = cl_make_array(x, x_size);
        *net.truth_cl = cl_make_array(y, y_size);
    }else{
        cl_write_array(*net.input_cl, x, x_size);
        cl_write_array(*net.truth_cl, y, y_size);
    }
    forward_network_gpu(net, *net.input_cl, *net.truth_cl, 1);
    backward_network_gpu(net, *net.input_cl);
    update_network_gpu(net);
    float error = get_network_cost(net);
    return error;
}

float train_network_sgd_gpu(network net, data d, int n)
{
    int batch = net.batch;
    float *X = calloc(batch*d.X.cols, sizeof(float));
    float *y = calloc(batch*d.y.cols, sizeof(float));

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_random_batch(d, batch, X, y);
        float err = train_network_datum_gpu(net, X, y);
        sum += err;
    }
    free(X);
    free(y);
    return (float)sum/(n*batch);
}

float train_network_data_gpu(network net, data d, int n)
{
    int batch = net.batch;
    float *X = calloc(batch*d.X.cols, sizeof(float));
    float *y = calloc(batch*d.y.cols, sizeof(float));

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, X, y);
        float err = train_network_datum_gpu(net, X, y);
        sum += err;
    }
    free(X);
    free(y);
    return (float)sum/(n*batch);
}

float *get_network_output_layer_gpu(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return layer.output;
    }
    else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
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
    cl_mem input_cl = cl_make_array(input, size);
    forward_network_gpu(net, input_cl, 0, 0);
    float *out = get_network_output_gpu(net);
    clReleaseMemObject(input_cl);
    return out;
}

matrix network_predict_data_gpu(network net, data test)
{
    int i,j,b;
    int k = get_network_output_size(net);
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net.batch*test.X.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net.batch){
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict_gpu(net, X);
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;   
}
float network_accuracy_gpu(network net, data d)
{
    matrix guess = network_predict_data_gpu(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

float *network_accuracies_gpu(network net, data d)
{
    static float acc[2];
    matrix guess = network_predict_data_gpu(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess,1);
    acc[1] = matrix_topk_accuracy(d.y, guess,5);
    free_matrix(guess);
    return acc;
}


#else
void forward_network_gpu(network net, cl_mem input, cl_mem truth, int train){}
void backward_network_gpu(network net, cl_mem input){}
void update_network_gpu(network net){}
float train_network_sgd_gpu(network net, data d, int n){return 0;}
float train_network_data_gpu(network net, data d, int n){return 0;}
float *network_predict_gpu(network net, float *input){return 0;}
float network_accuracy_gpu(network net, data d){return 0;}

#endif
