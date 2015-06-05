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
#include "detection_layer.h"
#include "maxpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return "convolutional";
        case DECONVOLUTIONAL:
            return "deconvolutional";
        case CONNECTED:
            return "connected";
        case MAXPOOL:
            return "maxpool";
        case SOFTMAX:
            return "softmax";
        case DETECTION:
            return "detection";
        case DROPOUT:
            return "dropout";
        case CROP:
            return "crop";
        case COST:
            return "cost";
        case ROUTE:
            return "route";
        default:
            break;
    }
    return "none";
}

network make_network(int n)
{
    network net = {0};
    net.n = n;
    net.layers = calloc(net.n, sizeof(layer));
    #ifdef GPU
    net.input_gpu = calloc(1, sizeof(float *));
    net.truth_gpu = calloc(1, sizeof(float *));
    #endif
    return net;
}

void forward_network(network net, network_state state)
{
    int i;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            forward_convolutional_layer(l, state);
        } else if(l.type == DECONVOLUTIONAL){
            forward_deconvolutional_layer(l, state);
        } else if(l.type == DETECTION){
            forward_detection_layer(l, state);
        } else if(l.type == CONNECTED){
            forward_connected_layer(l, state);
        } else if(l.type == CROP){
            forward_crop_layer(l, state);
        } else if(l.type == COST){
            forward_cost_layer(l, state);
        } else if(l.type == SOFTMAX){
            forward_softmax_layer(l, state);
        } else if(l.type == MAXPOOL){
            forward_maxpool_layer(l, state);
        } else if(l.type == DROPOUT){
            forward_dropout_layer(l, state);
        } else if(l.type == ROUTE){
            forward_route_layer(l, net);
        }
        state.input = l.output;
    }
}

void update_network(network net)
{
    int i;
    int update_batch = net.batch*net.subdivisions;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            update_convolutional_layer(l, update_batch, net.learning_rate, net.momentum, net.decay);
        } else if(l.type == DECONVOLUTIONAL){
            update_deconvolutional_layer(l, net.learning_rate, net.momentum, net.decay);
        } else if(l.type == CONNECTED){
            update_connected_layer(l, update_batch, net.learning_rate, net.momentum, net.decay);
        }
    }
}

float *get_network_output(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].output;
}

float get_network_cost(network net)
{
    if(net.layers[net.n-1].type == COST){
        return net.layers[net.n-1].output[0];
    }
    if(net.layers[net.n-1].type == DETECTION){
        return net.layers[net.n-1].cost[0];
    }
    return 0;
}

int get_predicted_class_network(network net)
{
    float *out = get_network_output(net);
    int k = get_network_output_size(net);
    return max_index(out, k);
}

void backward_network(network net, network_state state)
{
    int i;
    float *original_input = state.input;
    for(i = net.n-1; i >= 0; --i){
        if(i == 0){
            state.input = original_input;
            state.delta = 0;
        }else{
            layer prev = net.layers[i-1];
            state.input = prev.output;
            state.delta = prev.delta;
        }
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            backward_convolutional_layer(l, state);
        } else if(l.type == DECONVOLUTIONAL){
            backward_deconvolutional_layer(l, state);
        } else if(l.type == MAXPOOL){
            if(i != 0) backward_maxpool_layer(l, state);
        } else if(l.type == DROPOUT){
            backward_dropout_layer(l, state);
        } else if(l.type == DETECTION){
            backward_detection_layer(l, state);
        } else if(l.type == SOFTMAX){
            if(i != 0) backward_softmax_layer(l, state);
        } else if(l.type == CONNECTED){
            backward_connected_layer(l, state);
        } else if(l.type == COST){
            backward_cost_layer(l, state);
        } else if(l.type == ROUTE){
            backward_route_layer(l, net);
        }
    }
}

float train_network_datum(network net, float *x, float *y)
{
    #ifdef GPU
    if(gpu_index >= 0) return train_network_datum_gpu(net, x, y);
    #endif
    network_state state;
    state.input = x;
    state.truth = y;
    state.train = 1;
    forward_network(net, state);
    backward_network(net, state);
    float error = get_network_cost(net);
    if((net.seen/net.batch)%net.subdivisions == 0) update_network(net);
    return error;
}

float train_network_sgd(network net, data d, int n)
{
    int batch = net.batch;
    float *X = calloc(batch*d.X.cols, sizeof(float));
    float *y = calloc(batch*d.y.cols, sizeof(float));

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        net.seen += batch;
        get_random_batch(d, batch, X, y);
        float err = train_network_datum(net, X, y);
        sum += err;
    }
    free(X);
    free(y);
    return (float)sum/(n*batch);
}

float train_network(network net, data d)
{
    int batch = net.batch;
    int n = d.X.rows / batch;
    float *X = calloc(batch*d.X.cols, sizeof(float));
    float *y = calloc(batch*d.y.cols, sizeof(float));

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, X, y);
        net.seen += batch;
        float err = train_network_datum(net, X, y);
        sum += err;
    }
    free(X);
    free(y);
    return (float)sum/(n*batch);
}

float train_network_batch(network net, data d, int n)
{
    int i,j;
    network_state state;
    state.train = 1;
    float sum = 0;
    int batch = 2;
    for(i = 0; i < n; ++i){
        for(j = 0; j < batch; ++j){
            int index = rand()%d.X.rows;
            state.input = d.X.vals[index];
            state.truth = d.y.vals[index];
            forward_network(net, state);
            backward_network(net, state);
            sum += get_network_cost(net);
        }
        update_network(net);
    }
    return (float)sum/(n*batch);
}

void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
    }
}

/*
int resize_network(network net, int h, int w, int c)
{
    fprintf(stderr, "Might be broken, careful!!");
    int i;
    for (i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer *layer = (convolutional_layer *)net.layers[i];
            resize_convolutional_layer(layer, h, w);
            image output = get_convolutional_image(*layer);
            h = output.h;
            w = output.w;
            c = output.c;
        } else if(net.types[i] == DECONVOLUTIONAL){
            deconvolutional_layer *layer = (deconvolutional_layer *)net.layers[i];
            resize_deconvolutional_layer(layer, h, w);
            image output = get_deconvolutional_image(*layer);
            h = output.h;
            w = output.w;
            c = output.c;
        }else if(net.types[i] == MAXPOOL){
            maxpool_layer *layer = (maxpool_layer *)net.layers[i];
            resize_maxpool_layer(layer, h, w);
            image output = get_maxpool_image(*layer);
            h = output.h;
            w = output.w;
            c = output.c;
        }else if(net.types[i] == DROPOUT){
            dropout_layer *layer = (dropout_layer *)net.layers[i];
            resize_dropout_layer(layer, h*w*c);
        }else{
            error("Cannot resize this type of layer");
        }
    }
    return 0;
}
*/

int get_network_output_size(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].outputs;
}

int get_network_input_size(network net)
{
    return net.layers[0].inputs;
}

detection_layer get_network_detection_layer(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].type == DETECTION){
            return net.layers[i];
        }
    }
    fprintf(stderr, "Detection layer not found!!\n");
    detection_layer l = {0};
    return l;
}

image get_network_image_layer(network net, int i)
{
    layer l = net.layers[i];
    if (l.out_w && l.out_h && l.out_c){
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}

image get_network_image(network net)
{
    int i;
    for(i = net.n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    image def = {0};
    return def;
}

void visualize_network(network net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net.n; ++i){
        sprintf(buff, "Layer %d", i);
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    } 
}

void top_predictions(network net, int k, int *index)
{
    int size = get_network_output_size(net);
    float *out = get_network_output(net);
    top_k(out, size, k, index);
}


float *network_predict(network net, float *input)
{
#ifdef GPU
    if(gpu_index >= 0)  return network_predict_gpu(net, input);
#endif

    network_state state;
    state.input = input;
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    forward_network(net, state);
    float *out = get_network_output(net);
    return out;
}

matrix network_predict_data_multi(network net, data test, int n)
{
    int i,j,b,m;
    int k = get_network_output_size(net);
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net.batch*test.X.rows, sizeof(float));
    for(i = 0; i < test.X.rows; i += net.batch){
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        for(m = 0; m < n; ++m){
            float *out = network_predict(net, X);
            for(b = 0; b < net.batch; ++b){
                if(i+b == test.X.rows) break;
                for(j = 0; j < k; ++j){
                    pred.vals[i+b][j] += out[j+b*k]/n;
                }
            }
        }
    }
    free(X);
    return pred;   
}

matrix network_predict_data(network net, data test)
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
        float *out = network_predict(net, X);
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

void print_network(network net)
{
    int i,j;
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}

void compare_networks(network n1, network n2, data test)
{
    matrix g1 = network_predict_data(n1, test);
    matrix g2 = network_predict_data(n2, test);
    int i;
    int a,b,c,d;
    a = b = c = d = 0;
    for(i = 0; i < g1.rows; ++i){
        int truth = max_index(test.y.vals[i], test.y.cols);
        int p1 = max_index(g1.vals[i], g1.cols);
        int p2 = max_index(g2.vals[i], g2.cols);
        if(p1 == truth){
            if(p2 == truth) ++d;
            else ++c;
        }else{
            if(p2 == truth) ++b;
            else ++a;
        }
    }
    printf("%5d %5d\n%5d %5d\n", a, b, c, d);
    float num = pow((abs(b - c) - 1.), 2.);
    float den = b + c;
    printf("%f\n", num/den); 
}

float network_accuracy(network net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

float *network_accuracies(network net, data d)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess,1);
    acc[1] = matrix_topk_accuracy(d.y, guess,5);
    free_matrix(guess);
    return acc;
}


float network_accuracy_multi(network net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}


