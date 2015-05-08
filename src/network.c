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
#include "normalization_layer.h"
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
        case NORMALIZATION:
            return "normalization";
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
    network net;
    net.n = n;
    net.layers = calloc(net.n, sizeof(void *));
    net.types = calloc(net.n, sizeof(LAYER_TYPE));
    net.outputs = 0;
    net.output = 0;
    net.seen = 0;
    net.batch = 0;
    net.inputs = 0;
    net.h = net.w = net.c = 0;
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
        if(net.types[i] == CONVOLUTIONAL){
            forward_convolutional_layer(*(convolutional_layer *)net.layers[i], state);
        }
        else if(net.types[i] == DECONVOLUTIONAL){
            forward_deconvolutional_layer(*(deconvolutional_layer *)net.layers[i], state);
        }
        else if(net.types[i] == DETECTION){
            forward_detection_layer(*(detection_layer *)net.layers[i], state);
        }
        else if(net.types[i] == CONNECTED){
            forward_connected_layer(*(connected_layer *)net.layers[i], state);
        }
        else if(net.types[i] == CROP){
            forward_crop_layer(*(crop_layer *)net.layers[i], state);
        }
        else if(net.types[i] == COST){
            forward_cost_layer(*(cost_layer *)net.layers[i], state);
        }
        else if(net.types[i] == SOFTMAX){
            forward_softmax_layer(*(softmax_layer *)net.layers[i], state);
        }
        else if(net.types[i] == MAXPOOL){
            forward_maxpool_layer(*(maxpool_layer *)net.layers[i], state);
        }
        else if(net.types[i] == NORMALIZATION){
            forward_normalization_layer(*(normalization_layer *)net.layers[i], state);
        }
        else if(net.types[i] == DROPOUT){
            forward_dropout_layer(*(dropout_layer *)net.layers[i], state);
        }
        else if(net.types[i] == ROUTE){
            forward_route_layer(*(route_layer *)net.layers[i], net);
        }
        state.input = get_network_output_layer(net, i);
    }
}

void update_network(network net)
{
    int i;
    int update_batch = net.batch*net.subdivisions;
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            update_convolutional_layer(layer, update_batch, net.learning_rate, net.momentum, net.decay);
        }
        else if(net.types[i] == DECONVOLUTIONAL){
            deconvolutional_layer layer = *(deconvolutional_layer *)net.layers[i];
            update_deconvolutional_layer(layer, net.learning_rate, net.momentum, net.decay);
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            update_connected_layer(layer, update_batch, net.learning_rate, net.momentum, net.decay);
        }
    }
}

float *get_network_output_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        return ((convolutional_layer *)net.layers[i]) -> output;
    } else if(net.types[i] == DECONVOLUTIONAL){
        return ((deconvolutional_layer *)net.layers[i]) -> output;
    } else if(net.types[i] == MAXPOOL){
        return ((maxpool_layer *)net.layers[i]) -> output;
    } else if(net.types[i] == DETECTION){
        return ((detection_layer *)net.layers[i]) -> output;
    } else if(net.types[i] == SOFTMAX){
        return ((softmax_layer *)net.layers[i]) -> output;
    } else if(net.types[i] == DROPOUT){
        return get_network_output_layer(net, i-1);
    } else if(net.types[i] == CONNECTED){
        return ((connected_layer *)net.layers[i]) -> output;
    } else if(net.types[i] == CROP){
        return ((crop_layer *)net.layers[i]) -> output;
    } else if(net.types[i] == NORMALIZATION){
        return ((normalization_layer *)net.layers[i]) -> output;
    } else if(net.types[i] == ROUTE){
        return ((route_layer *)net.layers[i]) -> output;
    }
    return 0;
}

float *get_network_output(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.types[i] != COST) break;
    return get_network_output_layer(net, i);
}

float *get_network_delta_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return layer.delta;
    } else if(net.types[i] == DECONVOLUTIONAL){
        deconvolutional_layer layer = *(deconvolutional_layer *)net.layers[i];
        return layer.delta;
    } else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        return layer.delta;
    } else if(net.types[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net.layers[i];
        return layer.delta;
    } else if(net.types[i] == DETECTION){
        detection_layer layer = *(detection_layer *)net.layers[i];
        return layer.delta;
    } else if(net.types[i] == DROPOUT){
        if(i == 0) return 0;
        return get_network_delta_layer(net, i-1);
    } else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.delta;
    } else if(net.types[i] == ROUTE){
        return ((route_layer *)net.layers[i]) -> delta;
    }
    return 0;
}

float get_network_cost(network net)
{
    if(net.types[net.n-1] == COST){
        return ((cost_layer *)net.layers[net.n-1])->output[0];
    }
    if(net.types[net.n-1] == DETECTION){
        return ((detection_layer *)net.layers[net.n-1])->cost[0];
    }
    return 0;
}

float *get_network_delta(network net)
{
    return get_network_delta_layer(net, net.n-1);
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
            state.input = get_network_output_layer(net, i-1);
            state.delta = get_network_delta_layer(net, i-1);
        }

        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            backward_convolutional_layer(layer, state);
        } else if(net.types[i] == DECONVOLUTIONAL){
            deconvolutional_layer layer = *(deconvolutional_layer *)net.layers[i];
            backward_deconvolutional_layer(layer, state);
        }
        else if(net.types[i] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net.layers[i];
            if(i != 0) backward_maxpool_layer(layer, state);
        }
        else if(net.types[i] == DROPOUT){
            dropout_layer layer = *(dropout_layer *)net.layers[i];
            backward_dropout_layer(layer, state);
        }
        else if(net.types[i] == DETECTION){
            detection_layer layer = *(detection_layer *)net.layers[i];
            backward_detection_layer(layer, state);
        }
        else if(net.types[i] == NORMALIZATION){
            normalization_layer layer = *(normalization_layer *)net.layers[i];
            if(i != 0) backward_normalization_layer(layer, state);
        }
        else if(net.types[i] == SOFTMAX){
            softmax_layer layer = *(softmax_layer *)net.layers[i];
            if(i != 0) backward_softmax_layer(layer, state);
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            backward_connected_layer(layer, state);
        } else if(net.types[i] == COST){
            cost_layer layer = *(cost_layer *)net.layers[i];
            backward_cost_layer(layer, state);
        } else if(net.types[i] == ROUTE){
            route_layer layer = *(route_layer *)net.layers[i];
            backward_route_layer(layer, net);
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
        if(net->types[i] == CONVOLUTIONAL){
            convolutional_layer *layer = (convolutional_layer *)net->layers[i];
            layer->batch = b;
        }else if(net->types[i] == DECONVOLUTIONAL){
            deconvolutional_layer *layer = (deconvolutional_layer *)net->layers[i];
            layer->batch = b;
        }
        else if(net->types[i] == MAXPOOL){
            maxpool_layer *layer = (maxpool_layer *)net->layers[i];
            layer->batch = b;
        }
        else if(net->types[i] == CONNECTED){
            connected_layer *layer = (connected_layer *)net->layers[i];
            layer->batch = b;
        } else if(net->types[i] == DROPOUT){
            dropout_layer *layer = (dropout_layer *) net->layers[i];
            layer->batch = b;
        } else if(net->types[i] == DETECTION){
            detection_layer *layer = (detection_layer *) net->layers[i];
            layer->batch = b;
        }
        else if(net->types[i] == SOFTMAX){
            softmax_layer *layer = (softmax_layer *)net->layers[i];
            layer->batch = b;
        }
        else if(net->types[i] == COST){
            cost_layer *layer = (cost_layer *)net->layers[i];
            layer->batch = b;
        }
        else if(net->types[i] == CROP){
            crop_layer *layer = (crop_layer *)net->layers[i];
            layer->batch = b;
        }
        else if(net->types[i] == ROUTE){
            route_layer *layer = (route_layer *)net->layers[i];
            layer->batch = b;
        }
    }
}


int get_network_input_size_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return layer.h*layer.w*layer.c;
    }
    if(net.types[i] == DECONVOLUTIONAL){
        deconvolutional_layer layer = *(deconvolutional_layer *)net.layers[i];
        return layer.h*layer.w*layer.c;
    }
    else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        return layer.h*layer.w*layer.c;
    }
    else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.inputs;
    } else if(net.types[i] == DROPOUT){
        dropout_layer layer = *(dropout_layer *) net.layers[i];
        return layer.inputs;
    } else if(net.types[i] == DETECTION){
        detection_layer layer = *(detection_layer *) net.layers[i];
        return layer.inputs;
    } else if(net.types[i] == CROP){
        crop_layer layer = *(crop_layer *) net.layers[i];
        return layer.c*layer.h*layer.w;
    }
    else if(net.types[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net.layers[i];
        return layer.inputs;
    }
    fprintf(stderr, "Can't find input size\n");
    return 0;
}

int get_network_output_size_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        image output = get_convolutional_image(layer);
        return output.h*output.w*output.c;
    }
    else if(net.types[i] == DECONVOLUTIONAL){
        deconvolutional_layer layer = *(deconvolutional_layer *)net.layers[i];
        image output = get_deconvolutional_image(layer);
        return output.h*output.w*output.c;
    }
    else if(net.types[i] == DETECTION){
        detection_layer layer = *(detection_layer *)net.layers[i];
        return get_detection_layer_output_size(layer);
    }
    else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        image output = get_maxpool_image(layer);
        return output.h*output.w*output.c;
    }
    else if(net.types[i] == CROP){
        crop_layer layer = *(crop_layer *) net.layers[i];
        return layer.c*layer.crop_height*layer.crop_width;
    }
    else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.outputs;
    }
    else if(net.types[i] == DROPOUT){
        dropout_layer layer = *(dropout_layer *) net.layers[i];
        return layer.inputs;
    }
    else if(net.types[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net.layers[i];
        return layer.inputs;
    }
    else if(net.types[i] == ROUTE){
        route_layer layer = *(route_layer *)net.layers[i];
        return layer.outputs;
    }
    fprintf(stderr, "Can't find output size\n");
    return 0;
}

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
        }else if(net.types[i] == NORMALIZATION){
            normalization_layer *layer = (normalization_layer *)net.layers[i];
            resize_normalization_layer(layer, h, w);
            image output = get_normalization_image(*layer);
            h = output.h;
            w = output.w;
            c = output.c;
        }else{
            error("Cannot resize this type of layer");
        }
    }
    return 0;
}

int get_network_output_size(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.types[i] != COST) break;
    return get_network_output_size_layer(net, i);
}

int get_network_input_size(network net)
{
    return get_network_input_size_layer(net, 0);
}

detection_layer *get_network_detection_layer(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == DETECTION){
            detection_layer *layer = (detection_layer *)net.layers[i];
            return layer;
        }
    }
    return 0;
}

image get_network_image_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return get_convolutional_image(layer);
    }
    else if(net.types[i] == DECONVOLUTIONAL){
        deconvolutional_layer layer = *(deconvolutional_layer *)net.layers[i];
        return get_deconvolutional_image(layer);
    }
    else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        return get_maxpool_image(layer);
    }
    else if(net.types[i] == NORMALIZATION){
        normalization_layer layer = *(normalization_layer *)net.layers[i];
        return get_normalization_image(layer);
    }
    else if(net.types[i] == DROPOUT){
        return get_network_image_layer(net, i-1);
    }
    else if(net.types[i] == CROP){
        crop_layer layer = *(crop_layer *)net.layers[i];
        return get_crop_image(layer);
    }
    else if(net.types[i] == ROUTE){
        route_layer layer = *(route_layer *)net.layers[i];
        return get_network_image_layer(net, layer.input_layers[0]);
    }
    return make_empty_image(0,0,0);
}

image get_network_image(network net)
{
    int i;
    for(i = net.n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    return make_empty_image(0,0,0);
}

void visualize_network(network net)
{
    image *prev = 0;
    int i;
    char buff[256];
    //show_image(get_network_image_layer(net, 0), "Crop");
    for(i = 0; i < net.n; ++i){
        sprintf(buff, "Layer %d", i);
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            prev = visualize_convolutional_layer(layer, buff, prev);
        }
        if(net.types[i] == NORMALIZATION){
            normalization_layer layer = *(normalization_layer *)net.layers[i];
            visualize_normalization_layer(layer, buff);
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
        float *output = 0;
        int n = 0;
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            output = layer.output;
            image m = get_convolutional_image(layer);
            n = m.h*m.w*m.c;
        }
        else if(net.types[i] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net.layers[i];
            output = layer.output;
            image m = get_maxpool_image(layer);
            n = m.h*m.w*m.c;
        }
        else if(net.types[i] == CROP){
            crop_layer layer = *(crop_layer *)net.layers[i];
            output = layer.output;
            image m = get_crop_image(layer);
            n = m.h*m.w*m.c;
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            output = layer.output;
            n = layer.outputs;
        }
        else if(net.types[i] == SOFTMAX){
            softmax_layer layer = *(softmax_layer *)net.layers[i];
            output = layer.output;
            n = layer.inputs;
        }
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


