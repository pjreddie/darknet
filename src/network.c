#include <stdio.h>
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

network make_network(int n, int batch)
{
    network net;
    net.n = n;
    net.batch = batch;
    net.layers = calloc(net.n, sizeof(void *));
    net.types = calloc(net.n, sizeof(LAYER_TYPE));
    net.outputs = 0;
    net.output = 0;
    #ifdef GPU
    net.input_cl = 0;
    #endif
    return net;
}

#ifdef GPU
void forward_network_gpu(network net, cl_mem input, cl_mem truth, int train)
{
    int i;
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            forward_convolutional_layer_gpu(layer, input);
            input = layer.output_cl;
        }
        else if(net.types[i] == COST){
            cost_layer layer = *(cost_layer *)net.layers[i];
            forward_cost_layer_gpu(layer, input, truth);
        }
        /*
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            forward_connected_layer(layer, input, train);
            input = layer.output;
        }
        else if(net.types[i] == SOFTMAX){
            softmax_layer layer = *(softmax_layer *)net.layers[i];
            forward_softmax_layer(layer, input);
            input = layer.output;
        }
        else if(net.types[i] == CROP){
            crop_layer layer = *(crop_layer *)net.layers[i];
            forward_crop_layer(layer, input);
            input = layer.output;
        }
        else if(net.types[i] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net.layers[i];
            forward_maxpool_layer(layer, input);
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
        if(i == 0){
            prev_input = input;
            prev_delta = 0;
        }else{
            prev_input = get_network_output_cl_layer(net, i-1);
            prev_delta = get_network_delta_cl_layer(net, i-1);
        }
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            backward_convolutional_layer_gpu(layer, prev_delta);
        }
        else if(net.types[i] == COST){
            cost_layer layer = *(cost_layer *)net.layers[i];
            backward_cost_layer_gpu(layer, prev_input, prev_delta);
        }
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
        else if(net.types[i] == MAXPOOL){
            //maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        }
        else if(net.types[i] == SOFTMAX){
            //maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        }
        else if(net.types[i] == NORMALIZATION){
            //maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            update_connected_layer(layer);
        }
    }
}

cl_mem get_network_output_cl_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return layer.output_cl;
    }
    return 0;
}

cl_mem get_network_delta_cl_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return layer.delta_cl;
    }
    return 0;
}

#endif

void forward_network(network net, float *input, float *truth, int train)
{
    int i;
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            forward_convolutional_layer(layer, input);
            input = layer.output;
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            forward_connected_layer(layer, input);
            input = layer.output;
        }
        else if(net.types[i] == CROP){
            crop_layer layer = *(crop_layer *)net.layers[i];
            forward_crop_layer(layer, input);
            input = layer.output;
        }
        else if(net.types[i] == COST){
            cost_layer layer = *(cost_layer *)net.layers[i];
            forward_cost_layer(layer, input, truth);
        }
        else if(net.types[i] == SOFTMAX){
            softmax_layer layer = *(softmax_layer *)net.layers[i];
            forward_softmax_layer(layer, input);
            input = layer.output;
        }
        else if(net.types[i] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net.layers[i];
            forward_maxpool_layer(layer, input);
            input = layer.output;
        }
        else if(net.types[i] == NORMALIZATION){
            normalization_layer layer = *(normalization_layer *)net.layers[i];
            forward_normalization_layer(layer, input);
            input = layer.output;
        }
        else if(net.types[i] == DROPOUT){
            if(!train) continue;
            dropout_layer layer = *(dropout_layer *)net.layers[i];
            forward_dropout_layer(layer, input);
        }
        else if(net.types[i] == FREEWEIGHT){
            if(!train) continue;
            freeweight_layer layer = *(freeweight_layer *)net.layers[i];
            forward_freeweight_layer(layer, input);
        }
    }
}

void update_network(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            update_convolutional_layer(layer);
        }
        else if(net.types[i] == MAXPOOL){
            //maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        }
        else if(net.types[i] == SOFTMAX){
            //maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        }
        else if(net.types[i] == NORMALIZATION){
            //maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            update_connected_layer(layer);
        }
    }
}

float *get_network_output_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return layer.output;
    } else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        return layer.output;
    } else if(net.types[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net.layers[i];
        return layer.output;
    } else if(net.types[i] == DROPOUT){
        return get_network_output_layer(net, i-1);
    } else if(net.types[i] == FREEWEIGHT){
        return get_network_output_layer(net, i-1);
    } else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.output;
    } else if(net.types[i] == NORMALIZATION){
        normalization_layer layer = *(normalization_layer *)net.layers[i];
        return layer.output;
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
    } else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        return layer.delta;
    } else if(net.types[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net.layers[i];
        return layer.delta;
    } else if(net.types[i] == DROPOUT){
        return get_network_delta_layer(net, i-1);
    } else if(net.types[i] == FREEWEIGHT){
        return get_network_delta_layer(net, i-1);
    } else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.delta;
    }
    return 0;
}

float get_network_cost(network net)
{
    if(net.types[net.n-1] == COST){
        return ((cost_layer *)net.layers[net.n-1])->output[0];
    }
    return 0;
}

float *get_network_delta(network net)
{
    return get_network_delta_layer(net, net.n-1);
}

float calculate_error_network(network net, float *truth)
{
    float sum = 0;
    float *delta = get_network_delta(net);
    float *out = get_network_output(net);
    int i;
    for(i = 0; i < get_network_output_size(net)*net.batch; ++i){
        //if(i %get_network_output_size(net) == 0) printf("\n");
        //printf("%5.2f %5.2f, ", out[i], truth[i]);
        //if(i == get_network_output_size(net)) printf("\n");
        delta[i] = truth[i] - out[i];
        //printf("%.10f, ", out[i]);
        sum += delta[i]*delta[i];
    }
    //printf("\n");
    return sum;
}

int get_predicted_class_network(network net)
{
    float *out = get_network_output(net);
    int k = get_network_output_size(net);
    return max_index(out, k);
}

void backward_network(network net, float *input)
{
    int i;
    float *prev_input;
    float *prev_delta;
    for(i = net.n-1; i >= 0; --i){
        if(i == 0){
            prev_input = input;
            prev_delta = 0;
        }else{
            prev_input = get_network_output_layer(net, i-1);
            prev_delta = get_network_delta_layer(net, i-1);
        }
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            backward_convolutional_layer(layer, prev_delta);
        }
        else if(net.types[i] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net.layers[i];
            if(i != 0) backward_maxpool_layer(layer, prev_input, prev_delta);
        }
        else if(net.types[i] == NORMALIZATION){
            normalization_layer layer = *(normalization_layer *)net.layers[i];
            if(i != 0) backward_normalization_layer(layer, prev_input, prev_delta);
        }
        else if(net.types[i] == SOFTMAX){
            softmax_layer layer = *(softmax_layer *)net.layers[i];
            if(i != 0) backward_softmax_layer(layer, prev_input, prev_delta);
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            backward_connected_layer(layer, prev_input, prev_delta);
        }
        else if(net.types[i] == COST){
            cost_layer layer = *(cost_layer *)net.layers[i];
            backward_cost_layer(layer, prev_input, prev_delta);
        }
    }
}

float train_network_datum(network net, float *x, float *y)
{
    forward_network(net, x, y, 1);
    //int class = get_predicted_class_network(net);
    backward_network(net, x);
    float error = get_network_cost(net);
    update_network(net);
    //return (y[class]?1:0);
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
        get_batch(d, batch, X, y);
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
    float sum = 0;
    int batch = 2;
    for(i = 0; i < n; ++i){
        for(j = 0; j < batch; ++j){
            int index = rand()%d.X.rows;
            float *x = d.X.vals[index];
            float *y = d.y.vals[index];
            forward_network(net, x, y, 1);
            backward_network(net, x);
            sum += get_network_cost(net);
        }
        update_network(net);
    }
    return (float)sum/(n*batch);
}


void train_network(network net, data d)
{
    int i;
    int correct = 0;
    for(i = 0; i < d.X.rows; ++i){
        correct += train_network_datum(net, d.X.vals[i], d.y.vals[i]);
        if(i%100 == 0){
            visualize_network(net);
            cvWaitKey(10);
        }
    }
    visualize_network(net);
    cvWaitKey(100);
    fprintf(stderr, "Accuracy: %f\n", (float)correct/d.X.rows);
}

int get_network_input_size_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
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
    }
    else if(net.types[i] == FREEWEIGHT){
        freeweight_layer layer = *(freeweight_layer *) net.layers[i];
        return layer.inputs;
    }
    else if(net.types[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net.layers[i];
        return layer.inputs;
    }
    return 0;
}

int get_network_output_size_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        image output = get_convolutional_image(layer);
        return output.h*output.w*output.c;
    }
    else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        image output = get_maxpool_image(layer);
        return output.h*output.w*output.c;
    }
    else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.outputs;
    }
    else if(net.types[i] == DROPOUT){
        dropout_layer layer = *(dropout_layer *) net.layers[i];
        return layer.inputs;
    }
    else if(net.types[i] == FREEWEIGHT){
        freeweight_layer layer = *(freeweight_layer *) net.layers[i];
        return layer.inputs;
    }
    else if(net.types[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net.layers[i];
        return layer.inputs;
    }
    return 0;
}

int resize_network(network net, int h, int w, int c)
{
    int i;
    for (i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer *layer = (convolutional_layer *)net.layers[i];
            resize_convolutional_layer(layer, h, w, c);
            image output = get_convolutional_image(*layer);
            h = output.h;
            w = output.w;
            c = output.c;
        }else if(net.types[i] == MAXPOOL){
            maxpool_layer *layer = (maxpool_layer *)net.layers[i];
            resize_maxpool_layer(layer, h, w, c);
            image output = get_maxpool_image(*layer);
            h = output.h;
            w = output.w;
            c = output.c;
        }else if(net.types[i] == NORMALIZATION){
            normalization_layer *layer = (normalization_layer *)net.layers[i];
            resize_normalization_layer(layer, h, w, c);
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

image get_network_image_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return get_convolutional_image(layer);
    }
    else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        return get_maxpool_image(layer);
    }
    else if(net.types[i] == NORMALIZATION){
        normalization_layer layer = *(normalization_layer *)net.layers[i];
        return get_normalization_image(layer);
    }
    else if(net.types[i] == CROP){
        crop_layer layer = *(crop_layer *)net.layers[i];
        return get_crop_image(layer);
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
    show_image(get_network_image_layer(net, 0), "Crop");
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

float *network_predict(network net, float *input)
{
    forward_network(net, input, 0, 0);
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
    float *X = calloc(net.batch*test.X.rows, sizeof(float));
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

float network_accuracy(network net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_accuracy(d.y, guess);
    free_matrix(guess);
    return acc;
}

float network_accuracy_multi(network net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_accuracy(d.y, guess);
    free_matrix(guess);
    return acc;
}


