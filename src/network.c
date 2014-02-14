#include <stdio.h>
#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"

#include "connected_layer.h"
#include "convolutional_layer.h"
//#include "old_conv.h"
#include "maxpool_layer.h"
#include "softmax_layer.h"

network make_network(int n)
{
    network net;
    net.n = n;
    net.layers = calloc(net.n, sizeof(void *));
    net.types = calloc(net.n, sizeof(LAYER_TYPE));
    net.outputs = 0;
    net.output = 0;
    return net;
}

void print_convolutional_cfg(FILE *fp, convolutional_layer *l)
{
    int i;
    fprintf(fp, "[convolutional]\n"
                "height=%d\n"
                "width=%d\n"
                "channels=%d\n"
                "filters=%d\n"
                "size=%d\n"
                "stride=%d\n"
                "activation=%s\n",
                l->h, l->w, l->c,
                l->n, l->size, l->stride,
                get_activation_string(l->activation));
    fprintf(fp, "data=");
    for(i = 0; i < l->n; ++i) fprintf(fp, "%g,", l->biases[i]);
    for(i = 0; i < l->n*l->c*l->size*l->size; ++i) fprintf(fp, "%g,", l->filters[i]);
    fprintf(fp, "\n\n");
}
void print_connected_cfg(FILE *fp, connected_layer *l)
{
    int i;
    fprintf(fp, "[connected]\n"
                "input=%d\n"
                "output=%d\n"
                "activation=%s\n",
                l->inputs, l->outputs,
                get_activation_string(l->activation));
    fprintf(fp, "data=");
    for(i = 0; i < l->outputs; ++i) fprintf(fp, "%g,", l->biases[i]);
    for(i = 0; i < l->inputs*l->outputs; ++i) fprintf(fp, "%g,", l->weights[i]);
    fprintf(fp, "\n\n");
}

void print_maxpool_cfg(FILE *fp, maxpool_layer *l)
{
    fprintf(fp, "[maxpool]\n"
                "height=%d\n"
                "width=%d\n"
                "channels=%d\n"
                "stride=%d\n\n",
                l->h, l->w, l->c,
                l->stride);
}

void print_softmax_cfg(FILE *fp, softmax_layer *l)
{
    fprintf(fp, "[softmax]\n"
                "input=%d\n\n",
                l->inputs);
}

void save_network(network net, char *filename)
{
    FILE *fp = fopen(filename, "w");
    if(!fp) file_error(filename);
    int i;
    for(i = 0; i < net.n; ++i)
    {
        if(net.types[i] == CONVOLUTIONAL)
            print_convolutional_cfg(fp, (convolutional_layer *)net.layers[i]);
        else if(net.types[i] == CONNECTED)
            print_connected_cfg(fp, (connected_layer *)net.layers[i]);
        else if(net.types[i] == MAXPOOL)
            print_maxpool_cfg(fp, (maxpool_layer *)net.layers[i]);
        else if(net.types[i] == SOFTMAX)
            print_softmax_cfg(fp, (softmax_layer *)net.layers[i]);
    }
    fclose(fp);
}

void forward_network(network net, float *input)
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
    }
}

void update_network(network net, float step, float momentum, float decay)
{
    int i;
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            update_convolutional_layer(layer, step, momentum, decay);
        }
        else if(net.types[i] == MAXPOOL){
            //maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        }
        else if(net.types[i] == SOFTMAX){
            //maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            update_connected_layer(layer, step, momentum, decay);
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
    } else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.output;
    }
    return 0;
}
float *get_network_output(network net)
{
    return get_network_output_layer(net, net.n-1);
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
    } else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.delta;
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
    int i, k = get_network_output_size(net);
    for(i = 0; i < k; ++i){
        printf("%f, ", out[i]);
        delta[i] = truth[i] - out[i];
        sum += delta[i]*delta[i];
    }
    printf("\n");
    return sum;
}

int get_predicted_class_network(network net)
{
    float *out = get_network_output(net);
    int k = get_network_output_size(net);
    return max_index(out, k);
}

float backward_network(network net, float *input, float *truth)
{
    float error = calculate_error_network(net, truth);
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
            learn_convolutional_layer(layer);
            //learn_convolutional_layer(layer);
            if(i != 0) backward_convolutional_layer(layer, prev_delta);
        }
        else if(net.types[i] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net.layers[i];
            if(i != 0) backward_maxpool_layer(layer, prev_input, prev_delta);
        }
        else if(net.types[i] == SOFTMAX){
            softmax_layer layer = *(softmax_layer *)net.layers[i];
            if(i != 0) backward_softmax_layer(layer, prev_input, prev_delta);
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            learn_connected_layer(layer, prev_input);
            if(i != 0) backward_connected_layer(layer, prev_input, prev_delta);
        }
    }
    return error;
}

float train_network_datum(network net, float *x, float *y, float step, float momentum, float decay)
{
    forward_network(net, x);
    //int class = get_predicted_class_network(net);
    float error = backward_network(net, x, y);
    update_network(net, step, momentum, decay);
    //return (y[class]?1:0);
    return error;
}

float train_network_sgd(network net, data d, int n, float step, float momentum,float decay)
{
    int i;
    float error = 0;
    int correct = 0;
    for(i = 0; i < n; ++i){
        int index = rand()%d.X.rows;
        error += train_network_datum(net, d.X.vals[index], d.y.vals[index], step, momentum, decay);
        float *y = d.y.vals[index];
        int class = get_predicted_class_network(net);
        correct += (y[class]?1:0);
        //printf("%d %f %f\n", i,net.output[0], d.y.vals[index][0]);
        //if((i+1)%10 == 0){
        //    printf("%d: %f\n", (i+1), (float)correct/(i+1));
        //}
    }
    printf("Accuracy: %f\n",(float) correct/n);
    return error/n;
}
float train_network_batch(network net, data d, int n, float step, float momentum,float decay)
{
    int i;
    int correct = 0;
    for(i = 0; i < n; ++i){
        int index = rand()%d.X.rows;
        float *x = d.X.vals[index];
        float *y = d.y.vals[index];
        forward_network(net, x);
        int class = get_predicted_class_network(net);
        backward_network(net, x, y);
        correct += (y[class]?1:0);
    }
    update_network(net, step, momentum, decay);
    return (float)correct/n;

}


void train_network(network net, data d, float step, float momentum, float decay)
{
    int i;
    int correct = 0;
    for(i = 0; i < d.X.rows; ++i){
        correct += train_network_datum(net, d.X.vals[i], d.y.vals[i], step, momentum, decay);
        if(i%100 == 0){
            visualize_network(net);
            cvWaitKey(10);
        }
    }
    visualize_network(net);
    cvWaitKey(100);
    printf("Accuracy: %f\n", (float)correct/d.X.rows);
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
    else if(net.types[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net.layers[i];
        return layer.inputs;
    }
    return 0;
}

int get_network_output_size(network net)
{
    int i = net.n-1;
    return get_network_output_size_layer(net, i);
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
    int i;
    char buff[256];
    for(i = 0; i < net.n; ++i){
        sprintf(buff, "Layer %d", i);
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            visualize_convolutional_layer(layer, buff);
        }
    } 
}

float *network_predict(network net, float *input)
{
    forward_network(net, input);
    float *out = get_network_output(net);
    return out;
}

matrix network_predict_data(network net, data test)
{
    int i,j;
    int k = get_network_output_size(net);
    matrix pred = make_matrix(test.X.rows, k);
    for(i = 0; i < test.X.rows; ++i){
        float *out = network_predict(net, test.X.vals[i]);
        for(j = 0; j < k; ++j){
            pred.vals[i][j] = out[j];
        }
    }
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

