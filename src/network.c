#include <stdio.h>
#include "network.h"
#include "image.h"
#include "data.h"

#include "connected_layer.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"

network make_network(int n)
{
    network net;
    net.n = n;
    net.layers = calloc(net.n, sizeof(void *));
    net.types = calloc(net.n, sizeof(LAYER_TYPE));
    return net;
}

void forward_network(network net, double *input)
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
        else if(net.types[i] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net.layers[i];
            forward_maxpool_layer(layer, input);
            input = layer.output;
        }
    }
}

void update_network(network net, double step)
{
    int i;
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            update_convolutional_layer(layer, step);
        }
        else if(net.types[i] == MAXPOOL){
            //maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            update_connected_layer(layer, step, .3, 0);
        }
    }
}

double *get_network_output_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return layer.output;
    } else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        return layer.output;
    } else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.output;
    }
    return 0;
}
double *get_network_output(network net)
{
    return get_network_output_layer(net, net.n-1);
}

double *get_network_delta_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return layer.delta;
    } else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        return layer.delta;
    } else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.delta;
    }
    return 0;
}

double *get_network_delta(network net)
{
    return get_network_delta_layer(net, net.n-1);
}

void learn_network(network net, double *input)
{
    int i;
    double *prev_input;
    double *prev_delta;
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
            learn_convolutional_layer(layer, prev_input);
            if(i != 0) backward_convolutional_layer(layer, prev_input, prev_delta);
        }
        else if(net.types[i] == MAXPOOL){
            //maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            learn_connected_layer(layer, prev_input);
            if(i != 0) backward_connected_layer(layer, prev_input, prev_delta);
        }
    }
}

void train_network_batch(network net, batch b)
{
    int i,j;
    int k = get_network_output_size(net);
    int correct = 0;
    for(i = 0; i < b.n; ++i){
        forward_network(net, b.images[i].data);
        image o = get_network_image(net);
        double *output = get_network_output(net);
        double *delta = get_network_delta(net);
        for(j = 0; j < k; ++j){
            //printf("%f %f\n", b.truth[i][j], output[j]);
            delta[j] = b.truth[i][j]-output[j];
            if(fabs(delta[j]) < .5) ++correct;
            //printf("%f\n",  output[j]);
        }
        learn_network(net, b.images[i].data);
        update_network(net, .00001);
    }
    printf("Accuracy: %f\n", (double)correct/b.n);
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
    return make_image(0,0,0);
}

image get_network_image(network net)
{
    int i;
    for(i = net.n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    return make_image(1,1,1);
}

void visualize_network(network net)
{
    int i;
    for(i = 0; i < 1; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            visualize_convolutional_layer(layer);
        }
    } 
}

