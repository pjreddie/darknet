#include "network.h"
#include "image.h"

#include "connected_layer.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"

void run_network(image input, network net)
{
    int i;
    double *input_d = input.data;
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            run_convolutional_layer(input, layer);
            input = layer.output;
            input_d = layer.output.data;
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            run_connected_layer(input_d, layer);
            input_d = layer.output;
        }
        else if(net.types[i] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net.layers[i];
            run_maxpool_layer(input, layer);
            input = layer.output;
            input_d = layer.output.data;
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
            update_connected_layer(layer, step);
        }
    }
}

void learn_network(image input, network net)
{
    int i;
    image prev;
    double *prev_p;
    for(i = net.n-1; i >= 0; --i){
        if(i == 0){
            prev = input;
            prev_p = prev.data;
        } else if(net.types[i-1] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i-1];
            prev = layer.output;
            prev_p = prev.data;
        } else if(net.types[i-1] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net.layers[i-1];
            prev = layer.output;
            prev_p = prev.data;
        } else if(net.types[i-1] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i-1];
            prev_p = layer.output;
        }

        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            learn_convolutional_layer(prev, layer);
        }
        else if(net.types[i] == MAXPOOL){
            //maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            learn_connected_layer(prev_p, layer);
        }
    }
}

double *get_network_output(network net)
{
    int i = net.n-1;
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return layer.output.data;
    }
    else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        return layer.output.data;
    }
    else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.output;
    }
    return 0;
}
image get_network_image(network net)
{
    int i;
    for(i = net.n-1; i >= 0; --i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            return layer.output;
        }
        else if(net.types[i] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net.layers[i];
            return layer.output;
        }
    }
    return make_image(1,1,1);
}

