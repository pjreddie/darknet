#include "network.h"
#include "image.h"

#include "connected_layer.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"

void run_network(image input, network net)
{
    int i;
    double *input_d = 0;
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

