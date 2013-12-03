// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H

#include "image.h"
#include "data.h"

typedef enum {
    CONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX
} LAYER_TYPE;

typedef struct {
    int n;
    void **layers;
    LAYER_TYPE *types;
} network;

network make_network(int n);
void forward_network(network net, double *input);
void learn_network(network net, double *input);
void update_network(network net, double step);
void train_network_batch(network net, batch b);
double *get_network_output(network net);
double *get_network_output_layer(network net, int i);
double *get_network_delta_layer(network net, int i);
double *get_network_delta(network net);
int get_network_output_size_layer(network net, int i);
int get_network_output_size(network net);
image get_network_image(network net);
image get_network_image_layer(network net, int i);
void print_network(network net);
void visualize_network(network net);

#endif

