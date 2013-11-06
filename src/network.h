// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H

#include "image.h"

typedef enum {
    CONVOLUTIONAL,
    CONNECTED,
    MAXPOOL
} LAYER_TYPE;

typedef struct {
    int n;
    void **layers;
    LAYER_TYPE *types;
} network;

void run_network(image input, network net);
double *get_network_output(network net);
void learn_network(image input, network net);
void update_network(network net, double step);
image get_network_image(network net);

#endif

