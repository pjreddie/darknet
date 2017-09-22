// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H
#include "darknet.h"

#include "image.h"
#include "layer.h"
#include "data.h"
#include "tree.h"


#ifdef GPU
float train_network_datum_gpu(network net);
float *network_predict_gpu(network net, float *input);
void pull_network_output(network net);
#endif

void compare_networks(network n1, network n2, data d);
char *get_layer_string(LAYER_TYPE a);

network make_network(int n);


float network_accuracy_multi(network net, data d, int n);
int get_predicted_class_network(network net);
void print_network(network net);
int resize_network(network *net, int w, int h);
void calc_network_cost(network net);

#endif

