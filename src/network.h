// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H
#include "darknet.h"

#include "image.h"
#include "layer.h"
#include "data.h"
#include "tree.h"


#ifdef GPU
void pull_network_output(network *net);
#endif

void compare_networks(dn_network *n1, dn_network *n2, dn_data d);
char *get_layer_string(LAYER_TYPE a);

dn_network *make_network(int n);


float network_accuracy_multi(dn_network *net, dn_data d, int n);
int get_predicted_class_network(dn_network *net);
void print_network(dn_network *net);
int resize_network(dn_network *net, int w, int h);
void calc_network_cost(dn_network *net);

#endif

