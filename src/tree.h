#ifndef TREE_H
#define TREE_H
#include "darknet.h"

int hierarchy_top_prediction(float *predictions, dn_tree *hier, float thresh, int stride);
float get_hierarchy_probability(float *x, dn_tree *hier, int c, int stride);

#endif
