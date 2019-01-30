#ifndef TREE_H
#define TREE_H
#include "darknet.h"

int hierarchy_top_prediction(float *predictions, tree *hier, float thresh, int stride);
float get_hierarchy_probability(float *x, tree *hier, int c, int stride);

#endif
