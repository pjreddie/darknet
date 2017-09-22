#ifndef TREE_H
#define TREE_H
#include "darknet.h"

tree *read_tree(char *filename);
int hierarchy_top_prediction(float *predictions, tree *hier, float thresh, int stride);
float get_hierarchy_probability(float *x, tree *hier, int c, int stride);

#endif
