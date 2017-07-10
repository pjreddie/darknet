#ifndef TREE_H
#define TREE_H
#include "darknet.h"

#ifdef __cplusplus
extern "C" {
#endif

tree *read_tree(char *filename);
int hierarchy_top_prediction(float *predictions, tree *hier, float thresh, int stride);
float get_hierarchy_probability(float *x, tree *hier, int c, int stride);

#ifdef __cplusplus
}
#endif

#endif
