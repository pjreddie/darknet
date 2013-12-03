#ifndef DATA_H
#define DATA_H

#include "image.h"

typedef struct{
    int n;
    image *images;
    double **truth;
} batch;

batch get_all_data(char *filename, char **labels, int k);
batch random_batch(char *filename, int n, char **labels, int k);
batch get_batch(char *filename, int curr, int total, char **labels, int k);
void free_batch(batch b);


#endif
