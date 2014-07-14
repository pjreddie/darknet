#ifndef DATA_H
#define DATA_H

#include "matrix.h"
#include "list.h"

typedef struct{
    matrix X;
    matrix y;
    int shallow;
} data;


void free_data(data d);
data load_data_image_pathfile(char *filename, char **labels, int k, int h, int w);
data load_data_image_pathfile_part(char *filename, int part, int total, 
                                    char **labels, int k, int h, int w);
data load_data_image_pathfile_random(char *filename, int n, char **labels, 
                                        int k, int h, int w);
data load_cifar10_data(char *filename);
list *get_paths(char *filename);
data load_categorical_data_csv(char *filename, int target, int k);
void normalize_data_rows(data d);
void scale_data_rows(data d, float s);
void translate_data_rows(data d, float s);
void randomize_data(data d);
data *split_data(data d, int part, int total);

#endif
