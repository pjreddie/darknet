#ifndef DATA_H
#define DATA_H
#include <pthread.h>

#include "matrix.h"
#include "list.h"
#include "image.h"

extern unsigned int data_seed;

static inline float distance_from_edge(int x, int max)
{
    int dx = (max/2) - x;
    if (dx < 0) dx = -dx;
    dx = (max/2) + 1 - dx;
    dx *= 2;
    float dist = (float)dx/max;
    if (dist > 1) dist = 1;
    return dist;
}

typedef struct{
    int w, h;
    matrix X;
    matrix y;
    int shallow;
} data;

typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA
} data_type;

typedef struct load_args{
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    float jitter;
    data *d;
    image *im;
    image *resized;
    data_type type;
} load_args;

typedef struct{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} box_label;

void free_data(data d);

pthread_t load_data_in_thread(load_args args);

void print_letters(float *pred, int n);
data load_data_captcha(char **paths, int n, int m, int k, int w, int h);
data load_data_captcha_encode(char **paths, int n, int m, int w, int h);
data load_data(char **paths, int n, int m, char **labels, int k, int w, int h);
data load_data_detection(int n, char **paths, int m, int classes, int w, int h, int num_boxes, int background);
data load_data_tag(char **paths, int n, int m, int k, int min, int max, int size);
data load_data_augment(char **paths, int n, int m, char **labels, int k, int min, int max, int size);
data load_go(char *filename);

box_label *read_boxes(char *filename, int *n);
data load_cifar10_data(char *filename);
data load_all_cifar10();

data load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h);

list *get_paths(char *filename);
char **get_labels(char *filename);
void get_random_batch(data d, int n, float *X, float *y);
data get_random_data(data d, int num);
void get_next_batch(data d, int n, int offset, float *X, float *y);
data load_categorical_data_csv(char *filename, int target, int k);
void normalize_data_rows(data d);
void scale_data_rows(data d, float s);
void translate_data_rows(data d, float s);
void randomize_data(data d);
data *split_data(data d, int part, int total);
data concat_data(data d1, data d2);

#endif
