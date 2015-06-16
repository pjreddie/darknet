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
    matrix X;
    matrix y;
    int shallow;
} data;


void free_data(data d);

void print_letters(float *pred, int n);
data load_data_captcha(char **paths, int n, int m, int k, int w, int h);
data load_data_captcha_encode(char **paths, int n, int m, int w, int h);
data load_data(char **paths, int n, int m, char **labels, int k, int w, int h);
pthread_t load_data_thread(char **paths, int n, int m, char **labels, int k, int w, int h, data *d);
pthread_t load_image_thread(char *path, image *im, image *resized, int w, int h);

pthread_t load_data_detection_thread(int n, char **paths, int m, int classes, int w, int h, int nh, int nw, int background, data *d);
data load_data_detection_jitter_random(int n, char **paths, int m, int classes, int w, int h, int num_boxes, int background);
pthread_t load_data_localization_thread(int n, char **paths, int m, int classes, int w, int h, data *d);

data load_cifar10_data(char *filename);
data load_all_cifar10();

data load_data_writing(char **paths, int n, int m, int w, int h);

list *get_paths(char *filename);
char **get_labels(char *filename);
void get_random_batch(data d, int n, float *X, float *y);
void get_next_batch(data d, int n, int offset, float *X, float *y);
data load_categorical_data_csv(char *filename, int target, int k);
void normalize_data_rows(data d);
void scale_data_rows(data d, float s);
void translate_data_rows(data d, float s);
void randomize_data(data d);
data *split_data(data d, int part, int total);
data concat_data(data d1, data d2);

#endif
