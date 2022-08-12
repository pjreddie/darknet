#ifndef DATA_H
#define DATA_H
#include <pthread.h>

#include "darknet.h"
#include "matrix.h"
#include "list.h"
#include "image.h"
#include "tree.h"

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
void load_data_blocking(dn_load_args args);


void print_letters(float *pred, int n);
dn_data load_data_captcha(char **paths, int n, int m, int k, int w, int h);
dn_data load_data_captcha_encode(char **paths, int n, int m, int w, int h);
dn_data load_data_detection(int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure);
dn_data load_data_tag(char **paths, int n, int m, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure);
dn_matrix load_image_augment_paths(char **paths, int n, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center);
dn_data load_data_super(char **paths, int n, int m, int w, int h, int scale);
dn_data load_data_augment(char **paths, int n, int m, char **labels, int k, dn_tree *hierarchy, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure, int center);
dn_data load_data_regression(char **paths, int n, int m, int classes, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure);
dn_data load_go(const char *filename);


dn_data load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h);

void get_random_batch(dn_data d, int n, float *X, float *y);
dn_data get_data_part(dn_data d, int part, int total);
dn_data get_random_data(dn_data d, int num);
dn_data load_categorical_data_csv(const char *filename, int target, int k);
void normalize_data_rows(dn_data d);
void scale_data_rows(dn_data d, float s);
void translate_data_rows(dn_data d, float s);
void randomize_data(dn_data d);
dn_data *split_data(dn_data d, int part, int total);
dn_data concat_datas(dn_data *d, int n);
void fill_truth(char *path, char **labels, int k, float *truth);

#endif
