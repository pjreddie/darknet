#include "data.h"
#include "utils.h"
#include "image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

void fill_truth_detection(char *path, float *truth, int height, int width, int num_height, int num_width, float scale)
{
    int box_height = height/num_height;
    int box_width = width/num_width;
    char *labelpath = find_replace(path, "imgs", "det");
    labelpath = find_replace(labelpath, ".JPEG", ".txt");
    FILE *file = fopen(labelpath, "r");
    int x, y, h, w;
    while(fscanf(file, "%d %d %d %d", &x, &y, &w, &h) == 4){
        int i = x/box_width;
        int j = y/box_height;
        float dh = (float)(x%box_width)/box_height;
        float dw = (float)(y%box_width)/box_width;
        float sh = h/scale;
        float sw = w/scale;
        int index = (i+j*num_width)*5;
        truth[index++] = 1;
        truth[index++] = dh;
        truth[index++] = dw;
        truth[index++] = sh;
        truth[index++] = sw;
    }
}

void fill_truth(char *path, char **labels, int k, float *truth)
{
    int i;
    memset(truth, 0, k*sizeof(float));
    for(i = 0; i < k; ++i){
        if(strstr(path, labels[i])){
            truth[i] = 1;
        }
    }
}

matrix load_image_paths(char **paths, int n, int h, int w)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = calloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], h, w);
        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_labels_paths(char **paths, int n, char **labels, int k)
{
    matrix y = make_matrix(n, k);
    int i;
    for(i = 0; i < n; ++i){
        fill_truth(paths[i], labels, k, y.vals[i]);
    }
    return y;
}

matrix load_labels_detection(char **paths, int n, int height, int width, int num_height, int num_width, float scale)
{
    int k = num_height*num_width*5;
    matrix y = make_matrix(n, k);
    int i;
    for(i = 0; i < n; ++i){
        fill_truth_detection(paths[i], y.vals[i], height, width, num_height, num_width, scale);
    }
    return y;
}

data load_data_image_pathfile(char *filename, char **labels, int k, int h, int w)
{
    list *plist = get_paths(filename);
    char **paths = (char **)list_to_array(plist);
    int n = plist->size;
    data d;
    d.shallow = 0;
    d.X = load_image_paths(paths, n, h, w);
    d.y = load_labels_paths(paths, n, labels, k);
    free_list_contents(plist);
    free_list(plist);
    free(paths);
    return d;
}

char **get_labels(char *filename)
{
    list *plist = get_paths(filename);
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}

void free_data(data d)
{
    if(!d.shallow){
        free_matrix(d.X);
        free_matrix(d.y);
    }else{
        free(d.X.vals);
        free(d.y.vals);
    }
}

data load_data_detection_random(int n, char **paths, int m, char **labels, int h, int w, int nh, int nw, float scale)
{
    char **random_paths = calloc(n, sizeof(char*));
    int i;
    for(i = 0; i < n; ++i){
        int index = rand()%m;
        random_paths[i] = paths[index];
        if(i == 0) printf("%s\n", paths[index]);
    }
    data d;
    d.shallow = 0;
    d.X = load_image_paths(random_paths, n, h, w);
    d.y = load_labels_detection(random_paths, n, h, w, nh, nw, scale);
    free(random_paths);
    return d;
}

data load_data(char **paths, int n, char **labels, int k, int h, int w)
{
    data d;
    d.shallow = 0;
    d.X = load_image_paths(paths, n, h, w);
    d.y = load_labels_paths(paths, n, labels, k);
    return d;
}

data load_data_random(int n, char **paths, int m, char **labels, int k, int h, int w)
{
    char **random_paths = calloc(n, sizeof(char*));
    int i;
    for(i = 0; i < n; ++i){
        int index = rand()%m;
        random_paths[i] = paths[index];
        if(i == 0) printf("%s\n", paths[index]);
    }
    data d = load_data(random_paths, n, labels, k, h, w);
    free(random_paths);
    return d;
}

data load_categorical_data_csv(char *filename, int target, int k)
{
    data d;
    d.shallow = 0;
    matrix X = csv_to_matrix(filename);
    float *truth_1d = pop_column(&X, target);
    float **truth = one_hot_encode(truth_1d, X.rows, k);
    matrix y;
    y.rows = X.rows;
    y.cols = k;
    y.vals = truth;
    d.X = X;
    d.y = y;
    free(truth_1d);
    return d;
}

data load_cifar10_data(char *filename)
{
    data d;
    d.shallow = 0;
    long i,j;
    matrix X = make_matrix(10000, 3072);
    matrix y = make_matrix(10000, 10);
    d.X = X;
    d.y = y;

    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);
    for(i = 0; i < 10000; ++i){
        unsigned char bytes[3073];
        fread(bytes, 1, 3073, fp);
        int class = bytes[0];
        y.vals[i][class] = 1;
        for(j = 0; j < X.cols; ++j){
            X.vals[i][j] = (double)bytes[j+1];
        }
    }
	translate_data_rows(d, -144);
	scale_data_rows(d, 1./128);
	//normalize_data_rows(d);
    fclose(fp);
    return d;
}

void get_random_batch(data d, int n, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = rand()%d.X.rows;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}

void get_next_batch(data d, int n, int offset, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = offset + j;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
        memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
    }
}


data load_all_cifar10()
{
    data d;
    d.shallow = 0;
    int i,j,b;
    matrix X = make_matrix(50000, 3072);
    matrix y = make_matrix(50000, 10);
    d.X = X;
    d.y = y;


    for(b = 0; b < 5; ++b){
        char buff[256];
        sprintf(buff, "data/cifar10/data_batch_%d.bin", b+1);
        FILE *fp = fopen(buff, "rb");
        if(!fp) file_error(buff);
        for(i = 0; i < 10000; ++i){
            unsigned char bytes[3073];
            fread(bytes, 1, 3073, fp);
            int class = bytes[0];
            y.vals[i+b*10000][class] = 1;
            for(j = 0; j < X.cols; ++j){
                X.vals[i+b*10000][j] = (double)bytes[j+1];
            }
        }
        fclose(fp);
    }
    //normalize_data_rows(d);
    translate_data_rows(d, -144);
    scale_data_rows(d, 1./128);
    return d;
}

void randomize_data(data d)
{
    int i;
    for(i = d.X.rows-1; i > 0; --i){
        int index = rand()%i;
        float *swap = d.X.vals[index];
        d.X.vals[index] = d.X.vals[i];
        d.X.vals[i] = swap;

        swap = d.y.vals[index];
        d.y.vals[index] = d.y.vals[i];
        d.y.vals[i] = swap;
    }
}

void scale_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        scale_array(d.X.vals[i], d.X.cols, s);
    }
}

void translate_data_rows(data d, float s)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        translate_array(d.X.vals[i], d.X.cols, s);
    }
}

void normalize_data_rows(data d)
{
    int i;
    for(i = 0; i < d.X.rows; ++i){
        normalize_array(d.X.vals[i], d.X.cols);
    }
}

data *split_data(data d, int part, int total)
{
    data *split = calloc(2, sizeof(data));
    int i;
    int start = part*d.X.rows/total;
    int end = (part+1)*d.X.rows/total;
    data train;
    data test;
    train.shallow = test.shallow = 1;

    test.X.rows = test.y.rows = end-start;
    train.X.rows = train.y.rows = d.X.rows - (end-start);
    train.X.cols = test.X.cols = d.X.cols;
    train.y.cols = test.y.cols = d.y.cols;

    train.X.vals = calloc(train.X.rows, sizeof(float*));
    test.X.vals = calloc(test.X.rows, sizeof(float*));
    train.y.vals = calloc(train.y.rows, sizeof(float*));
    test.y.vals = calloc(test.y.rows, sizeof(float*));

    for(i = 0; i < start; ++i){
        train.X.vals[i] = d.X.vals[i];
        train.y.vals[i] = d.y.vals[i];
    }
    for(i = start; i < end; ++i){
        test.X.vals[i-start] = d.X.vals[i];
        test.y.vals[i-start] = d.y.vals[i];
    }
    for(i = end; i < d.X.rows; ++i){
        train.X.vals[i-(end-start)] = d.X.vals[i];
        train.y.vals[i-(end-start)] = d.y.vals[i];
    }
    split[0] = train;
    split[1] = test;
    return split;
}

