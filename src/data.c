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

data load_data_image_paths(char **paths, int n, char **labels, int k, int h, int w)
{
    int i;
    data d;
    d.shallow = 0;
    d.X.rows = n;
    d.X.vals = calloc(d.X.rows, sizeof(float*));
    d.y = make_matrix(n, k);

    for(i = 0; i < n; ++i){
        image im = load_image(paths[i], h, w);
        d.X.vals[i] = im.data;
        d.X.cols = im.h*im.w*im.c;
        fill_truth(paths[i], labels, k, d.y.vals[i]);
    }
    return d;
}

data load_data_image_pathfile(char *filename, char **labels, int k, int h, int w)
{
    list *plist = get_paths(filename);
    char **paths = (char **)list_to_array(plist);
    data d = load_data_image_paths(paths, plist->size, labels, k, h, w);
    free_list_contents(plist);
    free_list(plist);
    free(paths);
    return d;
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

data load_data_image_pathfile_part(char *filename, int part, int total, char **labels, int k, int h, int w)
{
    list *plist = get_paths(filename);
    char **paths = (char **)list_to_array(plist);
    int start = part*plist->size/total;
    int end = (part+1)*plist->size/total;
    data d = load_data_image_paths(paths+start, end-start, labels, k, h, w);
    free_list_contents(plist);
    free_list(plist);
    free(paths);
    return d;
}

data load_data_image_pathfile_random(char *filename, int n, char **labels, int k, int h, int w)
{
    int i;
    list *plist = get_paths(filename);
    char **paths = (char **)list_to_array(plist);
    char **random_paths = calloc(n, sizeof(char*));
    for(i = 0; i < n; ++i){
        int index = rand()%plist->size;
        random_paths[i] = paths[index];
        if(i == 0) printf("%s\n", paths[index]);
    }
    data d = load_data_image_paths(random_paths, n, labels, k, h, w);
    free_list_contents(plist);
    free_list(plist);
    free(paths);
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
    for(i = 0; i < 10000; ++i){
        unsigned char bytes[3073];
        fread(bytes, 1, 3073, fp);
        int class = bytes[0];
        y.vals[i][class] = 1;
        for(j = 0; j < X.cols; ++j){
            X.vals[i][j] = (double)bytes[j+1];
        }
    }
    fclose(fp);
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

