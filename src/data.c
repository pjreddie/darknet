#include "data.h"
#include "list.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

batch make_batch(int n, int k)
{
    batch b;
    b.n = n;
    if(k < 3) k = 1;
    b.images = calloc(n, sizeof(image));
    b.truth = calloc(n, sizeof(double *));
    int i;
    for(i =0 ; i < n; ++i) b.truth[i] = calloc(k, sizeof(double));
    return b;
}

list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

void fill_truth(char *path, char **labels, int k, double *truth)
{
    int i;
    memset(truth, 0, k*sizeof(double));
    for(i = 0; i < k; ++i){
        if(strstr(path, labels[i])){
            truth[i] = 1;
        }
    }
}

batch load_list(list *paths, char **labels, int k)
{
    char *path;
    batch data = make_batch(paths->size, 2);
    node *n = paths->front;
    int i;
    for(i = 0; i < data.n; ++i){
        path = (char *)n->val;
        data.images[i] = load_image(path);
        fill_truth(path, labels, k, data.truth[i]);
        n = n->next;
    }
    return data;
}

batch get_all_data(char *filename, char **labels, int k)
{
    list *paths = get_paths(filename);
    batch b = load_list(paths, labels, k);
    free_list_contents(paths);
    free_list(paths);
    return b;
}

void free_batch(batch b)
{
    int i;
    for(i = 0; i < b.n; ++i){
        free_image(b.images[i]);
        free(b.truth[i]);
    }
    free(b.images);
    free(b.truth);
}

batch get_batch(char *filename, int curr, int total, char **labels, int k)
{
    list *plist = get_paths(filename);
    char **paths = (char **)list_to_array(plist);
    int i;
    int start = curr*plist->size/total;
    int end = (curr+1)*plist->size/total;
    batch b = make_batch(end-start, 2);
    for(i = start; i < end; ++i){
        b.images[i-start] = load_image(paths[i]);
        fill_truth(paths[i], labels, k, b.truth[i-start]);
    }
    free_list_contents(plist);
    free_list(plist);
    free(paths);
    return b;
}

batch random_batch(char *filename, int n, char **labels, int k)
{
    list *plist = get_paths(filename);
    char **paths = (char **)list_to_array(plist);
    int i;
    batch b = make_batch(n, 2);
    for(i = 0; i < n; ++i){
        int index = rand()%plist->size;
        b.images[i] = load_image(paths[index]);
        //scale_image(b.images[i], 1./255.);
        z_normalize_image(b.images[i]);
        fill_truth(paths[index], labels, k, b.truth[i]);
        //print_image(b.images[i]);
    }
    free_list_contents(plist);
    free_list(plist);
    free(paths);
    return b;
}
