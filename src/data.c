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

int get_truth(char *path)
{
    if(strstr(path, "dog")) return 1;
    return 0;
}

batch load_list(list *paths)
{
    char *path;
    batch data = make_batch(paths->size, 2);
    node *n = paths->front;
    int i;
    for(i = 0; i < data.n; ++i){
        path = (char *)n->val;
        data.images[i] = load_image(path);
        data.truth[i][0] = get_truth(path);
        n = n->next;
    }
    return data;
}

batch get_all_data(char *filename)
{
    list *paths = get_paths(filename);
    batch b = load_list(paths);
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

batch get_batch(char *filename, int curr, int total)
{
    list *plist = get_paths(filename);
    char **paths = (char **)list_to_array(plist);
    int i;
    int start = curr*plist->size/total;
    int end = (curr+1)*plist->size/total;
    batch b = make_batch(end-start, 2);
    for(i = start; i < end; ++i){
        b.images[i-start] = load_image(paths[i]);
        b.truth[i-start][0] = get_truth(paths[i]);
    }
    free_list_contents(plist);
    free_list(plist);
    free(paths);
    return b;
}

batch random_batch(char *filename, int n)
{
    list *plist = get_paths(filename);
    char **paths = (char **)list_to_array(plist);
    int i;
    batch b = make_batch(n, 2);
    for(i = 0; i < n; ++i){
        int index = rand()%plist->size;
        b.images[i] = load_image(paths[index]);
        normalize_image(b.images[i]);
        b.truth[i][0] = get_truth(paths[index]);
    }
    free_list_contents(plist);
    free_list(plist);
    free(paths);
    return b;
}
