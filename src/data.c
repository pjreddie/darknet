#include "data.h"
#include "utils.h"
#include "image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct load_args{
    char **paths;
    int n;
    int m;
    char **labels;
    int k;
    int h;
    int w;
    int nh;
    int nw;
    int num_boxes;
    int jitter;
    int classes;
    int background;
    data *d;
};

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

char **get_random_paths(char **paths, int n, int m)
{
    char **random_paths = calloc(n, sizeof(char*));
    int i;
    for(i = 0; i < n; ++i){
        int index = rand()%m;
        random_paths[i] = paths[index];
        if(i == 0) printf("%s\n", paths[index]);
    }
    return random_paths;
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

typedef struct box{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} box;

box *read_boxes(char *filename, int *n)
{
    box *boxes = calloc(1, sizeof(box));
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    float x, y, h, w;
    int id;
    int count = 0;
    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5){
        boxes = realloc(boxes, (count+1)*sizeof(box));
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        boxes[count].left   = x - w/2;
        boxes[count].right  = x + w/2;
        boxes[count].top    = y - h/2;
        boxes[count].bottom = y + h/2;
        ++count;
    }
    fclose(file);
    *n = count;
    return boxes;
}

void randomize_boxes(box *b, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        box swap = b[i];
        int index = rand()%n;
        b[i] = b[index];
        b[index] = swap;
    }
}

void fill_truth_detection(char *path, float *truth, int classes, int num_boxes, int flip, int background, float dx, float dy, float sx, float sy)
{
    char *labelpath = find_replace(path, "VOC2012/JPEGImages", "labels");
    labelpath = find_replace(labelpath, ".jpg", ".txt");
    int count = 0;
    box *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    float l,r,t,b;
    int id;
    int i;
    if(background){
        for(i = 0; i < num_boxes*num_boxes*(4+classes+background); i += 4+classes+background){
            truth[i] = 1;
        }
    }
    for(i = 0; i < count; ++i){
        l = boxes[i].left;
        r = boxes[i].right;
        t = boxes[i].top;
        b = boxes[i].bottom;
        id = boxes[i].id;

        if(flip){
            float left = l;
            float right = r;
            r = 1-left;
            l = 1-right;
        }

        l = l*sx-dx;
        r = r*sx-dx;
        t = t*sy-dy;
        b = b*sy-dy;
        
        float x = (l+r)/2.;
        float y = (t+b)/2.;

        if (x < 0 || x >= 1 || y < 0 || y >= 1) continue;

        int i = (int)(x*num_boxes);
        int j = (int)(y*num_boxes);

        l = constrain(0, 1, l);
        r = constrain(0, 1, r);
        t = constrain(0, 1, t);
        b = constrain(0, 1, b);

        int index = (i+j*num_boxes)*(4+classes+background);
        if(truth[index+classes+background+2]) continue;
        if(background) truth[index++] = 0;
        truth[index+id] = 1;
        index += classes;
        truth[index++] = l;
        truth[index++] = r;
        truth[index++] = t;
        truth[index++] = b;
    }
    free(boxes);
}

#define NUMCHARS 37

void print_letters(float *pred, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        int index = max_index(pred+i*NUMCHARS, NUMCHARS);
        printf("%c", int_to_alphanum(index));
    }
    printf("\n");
}

void fill_truth_captcha(char *path, int n, float *truth)
{
    char *begin = strrchr(path, '/');
    ++begin;
    int i;
    for(i = 0; i < strlen(begin) && i < n && begin[i] != '.'; ++i){
        int index = alphanum_to_int(begin[i]);
        if(index > 35) printf("Bad %c\n", begin[i]);
        truth[i*NUMCHARS+index] = 1;
    }
    for(;i < n; ++i){
        truth[i*NUMCHARS + NUMCHARS-1] = 1;
    }
}

data load_data_captcha(char **paths, int n, int m, int k, int h, int w)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d;
    d.shallow = 0;
    d.X = load_image_paths(paths, n, h, w);
    d.y = make_matrix(n, k*NUMCHARS);
    int i;
    for(i = 0; i < n; ++i){
        fill_truth_captcha(paths[i], k, d.y.vals[i]);
    }
    if(m) free(paths);
    return d;
}

data load_data_captcha_encode(char **paths, int n, int m, int h, int w)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d;
    d.shallow = 0;
    d.X = load_image_paths(paths, n, h, w);
    d.X.cols = 17100;
    d.y = d.X;
    if(m) free(paths);
    return d;
}

void fill_truth(char *path, char **labels, int k, float *truth)
{
    int i;
    memset(truth, 0, k*sizeof(float));
    int count = 0;
    for(i = 0; i < k; ++i){
        if(strstr(path, labels[i])){
            truth[i] = 1;
            ++count;
        }
    }
    if(count != 1) printf("%d, %s\n", count, path);
}

matrix load_labels_paths(char **paths, int n, char **labels, int k)
{
    matrix y = make_matrix(n, k);
    int i;
    for(i = 0; i < n && labels; ++i){
        fill_truth(paths[i], labels, k, y.vals[i]);
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

data load_data_detection_jitter_random(int n, char **paths, int m, int classes, int h, int w, int num_boxes, int jitter, int background)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d;
    d.shallow = 0;
    d.X = load_image_paths(random_paths, n, h, w);
    int k = num_boxes*num_boxes*(4+classes+background);
    d.y = make_matrix(n, k);
    for(i = 0; i < n; ++i){
        int px = rand()%jitter;
        px = 0;
        int py = rand()%jitter;
        py = 0;
        float sy = (float) h / (h-jitter);
        float sx = (float) w / (w-jitter);
        float dy = (float) py / (h-jitter);
        float dx = (float) px / (w-jitter);

        int flip = rand()%2;
        fill_truth_detection(random_paths[i], d.y.vals[i], classes, num_boxes, flip, background, dx, dy, sx, sy);
        image a = float_to_image(h, w, 3, d.X.vals[i]);
        if(flip) flip_image(a);
        jitter_image(a, h-jitter, w-jitter, py, px);
    }
    d.X.cols = (h-jitter)*(w-jitter)*3;
    free(random_paths);
    return d;
}

void *load_detection_thread(void *ptr)
{
    printf("Loading data: %d\n", rand());
    struct load_args a = *(struct load_args*)ptr;
    *a.d = load_data_detection_jitter_random(a.n, a.paths, a.m, a.classes, a.h, a.w, a.num_boxes, a.jitter, a.background);
    translate_data_rows(*a.d, -128);
    scale_data_rows(*a.d, 1./128);
    free(ptr);
    return 0;
}

pthread_t load_data_detection_thread(int n, char **paths, int m, int classes, int h, int w, int nh, int nw, int jitter, int background, data *d)
{
    pthread_t thread;
    struct load_args *args = calloc(1, sizeof(struct load_args));
    args->n = n;
    args->paths = paths;
    args->m = m;
    args->h = h;
    args->w = w;
    args->nh = nh;
    args->nw = nw;
    args->num_boxes = nw;
    args->classes = classes;
    args->jitter = jitter;
    args->background = background;
    args->d = d;
    if(pthread_create(&thread, 0, load_detection_thread, args)) {
        error("Thread creation failed");
    }
    return thread;
}

data load_data(char **paths, int n, int m, char **labels, int k, int h, int w)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d;
    d.shallow = 0;
    d.X = load_image_paths(paths, n, h, w);
    d.y = load_labels_paths(paths, n, labels, k);
    if(m) free(paths);
    return d;
}

void *load_in_thread(void *ptr)
{
    struct load_args a = *(struct load_args*)ptr;
    *a.d = load_data(a.paths, a.n, a.m, a.labels, a.k, a.h, a.w);
    translate_data_rows(*a.d, -128);
    scale_data_rows(*a.d, 1./128);
    free(ptr);
    return 0;
}

pthread_t load_data_thread(char **paths, int n, int m, char **labels, int k, int h, int w, data *d)
{
    pthread_t thread;
    struct load_args *args = calloc(1, sizeof(struct load_args));
    args->n = n;
    args->paths = paths;
    args->m = m;
    args->labels = labels;
    args->k = k;
    args->h = h;
    args->w = w;
    args->d = d;
    if(pthread_create(&thread, 0, load_in_thread, args)) {
        error("Thread creation failed");
    }
    return thread;
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

