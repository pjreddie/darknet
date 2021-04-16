#include "data.h"
#include "utils.h"
#include "image.h"
#include "dark_cuda.h"
#include "box.h"
#include "http_stream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int check_mistakes;

#define NUMCHARS 37

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

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

/*
char **get_random_paths_indexes(char **paths, int n, int m, int *indexes)
{
    char **random_paths = calloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex);
    for(i = 0; i < n; ++i){
        int index = random_gen()%m;
        indexes[i] = index;
        random_paths[i] = paths[index];
        if(i == 0) printf("%s\n", paths[index]);
    }
    pthread_mutex_unlock(&mutex);
    return random_paths;
}
*/

char **get_sequential_paths(char **paths, int n, int m, int mini_batch, int augment_speed, int contrastive)
{
    int speed = rand_int(1, augment_speed);
    if (speed < 1) speed = 1;
    char** sequentia_paths = (char**)xcalloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex);
    //printf("n = %d, mini_batch = %d \n", n, mini_batch);
    unsigned int *start_time_indexes = (unsigned int *)xcalloc(mini_batch, sizeof(unsigned int));
    for (i = 0; i < mini_batch; ++i) {
        if (contrastive && (i % 2) == 1) start_time_indexes[i] = start_time_indexes[i - 1];
        else start_time_indexes[i] = random_gen() % m;

        //printf(" start_time_indexes[i] = %u, ", start_time_indexes[i]);
    }

    for (i = 0; i < n; ++i) {
        do {
            int time_line_index = i % mini_batch;
            unsigned int index = start_time_indexes[time_line_index] % m;
            start_time_indexes[time_line_index] += speed;

            //int index = random_gen() % m;
            sequentia_paths[i] = paths[index];
            //printf(" index = %d, ", index);
            //if(i == 0) printf("%s\n", paths[index]);
            //printf(" index = %u - grp: %s \n", index, paths[index]);
            if (strlen(sequentia_paths[i]) <= 4) printf(" Very small path to the image: %s \n", sequentia_paths[i]);
        } while (strlen(sequentia_paths[i]) == 0);
    }
    free(start_time_indexes);
    pthread_mutex_unlock(&mutex);
    return sequentia_paths;
}

char **get_random_paths_custom(char **paths, int n, int m, int contrastive)
{
    char** random_paths = (char**)xcalloc(n, sizeof(char*));
    int i;
    pthread_mutex_lock(&mutex);
    int old_index = 0;
    //printf("n = %d \n", n);
    for(i = 0; i < n; ++i){
        do {
            int index = random_gen() % m;
            if (contrastive && (i % 2 == 1)) index = old_index;
            else old_index = index;
            random_paths[i] = paths[index];
            //if(i == 0) printf("%s\n", paths[index]);
            //printf("grp: %s\n", paths[index]);
            if (strlen(random_paths[i]) <= 4) printf(" Very small path to the image: %s \n", random_paths[i]);
        } while (strlen(random_paths[i]) == 0);
    }
    pthread_mutex_unlock(&mutex);
    return random_paths;
}

char **get_random_paths(char **paths, int n, int m)
{
    return get_random_paths_custom(paths, n, m, 0);
}

char **find_replace_paths(char **paths, int n, char *find, char *replace)
{
    char** replace_paths = (char**)xcalloc(n, sizeof(char*));
    int i;
    for(i = 0; i < n; ++i){
        char replaced[4096];
        find_replace(paths[i], find, replace, replaced);
        replace_paths[i] = copy_string(replaced);
    }
    return replace_paths;
}

matrix load_image_paths_gray(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = (float**)xcalloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image(paths[i], w, h, 3);

        image gray = grayscale_image(im);
        free_image(im);
        im = gray;

        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_paths(char **paths, int n, int w, int h)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = (float**)xcalloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], w, h);
        X.vals[i] = im.data;
        X.cols = im.h*im.w*im.c;
    }
    return X;
}

matrix load_image_augment_paths(char **paths, int n, int use_flip, int min, int max, int w, int h, float angle, float aspect, float hue, float saturation, float exposure, int dontuse_opencv, int contrastive)
{
    int i;
    matrix X;
    X.rows = n;
    X.vals = (float**)xcalloc(X.rows, sizeof(float*));
    X.cols = 0;

    for(i = 0; i < n; ++i){
        int size = w > h ? w : h;
        image im;
        const int img_index = (contrastive) ? (i / 2) : i;
        if(dontuse_opencv) im = load_image_stb_resize(paths[img_index], 0, 0, 3);
        else im = load_image_color(paths[img_index], 0, 0);

        image crop = random_augment_image(im, angle, aspect, min, max, size);
        int flip = use_flip ? random_gen() % 2 : 0;
        if (flip)
            flip_image(crop);
        random_distort_image(crop, hue, saturation, exposure);

        image sized = resize_image(crop, w, h);

        //show_image(im, "orig");
        //show_image(sized, "sized");
        //show_image(sized, paths[img_index]);
        //wait_until_press_key_cv();
        //printf("w = %d, h = %d \n", sized.w, sized.h);

        free_image(im);
        free_image(crop);
        X.vals[i] = sized.data;
        X.cols = sized.h*sized.w*sized.c;
    }
    return X;
}


box_label *read_boxes(char *filename, int *n)
{
    box_label* boxes = (box_label*)xcalloc(1, sizeof(box_label));
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Can't open label file. (This can be normal only if you use MSCOCO): %s \n", filename);
        //file_error(filename);
        FILE* fw = fopen("bad.list", "a");
        fwrite(filename, sizeof(char), strlen(filename), fw);
        char *new_line = "\n";
        fwrite(new_line, sizeof(char), strlen(new_line), fw);
        fclose(fw);
        if (check_mistakes) {
            printf("\n Error in read_boxes() \n");
            getchar();
        }

        *n = 0;
        return boxes;
    }
    const int max_obj_img = 4000;// 30000;
    const int img_hash = (custom_hash(filename) % max_obj_img)*max_obj_img;
    //printf(" img_hash = %d, filename = %s; ", img_hash, filename);
    float x, y, h, w;
    int id;
    int count = 0;
    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5){
        boxes = (box_label*)xrealloc(boxes, (count + 1) * sizeof(box_label));
        boxes[count].track_id = count + img_hash;
        //printf(" boxes[count].track_id = %d, count = %d \n", boxes[count].track_id, count);
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

void randomize_boxes(box_label *b, int n)
{
    int i;
    for(i = 0; i < n; ++i){
        box_label swap = b[i];
        int index = random_gen()%n;
        b[i] = b[index];
        b[index] = swap;
    }
}

void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy, int flip)
{
    int i;
    for(i = 0; i < n; ++i){
        if(boxes[i].x == 0 && boxes[i].y == 0) {
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }
        if ((boxes[i].x + boxes[i].w / 2) < 0 || (boxes[i].y + boxes[i].h / 2) < 0 ||
            (boxes[i].x - boxes[i].w / 2) > 1 || (boxes[i].y - boxes[i].h / 2) > 1)
        {
            boxes[i].x = 999999;
            boxes[i].y = 999999;
            boxes[i].w = 999999;
            boxes[i].h = 999999;
            continue;
        }
        boxes[i].left   = boxes[i].left  * sx - dx;
        boxes[i].right  = boxes[i].right * sx - dx;
        boxes[i].top    = boxes[i].top   * sy - dy;
        boxes[i].bottom = boxes[i].bottom* sy - dy;

        if(flip){
            float swap = boxes[i].left;
            boxes[i].left = 1. - boxes[i].right;
            boxes[i].right = 1. - swap;
        }

        boxes[i].left =  constrain(0, 1, boxes[i].left);
        boxes[i].right = constrain(0, 1, boxes[i].right);
        boxes[i].top =   constrain(0, 1, boxes[i].top);
        boxes[i].bottom =   constrain(0, 1, boxes[i].bottom);

        boxes[i].x = (boxes[i].left+boxes[i].right)/2;
        boxes[i].y = (boxes[i].top+boxes[i].bottom)/2;
        boxes[i].w = (boxes[i].right - boxes[i].left);
        boxes[i].h = (boxes[i].bottom - boxes[i].top);

        boxes[i].w = constrain(0, 1, boxes[i].w);
        boxes[i].h = constrain(0, 1, boxes[i].h);
    }
}

void fill_truth_swag(char *path, float *truth, int classes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    replace_image_to_label(path, labelpath);

    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count && i < 30; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .0 || h < .0) continue;

        int index = (4+classes) * i;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;

        if (id < classes) truth[index+id] = 1;
    }
    free(boxes);
}

void fill_truth_region(char *path, float *truth, int classes, int num_boxes, int flip, float dx, float dy, float sx, float sy)
{
    char labelpath[4096];
    replace_image_to_label(path, labelpath);

    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    float x,y,w,h;
    int id;
    int i;

    for (i = 0; i < count; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .001 || h < .001) continue;

        int col = (int)(x*num_boxes);
        int row = (int)(y*num_boxes);

        x = x*num_boxes - col;
        y = y*num_boxes - row;

        int index = (col+row*num_boxes)*(5+classes);
        if (truth[index]) continue;
        truth[index++] = 1;

        if (id < classes) truth[index+id] = 1;
        index += classes;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;
    }
    free(boxes);
}

int fill_truth_detection(const char *path, int num_boxes, int truth_size, float *truth, int classes, int flip, float dx, float dy, float sx, float sy,
    int net_w, int net_h)
{
    char labelpath[4096];
    replace_image_to_label(path, labelpath);

    int count = 0;
    int i;
    box_label *boxes = read_boxes(labelpath, &count);
    int min_w_h = 0;
    float lowest_w = 1.F / net_w;
    float lowest_h = 1.F / net_h;
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    if (count > num_boxes) count = num_boxes;
    float x, y, w, h;
    int id;
    int sub = 0;

    for (i = 0; i < count; ++i) {
        x = boxes[i].x;
        y = boxes[i].y;
        w = boxes[i].w;
        h = boxes[i].h;
        id = boxes[i].id;
        int track_id = boxes[i].track_id;

        // not detect small objects
        //if ((w < 0.001F || h < 0.001F)) continue;
        // if truth (box for object) is smaller than 1x1 pix
        char buff[256];
        if (id >= classes) {
            printf("\n Wrong annotation: class_id = %d. But class_id should be [from 0 to %d], file: %s \n", id, (classes-1), labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: class_id = %d. But class_id should be [from 0 to %d]\" >> bad_label.list", labelpath, id, (classes-1));
            system(buff);
            if (check_mistakes) getchar();
            ++sub;
            continue;
        }
        if ((w < lowest_w || h < lowest_h)) {
            //sprintf(buff, "echo %s \"Very small object: w < lowest_w OR h < lowest_h\" >> bad_label.list", labelpath);
            //system(buff);
            ++sub;
            continue;
        }
        if (x == 999999 || y == 999999) {
            printf("\n Wrong annotation: x = 0, y = 0, < 0 or > 1, file: %s \n", labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: x = 0 or y = 0\" >> bad_label.list", labelpath);
            system(buff);
            ++sub;
            if (check_mistakes) getchar();
            continue;
        }
        if (x <= 0 || x > 1 || y <= 0 || y > 1) {
            printf("\n Wrong annotation: x = %f, y = %f, file: %s \n", x, y, labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: x = %f, y = %f\" >> bad_label.list", labelpath, x, y);
            system(buff);
            ++sub;
            if (check_mistakes) getchar();
            continue;
        }
        if (w > 1) {
            printf("\n Wrong annotation: w = %f, file: %s \n", w, labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: w = %f\" >> bad_label.list", labelpath, w);
            system(buff);
            w = 1;
            if (check_mistakes) getchar();
        }
        if (h > 1) {
            printf("\n Wrong annotation: h = %f, file: %s \n", h, labelpath);
            sprintf(buff, "echo %s \"Wrong annotation: h = %f\" >> bad_label.list", labelpath, h);
            system(buff);
            h = 1;
            if (check_mistakes) getchar();
        }
        if (x == 0) x += lowest_w;
        if (y == 0) y += lowest_h;

        truth[(i-sub)*truth_size +0] = x;
        truth[(i-sub)*truth_size +1] = y;
        truth[(i-sub)*truth_size +2] = w;
        truth[(i-sub)*truth_size +3] = h;
        truth[(i-sub)*truth_size +4] = id;
        truth[(i-sub)*truth_size +5] = track_id;
        //float val = track_id;
        //printf(" i = %d, sub = %d, truth_size = %d, track_id = %d, %f, %f\n", i, sub, truth_size, track_id, truth[(i - sub)*truth_size + 5], val);

        if (min_w_h == 0) min_w_h = w*net_w;
        if (min_w_h > w*net_w) min_w_h = w*net_w;
        if (min_w_h > h*net_h) min_w_h = h*net_h;
    }
    free(boxes);
    return min_w_h;
}


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

data load_data_captcha(char **paths, int n, int m, int k, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = make_matrix(n, k*NUMCHARS);
    int i;
    for(i = 0; i < n; ++i){
        fill_truth_captcha(paths[i], k, d.y.vals[i]);
    }
    if(m) free(paths);
    return d;
}

data load_data_captcha_encode(char **paths, int n, int m, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
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
    if (count != 1) {
        printf("Too many or too few labels: %d, %s\n", count, path);
        count = 0;
        for (i = 0; i < k; ++i) {
            if (strstr(path, labels[i])) {
                printf("\t label %d: %s  \n", count, labels[i]);
                count++;
            }
        }
    }
}

void fill_truth_smooth(char *path, char **labels, int k, float *truth, float label_smooth_eps)
{
    int i;
    memset(truth, 0, k * sizeof(float));
    int count = 0;
    for (i = 0; i < k; ++i) {
        if (strstr(path, labels[i])) {
            truth[i] = (1 - label_smooth_eps);
            ++count;
        }
        else {
            truth[i] = label_smooth_eps / (k - 1);
        }
    }
    if (count != 1) {
        printf("Too many or too few labels: %d, %s\n", count, path);
        count = 0;
        for (i = 0; i < k; ++i) {
            if (strstr(path, labels[i])) {
                printf("\t label %d: %s  \n", count, labels[i]);
                count++;
            }
        }
    }
}

void fill_hierarchy(float *truth, int k, tree *hierarchy)
{
    int j;
    for(j = 0; j < k; ++j){
        if(truth[j]){
            int parent = hierarchy->parent[j];
            while(parent >= 0){
                truth[parent] = 1;
                parent = hierarchy->parent[parent];
            }
        }
    }
    int i;
    int count = 0;
    for(j = 0; j < hierarchy->groups; ++j){
        //printf("%d\n", count);
        int mask = 1;
        for(i = 0; i < hierarchy->group_size[j]; ++i){
            if(truth[count + i]){
                mask = 0;
                break;
            }
        }
        if (mask) {
            for(i = 0; i < hierarchy->group_size[j]; ++i){
                truth[count + i] = SECRET_NUM;
            }
        }
        count += hierarchy->group_size[j];
    }
}

int find_max(float *arr, int size) {
    int i;
    float max = 0;
    int n = 0;
    for (i = 0; i < size; ++i) {
        if (arr[i] > max) {
            max = arr[i];
            n = i;
        }
    }
    return n;
}

matrix load_labels_paths(char **paths, int n, char **labels, int k, tree *hierarchy, float label_smooth_eps, int contrastive)
{
    matrix y = make_matrix(n, k);
    int i;
    if (labels) {
        // supervised learning
        for (i = 0; i < n; ++i) {
            const int img_index = (contrastive) ? (i / 2) : i;
            fill_truth_smooth(paths[img_index], labels, k, y.vals[i], label_smooth_eps);
            //printf(" n = %d, i = %d, img_index = %d, class_id = %d \n", n, i, img_index, find_max(y.vals[i], k));
            if (hierarchy) {
                fill_hierarchy(y.vals[i], k, hierarchy);
            }
        }
    } else {
        // unsupervised learning
        for (i = 0; i < n; ++i) {
            const int img_index = (contrastive) ? (i / 2) : i;
            const uintptr_t path_p = (uintptr_t)paths[img_index];// abs(random_gen());
            const int class_id = path_p % k;
            int l;
            for (l = 0; l < k; ++l) y.vals[i][l] = 0;
            y.vals[i][class_id] = 1;
        }
    }
    return y;
}

matrix load_tags_paths(char **paths, int n, int k)
{
    matrix y = make_matrix(n, k);
    int i;
    int count = 0;
    for(i = 0; i < n; ++i){
        char label[4096];
        find_replace(paths[i], "imgs", "labels", label);
        find_replace(label, "_iconl.jpeg", ".txt", label);
        FILE *file = fopen(label, "r");
        if(!file){
            find_replace(label, "labels", "labels2", label);
            file = fopen(label, "r");
            if(!file) continue;
        }
        ++count;
        int tag;
        while(fscanf(file, "%d", &tag) == 1){
            if(tag < k){
                y.vals[i][tag] = 1;
            }
        }
        fclose(file);
    }
    printf("%d/%d\n", count, n);
    return y;
}

char **get_labels_custom(char *filename, int *size)
{
    list *plist = get_paths(filename);
    if(size) *size = plist->size;
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}

char **get_labels(char *filename)
{
    return get_labels_custom(filename, NULL);
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

data load_data_region(int n, char **paths, int m, int w, int h, int size, int classes, float jitter, float hue, float saturation, float exposure)
{
    char **random_paths = get_random_paths(paths, n, m);
    int i;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = (float**)xcalloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;


    int k = size*size*(5+classes);
    d.y = make_matrix(n, k);
    for(i = 0; i < n; ++i){
        image orig = load_image_color(random_paths[i], 0, 0);

        int oh = orig.h;
        int ow = orig.w;

        int dw = (ow*jitter);
        int dh = (oh*jitter);

        int pleft  = rand_uniform(-dw, dw);
        int pright = rand_uniform(-dw, dw);
        int ptop   = rand_uniform(-dh, dh);
        int pbot   = rand_uniform(-dh, dh);

        int swidth =  ow - pleft - pright;
        int sheight = oh - ptop - pbot;

        float sx = (float)swidth  / ow;
        float sy = (float)sheight / oh;

        int flip = random_gen()%2;
        image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

        float dx = ((float)pleft/ow)/sx;
        float dy = ((float)ptop /oh)/sy;

        image sized = resize_image(cropped, w, h);
        if(flip) flip_image(sized);
        random_distort_image(sized, hue, saturation, exposure);
        d.X.vals[i] = sized.data;

        fill_truth_region(random_paths[i], d.y.vals[i], classes, size, flip, dx, dy, 1./sx, 1./sy);

        free_image(orig);
        free_image(cropped);
    }
    free(random_paths);
    return d;
}

data load_data_compare(int n, char **paths, int m, int classes, int w, int h)
{
    if(m) paths = get_random_paths(paths, 2*n, m);
    int i,j;
    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = (float**)xcalloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*6;

    int k = 2*(classes);
    d.y = make_matrix(n, k);
    for(i = 0; i < n; ++i){
        image im1 = load_image_color(paths[i*2],   w, h);
        image im2 = load_image_color(paths[i*2+1], w, h);

        d.X.vals[i] = (float*)xcalloc(d.X.cols, sizeof(float));
        memcpy(d.X.vals[i],         im1.data, h*w*3*sizeof(float));
        memcpy(d.X.vals[i] + h*w*3, im2.data, h*w*3*sizeof(float));

        int id;
        float iou;

        char imlabel1[4096];
        char imlabel2[4096];
        find_replace(paths[i*2],   "imgs", "labels", imlabel1);
        find_replace(imlabel1, "jpg", "txt", imlabel1);
        FILE *fp1 = fopen(imlabel1, "r");

        while(fscanf(fp1, "%d %f", &id, &iou) == 2){
            if (d.y.vals[i][2*id] < iou) d.y.vals[i][2*id] = iou;
        }

        find_replace(paths[i*2+1], "imgs", "labels", imlabel2);
        find_replace(imlabel2, "jpg", "txt", imlabel2);
        FILE *fp2 = fopen(imlabel2, "r");

        while(fscanf(fp2, "%d %f", &id, &iou) == 2){
            if (d.y.vals[i][2*id + 1] < iou) d.y.vals[i][2*id + 1] = iou;
        }

        for (j = 0; j < classes; ++j){
            if (d.y.vals[i][2*j] > .5 &&  d.y.vals[i][2*j+1] < .5){
                d.y.vals[i][2*j] = 1;
                d.y.vals[i][2*j+1] = 0;
            } else if (d.y.vals[i][2*j] < .5 &&  d.y.vals[i][2*j+1] > .5){
                d.y.vals[i][2*j] = 0;
                d.y.vals[i][2*j+1] = 1;
            } else {
                d.y.vals[i][2*j]   = SECRET_NUM;
                d.y.vals[i][2*j+1] = SECRET_NUM;
            }
        }
        fclose(fp1);
        fclose(fp2);

        free_image(im1);
        free_image(im2);
    }
    if(m) free(paths);
    return d;
}

data load_data_swag(char **paths, int n, int classes, float jitter)
{
    int index = random_gen()%n;
    char *random_path = paths[index];

    image orig = load_image_color(random_path, 0, 0);
    int h = orig.h;
    int w = orig.w;

    data d = {0};
    d.shallow = 0;
    d.w = w;
    d.h = h;

    d.X.rows = 1;
    d.X.vals = (float**)xcalloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*3;

    int k = (4+classes)*30;
    d.y = make_matrix(1, k);

    int dw = w*jitter;
    int dh = h*jitter;

    int pleft  = rand_uniform(-dw, dw);
    int pright = rand_uniform(-dw, dw);
    int ptop   = rand_uniform(-dh, dh);
    int pbot   = rand_uniform(-dh, dh);

    int swidth =  w - pleft - pright;
    int sheight = h - ptop - pbot;

    float sx = (float)swidth  / w;
    float sy = (float)sheight / h;

    int flip = random_gen()%2;
    image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

    float dx = ((float)pleft/w)/sx;
    float dy = ((float)ptop /h)/sy;

    image sized = resize_image(cropped, w, h);
    if(flip) flip_image(sized);
    d.X.vals[0] = sized.data;

    fill_truth_swag(random_path, d.y.vals[0], classes, flip, dx, dy, 1./sx, 1./sy);

    free_image(orig);
    free_image(cropped);

    return d;
}

void blend_truth(float *new_truth, int boxes, int truth_size, float *old_truth)
{
    int count_new_truth = 0;
    int t;
    for (t = 0; t < boxes; ++t) {
        float x = new_truth[t*truth_size];
        if (!x) break;
        count_new_truth++;

    }
    for (t = count_new_truth; t < boxes; ++t) {
        float *new_truth_ptr = new_truth + t*truth_size;
        float *old_truth_ptr = old_truth + (t - count_new_truth)*truth_size;
        float x = old_truth_ptr[0];
        if (!x) break;

        new_truth_ptr[0] = old_truth_ptr[0];
        new_truth_ptr[1] = old_truth_ptr[1];
        new_truth_ptr[2] = old_truth_ptr[2];
        new_truth_ptr[3] = old_truth_ptr[3];
        new_truth_ptr[4] = old_truth_ptr[4];
    }
    //printf("\n was %d bboxes, now %d bboxes \n", count_new_truth, t);
}


void blend_truth_mosaic(float *new_truth, int boxes, int truth_size, float *old_truth, int w, int h, float cut_x, float cut_y, int i_mixup,
    int left_shift, int right_shift, int top_shift, int bot_shift,
    int net_w, int net_h, int mosaic_bound)
{
    const float lowest_w = 1.F / net_w;
    const float lowest_h = 1.F / net_h;

    int count_new_truth = 0;
    int t;
    for (t = 0; t < boxes; ++t) {
        float x = new_truth[t*truth_size];
        if (!x) break;
        count_new_truth++;

    }
    int new_t = count_new_truth;
    for (t = count_new_truth; t < boxes; ++t) {
        float *new_truth_ptr = new_truth + new_t*truth_size;
        new_truth_ptr[0] = 0;
        float *old_truth_ptr = old_truth + (t - count_new_truth)*truth_size;
        float x = old_truth_ptr[0];
        if (!x) break;

        float xb = old_truth_ptr[0];
        float yb = old_truth_ptr[1];
        float wb = old_truth_ptr[2];
        float hb = old_truth_ptr[3];



        // shift 4 images
        if (i_mixup == 0) {
            xb = xb - (float)(w - cut_x - right_shift) / w;
            yb = yb - (float)(h - cut_y - bot_shift) / h;
        }
        if (i_mixup == 1) {
            xb = xb + (float)(cut_x - left_shift) / w;
            yb = yb - (float)(h - cut_y - bot_shift) / h;
        }
        if (i_mixup == 2) {
            xb = xb - (float)(w - cut_x - right_shift) / w;
            yb = yb + (float)(cut_y - top_shift) / h;
        }
        if (i_mixup == 3) {
            xb = xb + (float)(cut_x - left_shift) / w;
            yb = yb + (float)(cut_y - top_shift) / h;
        }

        int left = (xb - wb / 2)*w;
        int right = (xb + wb / 2)*w;
        int top = (yb - hb / 2)*h;
        int bot = (yb + hb / 2)*h;

        if(mosaic_bound)
        {
            // fix out of Mosaic-bound
            float left_bound = 0, right_bound = 0, top_bound = 0, bot_bound = 0;
            if (i_mixup == 0) {
                left_bound = 0;
                right_bound = cut_x;
                top_bound = 0;
                bot_bound = cut_y;
            }
            if (i_mixup == 1) {
                left_bound = cut_x;
                right_bound = w;
                top_bound = 0;
                bot_bound = cut_y;
            }
            if (i_mixup == 2) {
                left_bound = 0;
                right_bound = cut_x;
                top_bound = cut_y;
                bot_bound = h;
            }
            if (i_mixup == 3) {
                left_bound = cut_x;
                right_bound = w;
                top_bound = cut_y;
                bot_bound = h;
            }


            if (left < left_bound) {
                //printf(" i_mixup = %d, left = %d, left_bound = %f \n", i_mixup, left, left_bound);
                left = left_bound;
            }
            if (right > right_bound) {
                //printf(" i_mixup = %d, right = %d, right_bound = %f \n", i_mixup, right, right_bound);
                right = right_bound;
            }
            if (top < top_bound) top = top_bound;
            if (bot > bot_bound) bot = bot_bound;


            xb = ((float)(right + left) / 2) / w;
            wb = ((float)(right - left)) / w;
            yb = ((float)(bot + top) / 2) / h;
            hb = ((float)(bot - top)) / h;
        }
        else
        {
            // fix out of bound
            if (left < 0) {
                float diff = (float)left / w;
                xb = xb - diff / 2;
                wb = wb + diff;
            }

            if (right > w) {
                float diff = (float)(right - w) / w;
                xb = xb - diff / 2;
                wb = wb - diff;
            }

            if (top < 0) {
                float diff = (float)top / h;
                yb = yb - diff / 2;
                hb = hb + diff;
            }

            if (bot > h) {
                float diff = (float)(bot - h) / h;
                yb = yb - diff / 2;
                hb = hb - diff;
            }

            left = (xb - wb / 2)*w;
            right = (xb + wb / 2)*w;
            top = (yb - hb / 2)*h;
            bot = (yb + hb / 2)*h;
        }


        // leave only within the image
        if(left >= 0 && right <= w && top >= 0 && bot <= h &&
            wb > 0 && wb < 1 && hb > 0 && hb < 1 &&
            xb > 0 && xb < 1 && yb > 0 && yb < 1 &&
            wb > lowest_w && hb > lowest_h)
        {
            new_truth_ptr[0] = xb;
            new_truth_ptr[1] = yb;
            new_truth_ptr[2] = wb;
            new_truth_ptr[3] = hb;
            new_truth_ptr[4] = old_truth_ptr[4];
            new_t++;
        }
    }
    //printf("\n was %d bboxes, now %d bboxes \n", count_new_truth, t);
}

#ifdef OPENCV

#include "http_stream.h"

data load_data_detection(int n, char **paths, int m, int w, int h, int c, int boxes, int truth_size, int classes, int use_flip, int use_gaussian_noise, int use_blur, int use_mixup,
    float jitter, float resize, float hue, float saturation, float exposure, int mini_batch, int track, int augment_speed, int letter_box, int mosaic_bound, int contrastive, int contrastive_jit_flip, int contrastive_color, int show_imgs)
{
    const int random_index = random_gen();
    c = c ? c : 3;

    if (use_mixup == 2 || use_mixup == 4) {
        printf("\n cutmix=1 - isn't supported for Detector (use cutmix=1 only for Classifier) \n");
        if (check_mistakes) getchar();
        if(use_mixup == 2) use_mixup = 0;
        else use_mixup = 3;
    }
    if (use_mixup == 3 && letter_box) {
        //printf("\n Combination: letter_box=1 & mosaic=1 - isn't supported, use only 1 of these parameters \n");
        //if (check_mistakes) getchar();
        //exit(0);
    }
    if (random_gen() % 2 == 0) use_mixup = 0;
    int i;

    int *cut_x = NULL, *cut_y = NULL;
    if (use_mixup == 3) {
        cut_x = (int*)calloc(n, sizeof(int));
        cut_y = (int*)calloc(n, sizeof(int));
        const float min_offset = 0.2; // 20%
        for (i = 0; i < n; ++i) {
            cut_x[i] = rand_int(w*min_offset, w*(1 - min_offset));
            cut_y[i] = rand_int(h*min_offset, h*(1 - min_offset));
        }
    }

    data d = {0};
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = (float**)xcalloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*c;

    float r1 = 0, r2 = 0, r3 = 0, r4 = 0, r_scale = 0;
    float resize_r1 = 0, resize_r2 = 0;
    float dhue = 0, dsat = 0, dexp = 0, flip = 0, blur = 0;
    int augmentation_calculated = 0, gaussian_noise = 0;

    d.y = make_matrix(n, truth_size*boxes);
    int i_mixup = 0;
    for (i_mixup = 0; i_mixup <= use_mixup; i_mixup++) {
        if (i_mixup) augmentation_calculated = 0;   // recalculate augmentation for the 2nd sequence if(track==1)

        char **random_paths;
        if (track) random_paths = get_sequential_paths(paths, n, m, mini_batch, augment_speed, contrastive);
        else random_paths = get_random_paths_custom(paths, n, m, contrastive);

        for (i = 0; i < n; ++i) {
            float *truth = (float*)xcalloc(truth_size * boxes, sizeof(float));
            const char *filename = random_paths[i];

            int flag = (c >= 3);
            mat_cv *src;
            src = load_image_mat_cv(filename, flag);
            if (src == NULL) {
                printf("\n Error in load_data_detection() - OpenCV \n");
                fflush(stdout);
                if (check_mistakes) {
                    getchar();
                }
                continue;
            }

            int oh = get_height_mat(src);
            int ow = get_width_mat(src);

            int dw = (ow*jitter);
            int dh = (oh*jitter);

            float resize_down = resize, resize_up = resize;
            if (resize_down > 1.0) resize_down = 1 / resize_down;
            int min_rdw = ow*(1 - (1 / resize_down)) / 2;   // < 0
            int min_rdh = oh*(1 - (1 / resize_down)) / 2;   // < 0

            if (resize_up < 1.0) resize_up = 1 / resize_up;
            int max_rdw = ow*(1 - (1 / resize_up)) / 2;     // > 0
            int max_rdh = oh*(1 - (1 / resize_up)) / 2;     // > 0
            //printf(" down = %f, up = %f \n", (1 - (1 / resize_down)) / 2, (1 - (1 / resize_up)) / 2);

            if (!augmentation_calculated || !track)
            {
                augmentation_calculated = 1;
                resize_r1 = random_float();
                resize_r2 = random_float();

                if (!contrastive || contrastive_jit_flip || i % 2 == 0)
                {
                    r1 = random_float();
                    r2 = random_float();
                    r3 = random_float();
                    r4 = random_float();

                    flip = use_flip ? random_gen() % 2 : 0;
                }

                r_scale = random_float();

                if (!contrastive || contrastive_color || i % 2 == 0)
                {
                    dhue = rand_uniform_strong(-hue, hue);
                    dsat = rand_scale(saturation);
                    dexp = rand_scale(exposure);
                }

                if (use_blur) {
                    int tmp_blur = rand_int(0, 2);  // 0 - disable, 1 - blur background, 2 - blur the whole image
                    if (tmp_blur == 0) blur = 0;
                    else if (tmp_blur == 1) blur = 1;
                    else blur = use_blur;
                }

                if (use_gaussian_noise && rand_int(0, 1) == 1) gaussian_noise = use_gaussian_noise;
                else gaussian_noise = 0;
            }

            int pleft = rand_precalc_random(-dw, dw, r1);
            int pright = rand_precalc_random(-dw, dw, r2);
            int ptop = rand_precalc_random(-dh, dh, r3);
            int pbot = rand_precalc_random(-dh, dh, r4);

            if (resize < 1) {
                // downsize only
                pleft += rand_precalc_random(min_rdw, 0, resize_r1);
                pright += rand_precalc_random(min_rdw, 0, resize_r2);
                ptop += rand_precalc_random(min_rdh, 0, resize_r1);
                pbot += rand_precalc_random(min_rdh, 0, resize_r2);
            }
            else {
                pleft += rand_precalc_random(min_rdw, max_rdw, resize_r1);
                pright += rand_precalc_random(min_rdw, max_rdw, resize_r2);
                ptop += rand_precalc_random(min_rdh, max_rdh, resize_r1);
                pbot += rand_precalc_random(min_rdh, max_rdh, resize_r2);
            }

            //printf("\n pleft = %d, pright = %d, ptop = %d, pbot = %d, ow = %d, oh = %d \n", pleft, pright, ptop, pbot, ow, oh);

            //float scale = rand_precalc_random(.25, 2, r_scale); // unused currently
            //printf(" letter_box = %d \n", letter_box);

            if (letter_box)
            {
                float img_ar = (float)ow / (float)oh;
                float net_ar = (float)w / (float)h;
                float result_ar = img_ar / net_ar;
                //printf(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f, result_ar = %f \n", ow, oh, w, h, img_ar, net_ar, result_ar);
                if (result_ar > 1)  // sheight - should be increased
                {
                    float oh_tmp = ow / net_ar;
                    float delta_h = (oh_tmp - oh)/2;
                    ptop = ptop - delta_h;
                    pbot = pbot - delta_h;
                    //printf(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot = %f \n", result_ar, oh_tmp, delta_h, ptop, pbot);
                }
                else  // swidth - should be increased
                {
                    float ow_tmp = oh * net_ar;
                    float delta_w = (ow_tmp - ow)/2;
                    pleft = pleft - delta_w;
                    pright = pright - delta_w;
                    //printf(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f, pright = %f \n", result_ar, ow_tmp, delta_w, pleft, pright);
                }

                //printf("\n pleft = %d, pright = %d, ptop = %d, pbot = %d, ow = %d, oh = %d \n", pleft, pright, ptop, pbot, ow, oh);
            }

            // move each 2nd image to the corner - so that most of it was visible
            if (use_mixup == 3 && random_gen() % 2 == 0) {
                if (flip) {
                    if (i_mixup == 0) pleft += pright, pright = 0, pbot += ptop, ptop = 0;
                    if (i_mixup == 1) pright += pleft, pleft = 0, pbot += ptop, ptop = 0;
                    if (i_mixup == 2) pleft += pright, pright = 0, ptop += pbot, pbot = 0;
                    if (i_mixup == 3) pright += pleft, pleft = 0, ptop += pbot, pbot = 0;
                }
                else {
                    if (i_mixup == 0) pright += pleft, pleft = 0, pbot += ptop, ptop = 0;
                    if (i_mixup == 1) pleft += pright, pright = 0, pbot += ptop, ptop = 0;
                    if (i_mixup == 2) pright += pleft, pleft = 0, ptop += pbot, pbot = 0;
                    if (i_mixup == 3) pleft += pright, pright = 0, ptop += pbot, pbot = 0;
                }
            }

            int swidth = ow - pleft - pright;
            int sheight = oh - ptop - pbot;

            float sx = (float)swidth / ow;
            float sy = (float)sheight / oh;

            float dx = ((float)pleft / ow) / sx;
            float dy = ((float)ptop / oh) / sy;


            int min_w_h = fill_truth_detection(filename, boxes, truth_size, truth, classes, flip, dx, dy, 1. / sx, 1. / sy, w, h);
            //for (int z = 0; z < boxes; ++z) if(truth[z*truth_size] > 0) printf(" track_id = %f \n", truth[z*truth_size + 5]);
            //printf(" truth_size = %d \n", truth_size);

            if ((min_w_h / 8) < blur && blur > 1) blur = min_w_h / 8;   // disable blur if one of the objects is too small

            image ai = image_data_augmentation(src, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp,
                gaussian_noise, blur, boxes, truth_size, truth);

            if (use_mixup == 0) {
                d.X.vals[i] = ai.data;
                memcpy(d.y.vals[i], truth, truth_size * boxes * sizeof(float));
            }
            else if (use_mixup == 1) {
                if (i_mixup == 0) {
                    d.X.vals[i] = ai.data;
                    memcpy(d.y.vals[i], truth, truth_size * boxes * sizeof(float));
                }
                else if (i_mixup == 1) {
                    image old_img = make_empty_image(w, h, c);
                    old_img.data = d.X.vals[i];
                    //show_image(ai, "new");
                    //show_image(old_img, "old");
                    //wait_until_press_key_cv();
                    blend_images_cv(ai, 0.5, old_img, 0.5);
                    blend_truth(d.y.vals[i], boxes, truth_size, truth);
                    free_image(old_img);
                    d.X.vals[i] = ai.data;
                }
            }
            else if (use_mixup == 3) {
                if (i_mixup == 0) {
                    image tmp_img = make_image(w, h, c);
                    d.X.vals[i] = tmp_img.data;
                }

                if (flip) {
                    int tmp = pleft;
                    pleft = pright;
                    pright = tmp;
                }

                const int left_shift = min_val_cmp(cut_x[i], max_val_cmp(0, (-pleft*w / ow)));
                const int top_shift = min_val_cmp(cut_y[i], max_val_cmp(0, (-ptop*h / oh)));

                const int right_shift = min_val_cmp((w - cut_x[i]), max_val_cmp(0, (-pright*w / ow)));
                const int bot_shift = min_val_cmp(h - cut_y[i], max_val_cmp(0, (-pbot*h / oh)));


                int k, x, y;
                for (k = 0; k < c; ++k) {
                    for (y = 0; y < h; ++y) {
                        int j = y*w + k*w*h;
                        if (i_mixup == 0 && y < cut_y[i]) {
                            int j_src = (w - cut_x[i] - right_shift) + (y + h - cut_y[i] - bot_shift)*w + k*w*h;
                            memcpy(&d.X.vals[i][j + 0], &ai.data[j_src], cut_x[i] * sizeof(float));
                        }
                        if (i_mixup == 1 && y < cut_y[i]) {
                            int j_src = left_shift + (y + h - cut_y[i] - bot_shift)*w + k*w*h;
                            memcpy(&d.X.vals[i][j + cut_x[i]], &ai.data[j_src], (w-cut_x[i]) * sizeof(float));
                        }
                        if (i_mixup == 2 && y >= cut_y[i]) {
                            int j_src = (w - cut_x[i] - right_shift) + (top_shift + y - cut_y[i])*w + k*w*h;
                            memcpy(&d.X.vals[i][j + 0], &ai.data[j_src], cut_x[i] * sizeof(float));
                        }
                        if (i_mixup == 3 && y >= cut_y[i]) {
                            int j_src = left_shift + (top_shift + y - cut_y[i])*w + k*w*h;
                            memcpy(&d.X.vals[i][j + cut_x[i]], &ai.data[j_src], (w - cut_x[i]) * sizeof(float));
                        }
                    }
                }

                blend_truth_mosaic(d.y.vals[i], boxes, truth_size, truth, w, h, cut_x[i], cut_y[i], i_mixup, left_shift, right_shift, top_shift, bot_shift, w, h, mosaic_bound);

                free_image(ai);
                ai.data = d.X.vals[i];
            }


            if (show_imgs && i_mixup == use_mixup)   // delete i_mixup
            {
                image tmp_ai = copy_image(ai);
                char buff[1000];
                //sprintf(buff, "aug_%d_%d_%s_%d", random_index, i, basecfg((char*)filename), random_gen());
                sprintf(buff, "aug_%d_%d_%d", random_index, i, random_gen());
                int t;
                for (t = 0; t < boxes; ++t) {
                    box b = float_to_box_stride(d.y.vals[i] + t*truth_size, 1);
                    if (!b.x) break;
                    int left = (b.x - b.w / 2.)*ai.w;
                    int right = (b.x + b.w / 2.)*ai.w;
                    int top = (b.y - b.h / 2.)*ai.h;
                    int bot = (b.y + b.h / 2.)*ai.h;
                    draw_box_width(tmp_ai, left, top, right, bot, 1, 150, 100, 50); // 3 channels RGB
                }

                save_image(tmp_ai, buff);
                if (show_imgs == 1) {
                    //char buff_src[1000];
                    //sprintf(buff_src, "src_%d_%d_%s_%d", random_index, i, basecfg((char*)filename), random_gen());
                    //show_image_mat(src, buff_src);
                    show_image(tmp_ai, buff);
                    wait_until_press_key_cv();
                }
                printf("\nYou use flag -show_imgs, so will be saved aug_...jpg images. Click on window and press ESC button \n");
                free_image(tmp_ai);
            }

            release_mat(&src);
            free(truth);
        }
        if (random_paths) free(random_paths);
    }


    return d;
}
#else    // OPENCV
void blend_images(image new_img, float alpha, image old_img, float beta)
{
    int data_size = new_img.w * new_img.h * new_img.c;
    int i;
    #pragma omp parallel for
    for (i = 0; i < data_size; ++i)
        new_img.data[i] = new_img.data[i] * alpha + old_img.data[i] * beta;
}

data load_data_detection(int n, char **paths, int m, int w, int h, int c, int boxes, int truth_size, int classes, int use_flip, int gaussian_noise, int use_blur, int use_mixup,
    float jitter, float resize, float hue, float saturation, float exposure, int mini_batch, int track, int augment_speed, int letter_box, int mosaic_bound, int contrastive, int contrastive_jit_flip, int contrastive_color, int show_imgs)
{
    const int random_index = random_gen();
    c = c ? c : 3;
    char **random_paths;
    char **mixup_random_paths = NULL;
    if(track) random_paths = get_sequential_paths(paths, n, m, mini_batch, augment_speed, contrastive);
    else random_paths = get_random_paths_custom(paths, n, m, contrastive);

    //assert(use_mixup < 2);
    if (use_mixup == 2) {
        printf("\n cutmix=1 - isn't supported for Detector \n");
        exit(0);
    }
    if (use_mixup == 3 || use_mixup == 4) {
        printf("\n mosaic=1 - compile Darknet with OpenCV for using mosaic=1 \n");
        exit(0);
    }
    int mixup = use_mixup ? random_gen() % 2 : 0;
    //printf("\n mixup = %d \n", mixup);
    if (mixup) {
        if (track) mixup_random_paths = get_sequential_paths(paths, n, m, mini_batch, augment_speed, contrastive);
        else mixup_random_paths = get_random_paths(paths, n, m);
    }

    int i;
    data d = { 0 };
    d.shallow = 0;

    d.X.rows = n;
    d.X.vals = (float**)xcalloc(d.X.rows, sizeof(float*));
    d.X.cols = h*w*c;

    float r1 = 0, r2 = 0, r3 = 0, r4 = 0, r_scale;
    float resize_r1 = 0, resize_r2 = 0;
    float dhue = 0, dsat = 0, dexp = 0, flip = 0;
    int augmentation_calculated = 0;

    d.y = make_matrix(n, truth_size * boxes);
    int i_mixup = 0;
    for (i_mixup = 0; i_mixup <= mixup; i_mixup++) {
        if (i_mixup) augmentation_calculated = 0;
        for (i = 0; i < n; ++i) {
            float *truth = (float*)xcalloc(truth_size * boxes, sizeof(float));
            char *filename = (i_mixup) ? mixup_random_paths[i] : random_paths[i];

            image orig = load_image(filename, 0, 0, c);

            int oh = orig.h;
            int ow = orig.w;

            int dw = (ow*jitter);
            int dh = (oh*jitter);

            float resize_down = resize, resize_up = resize;
            if (resize_down > 1.0) resize_down = 1 / resize_down;
            int min_rdw = ow*(1 - (1 / resize_down)) / 2;
            int min_rdh = oh*(1 - (1 / resize_down)) / 2;

            if (resize_up < 1.0) resize_up = 1 / resize_up;
            int max_rdw = ow*(1 - (1 / resize_up)) / 2;
            int max_rdh = oh*(1 - (1 / resize_up)) / 2;

            if (!augmentation_calculated || !track)
            {
                augmentation_calculated = 1;
                resize_r1 = random_float();
                resize_r2 = random_float();

                if (!contrastive || contrastive_jit_flip || i % 2 == 0)
                {
                    r1 = random_float();
                    r2 = random_float();
                    r3 = random_float();
                    r4 = random_float();

                    flip = use_flip ? random_gen() % 2 : 0;
                }

                r_scale = random_float();

                if (!contrastive || contrastive_color || i % 2 == 0)
                {
                    dhue = rand_uniform_strong(-hue, hue);
                    dsat = rand_scale(saturation);
                    dexp = rand_scale(exposure);
                }
            }

            int pleft = rand_precalc_random(-dw, dw, r1);
            int pright = rand_precalc_random(-dw, dw, r2);
            int ptop = rand_precalc_random(-dh, dh, r3);
            int pbot = rand_precalc_random(-dh, dh, r4);

            if (resize < 1) {
                // downsize only
                pleft += rand_precalc_random(min_rdw, 0, resize_r1);
                pright += rand_precalc_random(min_rdw, 0, resize_r2);
                ptop += rand_precalc_random(min_rdh, 0, resize_r1);
                pbot += rand_precalc_random(min_rdh, 0, resize_r2);
            }
            else {
                pleft += rand_precalc_random(min_rdw, max_rdw, resize_r1);
                pright += rand_precalc_random(min_rdw, max_rdw, resize_r2);
                ptop += rand_precalc_random(min_rdh, max_rdh, resize_r1);
                pbot += rand_precalc_random(min_rdh, max_rdh, resize_r2);
            }

            if (letter_box)
            {
                float img_ar = (float)ow / (float)oh;
                float net_ar = (float)w / (float)h;
                float result_ar = img_ar / net_ar;
                //printf(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f, result_ar = %f \n", ow, oh, w, h, img_ar, net_ar, result_ar);
                if (result_ar > 1)  // sheight - should be increased
                {
                    float oh_tmp = ow / net_ar;
                    float delta_h = (oh_tmp - oh) / 2;
                    ptop = ptop - delta_h;
                    pbot = pbot - delta_h;
                    //printf(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot = %f \n", result_ar, oh_tmp, delta_h, ptop, pbot);
                }
                else  // swidth - should be increased
                {
                    float ow_tmp = oh * net_ar;
                    float delta_w = (ow_tmp - ow) / 2;
                    pleft = pleft - delta_w;
                    pright = pright - delta_w;
                    //printf(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f, pright = %f \n", result_ar, ow_tmp, delta_w, pleft, pright);
                }
            }

            int swidth = ow - pleft - pright;
            int sheight = oh - ptop - pbot;

            float sx = (float)swidth / ow;
            float sy = (float)sheight / oh;

            image cropped = crop_image(orig, pleft, ptop, swidth, sheight);

            float dx = ((float)pleft / ow) / sx;
            float dy = ((float)ptop / oh) / sy;

            image sized = resize_image(cropped, w, h);
            if (flip) flip_image(sized);
            distort_image(sized, dhue, dsat, dexp);
            //random_distort_image(sized, hue, saturation, exposure);

            fill_truth_detection(filename, boxes, truth_size, truth, classes, flip, dx, dy, 1. / sx, 1. / sy, w, h);

            if (i_mixup) {
                image old_img = sized;
                old_img.data = d.X.vals[i];
                //show_image(sized, "new");
                //show_image(old_img, "old");
                //wait_until_press_key_cv();
                blend_images(sized, 0.5, old_img, 0.5);
                blend_truth(truth, boxes, truth_size, d.y.vals[i]);
                free_image(old_img);
            }

            d.X.vals[i] = sized.data;
            memcpy(d.y.vals[i], truth, truth_size * boxes * sizeof(float));

            if (show_imgs)// && i_mixup)
            {
                char buff[1000];
                sprintf(buff, "aug_%d_%d_%s_%d", random_index, i, basecfg(filename), random_gen());

                int t;
                for (t = 0; t < boxes; ++t) {
                    box b = float_to_box_stride(d.y.vals[i] + t*truth_size, 1);
                    if (!b.x) break;
                    int left = (b.x - b.w / 2.)*sized.w;
                    int right = (b.x + b.w / 2.)*sized.w;
                    int top = (b.y - b.h / 2.)*sized.h;
                    int bot = (b.y + b.h / 2.)*sized.h;
                    draw_box_width(sized, left, top, right, bot, 1, 150, 100, 50); // 3 channels RGB
                }

                save_image(sized, buff);
                if (show_imgs == 1) {
                    show_image(sized, buff);
                    wait_until_press_key_cv();
                }
                printf("\nYou use flag -show_imgs, so will be saved aug_...jpg images. Press Enter: \n");
                //getchar();
            }

            free_image(orig);
            free_image(cropped);
            free(truth);
        }
    }
    free(random_paths);
    if (mixup_random_paths) free(mixup_random_paths);
    return d;
}
#endif    // OPENCV

void *load_thread(void *ptr)
{
    //srand(time(0));
    //printf("Loading data: %d\n", random_gen());
    load_args a = *(struct load_args*)ptr;
    if(a.exposure == 0) a.exposure = 1;
    if(a.saturation == 0) a.saturation = 1;
    if(a.aspect == 0) a.aspect = 1;

    if (a.type == OLD_CLASSIFICATION_DATA){
        *a.d = load_data_old(a.paths, a.n, a.m, a.labels, a.classes, a.w, a.h);
    } else if (a.type == CLASSIFICATION_DATA){
        *a.d = load_data_augment(a.paths, a.n, a.m, a.labels, a.classes, a.hierarchy, a.flip, a.min, a.max, a.w, a.h, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.mixup, a.blur, a.show_imgs, a.label_smooth_eps, a.dontuse_opencv, a.contrastive);
    } else if (a.type == SUPER_DATA){
        *a.d = load_data_super(a.paths, a.n, a.m, a.w, a.h, a.scale);
    } else if (a.type == WRITING_DATA){
        *a.d = load_data_writing(a.paths, a.n, a.m, a.w, a.h, a.out_w, a.out_h);
    } else if (a.type == REGION_DATA){
        *a.d = load_data_region(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure);
    } else if (a.type == DETECTION_DATA){
        *a.d = load_data_detection(a.n, a.paths, a.m, a.w, a.h, a.c, a.num_boxes, a.truth_size, a.classes, a.flip, a.gaussian_noise, a.blur, a.mixup, a.jitter, a.resize,
            a.hue, a.saturation, a.exposure, a.mini_batch, a.track, a.augment_speed, a.letter_box, a.mosaic_bound, a.contrastive, a.contrastive_jit_flip, a.contrastive_color, a.show_imgs);
    } else if (a.type == SWAG_DATA){
        *a.d = load_data_swag(a.paths, a.n, a.classes, a.jitter);
    } else if (a.type == COMPARE_DATA){
        *a.d = load_data_compare(a.n, a.paths, a.m, a.classes, a.w, a.h);
    } else if (a.type == IMAGE_DATA){
        *(a.im) = load_image(a.path, 0, 0, a.c);
        *(a.resized) = resize_image(*(a.im), a.w, a.h);
    }else if (a.type == LETTERBOX_DATA) {
        *(a.im) = load_image(a.path, 0, 0, a.c);
        *(a.resized) = letterbox_image(*(a.im), a.w, a.h);
    } else if (a.type == TAG_DATA){
        *a.d = load_data_tag(a.paths, a.n, a.m, a.classes, a.flip, a.min, a.max, a.w, a.h, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    }
    free(ptr);
    return 0;
}

pthread_t load_data_in_thread(load_args args)
{
    pthread_t thread;
    struct load_args* ptr = (load_args*)xcalloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_thread, ptr)) error("Thread creation failed");
    return thread;
}

static const int thread_wait_ms = 5;
static volatile int flag_exit;
static volatile int * run_load_data = NULL;
static load_args * args_swap = NULL;
static pthread_t* threads = NULL;

pthread_mutex_t mtx_load_data = PTHREAD_MUTEX_INITIALIZER;

void *run_thread_loop(void *ptr)
{
    const int i = *(int *)ptr;

    while (!custom_atomic_load_int(&flag_exit)) {
        while (!custom_atomic_load_int(&run_load_data[i])) {
            if (custom_atomic_load_int(&flag_exit)) {
                free(ptr);
                return 0;
            }
            this_thread_sleep_for(thread_wait_ms);
        }

        pthread_mutex_lock(&mtx_load_data);
        load_args *args_local = (load_args *)xcalloc(1, sizeof(load_args));
        *args_local = args_swap[i];
        pthread_mutex_unlock(&mtx_load_data);

        load_thread(args_local);

        custom_atomic_store_int(&run_load_data[i], 0);
    }
    free(ptr);
    return 0;
}

void *load_threads(void *ptr)
{
    //srand(time(0));
    int i;
    load_args args = *(load_args *)ptr;
    if (args.threads == 0) args.threads = 1;
    data *out = args.d;
    int total = args.n;
    free(ptr);
    data* buffers = (data*)xcalloc(args.threads, sizeof(data));
    if (!threads) {
        threads = (pthread_t*)xcalloc(args.threads, sizeof(pthread_t));
        run_load_data = (volatile int *)xcalloc(args.threads, sizeof(int));
        args_swap = (load_args *)xcalloc(args.threads, sizeof(load_args));
        fprintf(stderr, " Create %d permanent cpu-threads \n", args.threads);

        for (i = 0; i < args.threads; ++i) {
            int* ptr = (int*)xcalloc(1, sizeof(int));
            *ptr = i;
            if (pthread_create(&threads[i], 0, run_thread_loop, ptr)) error("Thread creation failed");
        }
    }

    for (i = 0; i < args.threads; ++i) {
        args.d = buffers + i;
        args.n = (i + 1) * total / args.threads - i * total / args.threads;

        pthread_mutex_lock(&mtx_load_data);
        args_swap[i] = args;
        pthread_mutex_unlock(&mtx_load_data);

        custom_atomic_store_int(&run_load_data[i], 1);  // run thread
    }
    for (i = 0; i < args.threads; ++i) {
        while (custom_atomic_load_int(&run_load_data[i])) this_thread_sleep_for(thread_wait_ms); //   join
    }

    /*
    pthread_t* threads = (pthread_t*)xcalloc(args.threads, sizeof(pthread_t));
    for(i = 0; i < args.threads; ++i){
        args.d = buffers + i;
        args.n = (i+1) * total/args.threads - i * total/args.threads;
        threads[i] = load_data_in_thread(args);
    }
    for(i = 0; i < args.threads; ++i){
        pthread_join(threads[i], 0);
    }
    */

    *out = concat_datas(buffers, args.threads);
    out->shallow = 0;
    for(i = 0; i < args.threads; ++i){
        buffers[i].shallow = 1;
        free_data(buffers[i]);
    }
    free(buffers);
    //free(threads);
    return 0;
}

void free_load_threads(void *ptr)
{
    load_args args = *(load_args *)ptr;
    if (args.threads == 0) args.threads = 1;
    int i;
    if (threads) {
        custom_atomic_store_int(&flag_exit, 1);
        for (i = 0; i < args.threads; ++i) {
            pthread_join(threads[i], 0);
        }
        free((void*)run_load_data);
        free(args_swap);
        free(threads);
        threads = NULL;
        custom_atomic_store_int(&flag_exit, 0);
    }
}

pthread_t load_data(load_args args)
{
    pthread_t thread;
    struct load_args* ptr = (load_args*)xcalloc(1, sizeof(struct load_args));
    *ptr = args;
    if(pthread_create(&thread, 0, load_threads, ptr)) error("Thread creation failed");
    return thread;
}

data load_data_writing(char **paths, int n, int m, int w, int h, int out_w, int out_h)
{
    if(m) paths = get_random_paths(paths, n, m);
    char **replace_paths = find_replace_paths(paths, n, ".png", "-label.png");
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = load_image_paths_gray(replace_paths, n, out_w, out_h);
    if(m) free(paths);
    int i;
    for(i = 0; i < n; ++i) free(replace_paths[i]);
    free(replace_paths);
    return d;
}

data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_paths(paths, n, w, h);
    d.y = load_labels_paths(paths, n, labels, k, 0, 0, 0);
    if(m) free(paths);
    return d;
}

/*
   data load_data_study(char **paths, int n, int m, char **labels, int k, int min, int max, int size, float angle, float aspect, float hue, float saturation, float exposure)
   {
   data d = {0};
   d.indexes = calloc(n, sizeof(int));
   if(m) paths = get_random_paths_indexes(paths, n, m, d.indexes);
   d.shallow = 0;
   d.X = load_image_augment_paths(paths, n, flip, min, max, size, angle, aspect, hue, saturation, exposure);
   d.y = load_labels_paths(paths, n, labels, k);
   if(m) free(paths);
   return d;
   }
 */

data load_data_super(char **paths, int n, int m, int w, int h, int scale)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;

    int i;
    d.X.rows = n;
    d.X.vals = (float**)xcalloc(n, sizeof(float*));
    d.X.cols = w*h*3;

    d.y.rows = n;
    d.y.vals = (float**)xcalloc(n, sizeof(float*));
    d.y.cols = w*scale * h*scale * 3;

    for(i = 0; i < n; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image crop = random_crop_image(im, w*scale, h*scale);
        int flip = random_gen()%2;
        if (flip) flip_image(crop);
        image resize = resize_image(crop, w, h);
        d.X.vals[i] = resize.data;
        d.y.vals[i] = crop.data;
        free_image(im);
    }

    if(m) free(paths);
    return d;
}

data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int use_flip, int min, int max, int w, int h, float angle,
    float aspect, float hue, float saturation, float exposure, int use_mixup, int use_blur, int show_imgs, float label_smooth_eps, int dontuse_opencv, int contrastive)
{
    char **paths_stored = paths;
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.shallow = 0;
    d.X = load_image_augment_paths(paths, n, use_flip, min, max, w, h, angle, aspect, hue, saturation, exposure, dontuse_opencv, contrastive);
    d.y = load_labels_paths(paths, n, labels, k, hierarchy, label_smooth_eps, contrastive);

    if (use_mixup && rand_int(0, 1)) {
        char **paths_mix = get_random_paths(paths_stored, n, m);
        data d2 = { 0 };
        d2.shallow = 0;
        d2.X = load_image_augment_paths(paths_mix, n, use_flip, min, max, w, h, angle, aspect, hue, saturation, exposure, dontuse_opencv, contrastive);
        d2.y = load_labels_paths(paths_mix, n, labels, k, hierarchy, label_smooth_eps, contrastive);
        free(paths_mix);

        data d3 = { 0 };
        d3.shallow = 0;
        data d4 = { 0 };
        d4.shallow = 0;
        if (use_mixup >= 3) {
            char **paths_mix3 = get_random_paths(paths_stored, n, m);
            d3.X = load_image_augment_paths(paths_mix3, n, use_flip, min, max, w, h, angle, aspect, hue, saturation, exposure, dontuse_opencv, contrastive);
            d3.y = load_labels_paths(paths_mix3, n, labels, k, hierarchy, label_smooth_eps, contrastive);
            free(paths_mix3);

            char **paths_mix4 = get_random_paths(paths_stored, n, m);
            d4.X = load_image_augment_paths(paths_mix4, n, use_flip, min, max, w, h, angle, aspect, hue, saturation, exposure, dontuse_opencv, contrastive);
            d4.y = load_labels_paths(paths_mix4, n, labels, k, hierarchy, label_smooth_eps, contrastive);
            free(paths_mix4);
        }


        // mix
        int i, j;
        for (i = 0; i < d2.X.rows; ++i) {

            int mixup = use_mixup;
            if (use_mixup == 4) mixup = rand_int(2, 3); // alternate CutMix and Mosaic

            // MixUp -----------------------------------
            if (mixup == 1) {
                // mix images
                for (j = 0; j < d2.X.cols; ++j) {
                    d.X.vals[i][j] = (d.X.vals[i][j] + d2.X.vals[i][j]) / 2.0f;
                }

                // mix labels
                for (j = 0; j < d2.y.cols; ++j) {
                    d.y.vals[i][j] = (d.y.vals[i][j] + d2.y.vals[i][j]) / 2.0f;
                }
            }
            // CutMix -----------------------------------
            else if (mixup == 2) {
                const float min = 0.3;  // 0.3*0.3 = 9%
                const float max = 0.8;  // 0.8*0.8 = 64%
                const int cut_w = rand_int(w*min, w*max);
                const int cut_h = rand_int(h*min, h*max);
                const int cut_x = rand_int(0, w - cut_w - 1);
                const int cut_y = rand_int(0, h - cut_h - 1);
                const int left = cut_x;
                const int right = cut_x + cut_w;
                const int top = cut_y;
                const int bot = cut_y + cut_h;

                assert(cut_x >= 0 && cut_x <= w);
                assert(cut_y >= 0 && cut_y <= h);
                assert(cut_w >= 0 && cut_w <= w);
                assert(cut_h >= 0 && cut_h <= h);

                assert(right >= 0 && right <= w);
                assert(bot >= 0 && bot <= h);

                assert(top <= bot);
                assert(left <= right);

                const float alpha = (float)(cut_w*cut_h) / (float)(w*h);
                const float beta = 1 - alpha;

                int c, x, y;
                for (c = 0; c < 3; ++c) {
                    for (y = top; y < bot; ++y) {
                        for (x = left; x < right; ++x) {
                            int j = x + y*w + c*w*h;
                            d.X.vals[i][j] = d2.X.vals[i][j];
                        }
                    }
                }

                //printf("\n alpha = %f, beta = %f \n", alpha, beta);
                // mix labels
                for (j = 0; j < d.y.cols; ++j) {
                    d.y.vals[i][j] = d.y.vals[i][j] * beta + d2.y.vals[i][j] * alpha;
                }
            }
            // Mosaic -----------------------------------
            else if (mixup == 3)
            {
                const float min_offset = 0.2; // 20%
                const int cut_x = rand_int(w*min_offset, w*(1 - min_offset));
                const int cut_y = rand_int(h*min_offset, h*(1 - min_offset));

                float s1 = (float)(cut_x * cut_y) / (w*h);
                float s2 = (float)((w - cut_x) * cut_y) / (w*h);
                float s3 = (float)(cut_x * (h - cut_y)) / (w*h);
                float s4 = (float)((w - cut_x) * (h - cut_y)) / (w*h);

                int c, x, y;
                for (c = 0; c < 3; ++c) {
                    for (y = 0; y < h; ++y) {
                        for (x = 0; x < w; ++x) {
                            int j = x + y*w + c*w*h;
                            if (x < cut_x && y < cut_y) d.X.vals[i][j] = d.X.vals[i][j];
                            if (x >= cut_x && y < cut_y) d.X.vals[i][j] = d2.X.vals[i][j];
                            if (x < cut_x && y >= cut_y) d.X.vals[i][j] = d3.X.vals[i][j];
                            if (x >= cut_x && y >= cut_y) d.X.vals[i][j] = d4.X.vals[i][j];
                        }
                    }
                }

                for (j = 0; j < d.y.cols; ++j) {
                    const float max_s = 1;// max_val_cmp(s1, max_val_cmp(s2, max_val_cmp(s3, s4)));

                    d.y.vals[i][j] = d.y.vals[i][j] * s1 / max_s + d2.y.vals[i][j] * s2 / max_s + d3.y.vals[i][j] * s3 / max_s + d4.y.vals[i][j] * s4 / max_s;
                }
            }
        }

        free_data(d2);

        if (use_mixup >= 3) {
            free_data(d3);
            free_data(d4);
        }
    }

#ifdef OPENCV
    if (use_blur) {
        int i;
        for (i = 0; i < d.X.rows; ++i) {
            if (random_gen() % 4 == 0) {
                image im = make_empty_image(w, h, 3);
                im.data = d.X.vals[i];
                int ksize = use_blur;
                if (use_blur == 1) ksize = 15;
                image blurred = blur_image(im, ksize);
                free_image(im);
                d.X.vals[i] = blurred.data;
                //if (i == 0) {
                //    show_image(im, "Not blurred");
                //    show_image(blurred, "blurred");
                //    wait_until_press_key_cv();
                //}
            }
        }
    }
#endif  // OPENCV

    if (show_imgs) {
        int i, j;
        for (i = 0; i < d.X.rows; ++i) {
            image im = make_empty_image(w, h, 3);
            im.data = d.X.vals[i];
            char buff[1000];
            sprintf(buff, "aug_%d_%s_%d", i, basecfg((char*)paths[i]), random_gen());
            save_image(im, buff);

            char buff_string[1000];
            sprintf(buff_string, "\n Classes: ");
            for (j = 0; j < d.y.cols; ++j) {
                if (d.y.vals[i][j] > 0) {
                    char buff_tmp[100];
                    sprintf(buff_tmp, " %d (%f), ", j, d.y.vals[i][j]);
                    strcat(buff_string, buff_tmp);
                }
            }
            printf("%s \n", buff_string);

            if (show_imgs == 1) {
                show_image(im, buff);
                wait_until_press_key_cv();
            }
        }
        printf("\nYou use flag -show_imgs, so will be saved aug_...jpg images. Click on window and press ESC button \n");
    }

    if (m) free(paths);

    return d;
}

data load_data_tag(char **paths, int n, int m, int k, int use_flip, int min, int max, int w, int h, float angle, float aspect, float hue, float saturation, float exposure)
{
    if(m) paths = get_random_paths(paths, n, m);
    data d = {0};
    d.w = w;
    d.h = h;
    d.shallow = 0;
    d.X = load_image_augment_paths(paths, n, use_flip, min, max, w, h, angle, aspect, hue, saturation, exposure, 0, 0);
    d.y = load_tags_paths(paths, n, k);
    if(m) free(paths);
    return d;
}

matrix concat_matrix(matrix m1, matrix m2)
{
    int i, count = 0;
    matrix m;
    m.cols = m1.cols;
    m.rows = m1.rows+m2.rows;
    m.vals = (float**)xcalloc(m1.rows + m2.rows, sizeof(float*));
    for(i = 0; i < m1.rows; ++i){
        m.vals[count++] = m1.vals[i];
    }
    for(i = 0; i < m2.rows; ++i){
        m.vals[count++] = m2.vals[i];
    }
    return m;
}

data concat_data(data d1, data d2)
{
    data d = {0};
    d.shallow = 1;
    d.X = concat_matrix(d1.X, d2.X);
    d.y = concat_matrix(d1.y, d2.y);
    return d;
}

data concat_datas(data *d, int n)
{
    int i;
    data out = {0};
    for(i = 0; i < n; ++i){
        data newdata = concat_data(d[i], out);
        free_data(out);
        out = newdata;
    }
    return out;
}

data load_categorical_data_csv(char *filename, int target, int k)
{
    data d = {0};
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
    data d = {0};
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
        int class_id = bytes[0];
        y.vals[i][class_id] = 1;
        for(j = 0; j < X.cols; ++j){
            X.vals[i][j] = (double)bytes[j+1];
        }
    }
    //translate_data_rows(d, -128);
    scale_data_rows(d, 1./255);
    //normalize_data_rows(d);
    fclose(fp);
    return d;
}

void get_random_batch(data d, int n, float *X, float *y)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = random_gen()%d.X.rows;
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

void smooth_data(data d)
{
    int i, j;
    float scale = 1. / d.y.cols;
    float eps = .1;
    for(i = 0; i < d.y.rows; ++i){
        for(j = 0; j < d.y.cols; ++j){
            d.y.vals[i][j] = eps * scale + (1-eps) * d.y.vals[i][j];
        }
    }
}

data load_all_cifar10()
{
    data d = {0};
    d.shallow = 0;
    int i,j,b;
    matrix X = make_matrix(50000, 3072);
    matrix y = make_matrix(50000, 10);
    d.X = X;
    d.y = y;


    for(b = 0; b < 5; ++b){
        char buff[256];
        sprintf(buff, "data/cifar/cifar-10-batches-bin/data_batch_%d.bin", b+1);
        FILE *fp = fopen(buff, "rb");
        if(!fp) file_error(buff);
        for(i = 0; i < 10000; ++i){
            unsigned char bytes[3073];
            fread(bytes, 1, 3073, fp);
            int class_id = bytes[0];
            y.vals[i+b*10000][class_id] = 1;
            for(j = 0; j < X.cols; ++j){
                X.vals[i+b*10000][j] = (double)bytes[j+1];
            }
        }
        fclose(fp);
    }
    //normalize_data_rows(d);
    //translate_data_rows(d, -128);
    scale_data_rows(d, 1./255);
    smooth_data(d);
    return d;
}

data load_go(char *filename)
{
    FILE *fp = fopen(filename, "rb");
    matrix X = make_matrix(3363059, 361);
    matrix y = make_matrix(3363059, 361);
    int row, col;

    if(!fp) file_error(filename);
    char *label;
    int count = 0;
    while((label = fgetl(fp))){
        int i;
        if(count == X.rows){
            X = resize_matrix(X, count*2);
            y = resize_matrix(y, count*2);
        }
        sscanf(label, "%d %d", &row, &col);
        char *board = fgetl(fp);

        int index = row*19 + col;
        y.vals[count][index] = 1;

        for(i = 0; i < 19*19; ++i){
            float val = 0;
            if(board[i] == '1') val = 1;
            else if(board[i] == '2') val = -1;
            X.vals[count][i] = val;
        }
        ++count;
        free(label);
        free(board);
    }
    X = resize_matrix(X, count);
    y = resize_matrix(y, count);

    data d = {0};
    d.shallow = 0;
    d.X = X;
    d.y = y;


    fclose(fp);

    return d;
}


void randomize_data(data d)
{
    int i;
    for(i = d.X.rows-1; i > 0; --i){
        int index = random_gen()%i;
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

data get_data_part(data d, int part, int total)
{
    data p = {0};
    p.shallow = 1;
    p.X.rows = d.X.rows * (part + 1) / total - d.X.rows * part / total;
    p.y.rows = d.y.rows * (part + 1) / total - d.y.rows * part / total;
    p.X.cols = d.X.cols;
    p.y.cols = d.y.cols;
    p.X.vals = d.X.vals + d.X.rows * part / total;
    p.y.vals = d.y.vals + d.y.rows * part / total;
    return p;
}

data get_random_data(data d, int num)
{
    data r = {0};
    r.shallow = 1;

    r.X.rows = num;
    r.y.rows = num;

    r.X.cols = d.X.cols;
    r.y.cols = d.y.cols;

    r.X.vals = (float**)xcalloc(num, sizeof(float*));
    r.y.vals = (float**)xcalloc(num, sizeof(float*));

    int i;
    for(i = 0; i < num; ++i){
        int index = random_gen()%d.X.rows;
        r.X.vals[i] = d.X.vals[index];
        r.y.vals[i] = d.y.vals[index];
    }
    return r;
}

data *split_data(data d, int part, int total)
{
    data* split = (data*)xcalloc(2, sizeof(data));
    int i;
    int start = part*d.X.rows/total;
    int end = (part+1)*d.X.rows/total;
    data train ={0};
    data test ={0};
    train.shallow = test.shallow = 1;

    test.X.rows = test.y.rows = end-start;
    train.X.rows = train.y.rows = d.X.rows - (end-start);
    train.X.cols = test.X.cols = d.X.cols;
    train.y.cols = test.y.cols = d.y.cols;

    train.X.vals = (float**)xcalloc(train.X.rows, sizeof(float*));
    test.X.vals = (float**)xcalloc(test.X.rows, sizeof(float*));
    train.y.vals = (float**)xcalloc(train.y.rows, sizeof(float*));
    test.y.vals = (float**)xcalloc(test.y.rows, sizeof(float*));

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
