#include <stdio.h>

#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"

void train_compare(char *cfgfile, char *weightfile)
{
    srand(time(0));
    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    char* backup_directory = "backup/";
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = 1024;
    list *plist = get_paths("data/compare.train.list");
    char **paths = (char **)list_to_array(plist);
    int N = plist->size;
    printf("%d\n", N);
    clock_t time;
    pthread_t load_thread;
    data train;
    data buffer;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.classes = 20;
    args.n = imgs;
    args.m = N;
    args.d = &buffer;
    args.type = COMPARE_DATA;

    load_thread = load_data_in_thread(args);
    int epoch = *net.seen/N;
    int i = 0;
    while(1){
        ++i;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;

        load_thread = load_data_in_thread(args);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%.3f: %f, %f avg, %lf seconds, %ld images\n", (float)*net.seen/N, loss, avg_loss, sec(clock()-time), *net.seen);
        free_data(train);
        if(i%100 == 0){
            char buff[256];
            sprintf(buff, "%s/%s_%d_minor_%d.weights",backup_directory,base, epoch, i);
            save_weights(net, buff);
        }
        if(*net.seen/N > epoch){
            epoch = *net.seen/N;
            i = 0;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
            if(epoch%22 == 0) net.learning_rate *= .1;
        }
    }
    pthread_join(load_thread, 0);
    free_data(buffer);
    free_network(net);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void validate_compare(char *filename, char *weightfile)
{
    int i = 0;
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *plist = get_paths("data/compare.val.list");
    //list *plist = get_paths("data/compare.val.old");
    char **paths = (char **)list_to_array(plist);
    int N = plist->size/2;
    free_list(plist);

    clock_t time;
    int correct = 0;
    int total = 0;
    int splits = 10;
    int num = (i+1)*N/splits - i*N/splits;

    data val, buffer;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.classes = 20;
    args.n = num;
    args.m = 0;
    args.d = &buffer;
    args.type = COMPARE_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    for(i = 1; i <= splits; ++i){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        num = (i+1)*N/splits - i*N/splits;
        char **part = paths+(i*N/splits);
        if(i != splits){
            args.paths = part;
            load_thread = load_data_in_thread(args);
        }
        printf("Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
        matrix pred = network_predict_data(net, val);
        int j,k;
        for(j = 0; j < val.y.rows; ++j){
            for(k = 0; k < 20; ++k){
                if(val.y.vals[j][k*2] != val.y.vals[j][k*2+1]){
                    ++total;
                    if((val.y.vals[j][k*2] < val.y.vals[j][k*2+1]) == (pred.vals[j][k*2] < pred.vals[j][k*2+1])){
                        ++correct;
                    }
                }
            }
        }
        free_matrix(pred);
        printf("%d: Acc: %f, %lf seconds, %d images\n", i, (float)correct/total, sec(clock()-time), val.X.rows);
        free_data(val);
    }
}

typedef struct {
    network net;
    char *filename;
    int class_id;
    int classes;
    float elo;
    float *elos;
} sortable_bbox;

int total_compares = 0;
int current_class_id = 0;

int elo_comparator(const void*a, const void *b)
{
    sortable_bbox box1 = *(sortable_bbox*)a;
    sortable_bbox box2 = *(sortable_bbox*)b;
    if(box1.elos[current_class_id] == box2.elos[current_class_id]) return 0;
    if(box1.elos[current_class_id] >  box2.elos[current_class_id]) return -1;
    return 1;
}

int bbox_comparator(const void *a, const void *b)
{
    ++total_compares;
    sortable_bbox box1 = *(sortable_bbox*)a;
    sortable_bbox box2 = *(sortable_bbox*)b;
    network net = box1.net;
    int class_id   = box1.class_id;

    image im1 = load_image_color(box1.filename, net.w, net.h);
    image im2 = load_image_color(box2.filename, net.w, net.h);
    float* X = (float*)calloc(net.w * net.h * net.c, sizeof(float));
    memcpy(X,                   im1.data, im1.w*im1.h*im1.c*sizeof(float));
    memcpy(X+im1.w*im1.h*im1.c, im2.data, im2.w*im2.h*im2.c*sizeof(float));
    float *predictions = network_predict(net, X);

    free_image(im1);
    free_image(im2);
    free(X);
    if (predictions[class_id*2] > predictions[class_id*2+1]){
        return 1;
    }
    return -1;
}

void bbox_update(sortable_bbox *a, sortable_bbox *b, int class_id, int result)
{
    int k = 32;
    float EA = 1./(1+pow(10, (b->elos[class_id] - a->elos[class_id])/400.));
    float EB = 1./(1+pow(10, (a->elos[class_id] - b->elos[class_id])/400.));
    float SA = result ? 1 : 0;
    float SB = result ? 0 : 1;
    a->elos[class_id] += k*(SA - EA);
    b->elos[class_id] += k*(SB - EB);
}

void bbox_fight(network net, sortable_bbox *a, sortable_bbox *b, int classes, int class_id)
{
    image im1 = load_image_color(a->filename, net.w, net.h);
    image im2 = load_image_color(b->filename, net.w, net.h);
    float* X = (float*)calloc(net.w * net.h * net.c, sizeof(float));
    memcpy(X,                   im1.data, im1.w*im1.h*im1.c*sizeof(float));
    memcpy(X+im1.w*im1.h*im1.c, im2.data, im2.w*im2.h*im2.c*sizeof(float));
    float *predictions = network_predict(net, X);
    ++total_compares;

    int i;
    for(i = 0; i < classes; ++i){
        if(class_id < 0 || class_id == i){
            int result = predictions[i*2] > predictions[i*2+1];
            bbox_update(a, b, i, result);
        }
    }

    free_image(im1);
    free_image(im2);
    free(X);
}

void SortMaster3000(char *filename, char *weightfile)
{
    int i = 0;
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));
    set_batch_network(&net, 1);

    list *plist = get_paths("data/compare.sort.list");
    //list *plist = get_paths("data/compare.val.old");
    char **paths = (char **)list_to_array(plist);
    int N = plist->size;
    free_list(plist);
    sortable_bbox* boxes = (sortable_bbox*)calloc(N, sizeof(sortable_bbox));
    printf("Sorting %d boxes...\n", N);
    for(i = 0; i < N; ++i){
        boxes[i].filename = paths[i];
        boxes[i].net = net;
        boxes[i].class_id = 7;
        boxes[i].elo = 1500;
    }
    clock_t time=clock();
    qsort(boxes, N, sizeof(sortable_bbox), bbox_comparator);
    for(i = 0; i < N; ++i){
        printf("%s\n", boxes[i].filename);
    }
    printf("Sorted in %d compares, %f secs\n", total_compares, sec(clock()-time));
}

void BattleRoyaleWithCheese(char *filename, char *weightfile)
{
    int classes = 20;
    int i,j;
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));
    set_batch_network(&net, 1);

    list *plist = get_paths("data/compare.sort.list");
    //list *plist = get_paths("data/compare.small.list");
    //list *plist = get_paths("data/compare.cat.list");
    //list *plist = get_paths("data/compare.val.old");
    char **paths = (char **)list_to_array(plist);
    int N = plist->size;
    int total = N;
    free_list(plist);
    sortable_bbox* boxes = (sortable_bbox*)calloc(N, sizeof(sortable_bbox));
    printf("Battling %d boxes...\n", N);
    for(i = 0; i < N; ++i){
        boxes[i].filename = paths[i];
        boxes[i].net = net;
        boxes[i].classes = classes;
        boxes[i].elos = (float*)calloc(classes, sizeof(float));
        for(j = 0; j < classes; ++j){
            boxes[i].elos[j] = 1500;
        }
    }
    int round;
    clock_t time=clock();
    for(round = 1; round <= 4; ++round){
        clock_t round_time=clock();
        printf("Round: %d\n", round);
        shuffle(boxes, N, sizeof(sortable_bbox));
        for(i = 0; i < N/2; ++i){
            bbox_fight(net, boxes+i*2, boxes+i*2+1, classes, -1);
        }
        printf("Round: %f secs, %d remaining\n", sec(clock()-round_time), N);
    }

    int class_id;

    for (class_id = 0; class_id < classes; ++class_id){

        N = total;
        current_class_id = class_id;
        qsort(boxes, N, sizeof(sortable_bbox), elo_comparator);
        N /= 2;

        for(round = 1; round <= 100; ++round){
            clock_t round_time=clock();
            printf("Round: %d\n", round);

            sorta_shuffle(boxes, N, sizeof(sortable_bbox), 10);
            for(i = 0; i < N/2; ++i){
                bbox_fight(net, boxes+i*2, boxes+i*2+1, classes, class_id);
            }
            qsort(boxes, N, sizeof(sortable_bbox), elo_comparator);
            if(round <= 20) N = (N*9/10)/2*2;

            printf("Round: %f secs, %d remaining\n", sec(clock()-round_time), N);
        }
        char buff[256];
        sprintf(buff, "results/battle_%d.log", class_id);
        FILE *outfp = fopen(buff, "w");
        for(i = 0; i < N; ++i){
            fprintf(outfp, "%s %f\n", boxes[i].filename, boxes[i].elos[class_id]);
        }
        fclose(outfp);
    }
    printf("Tournament in %d compares, %f secs\n", total_compares, sec(clock()-time));
}

void run_compare(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    //char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "train")) train_compare(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_compare(cfg, weights);
    else if(0==strcmp(argv[2], "sort")) SortMaster3000(cfg, weights);
    else if(0==strcmp(argv[2], "battle")) BattleRoyaleWithCheese(cfg, weights);
    /*
       else if(0==strcmp(argv[2], "train")) train_coco(cfg, weights);
       else if(0==strcmp(argv[2], "extract")) extract_boxes(cfg, weights);
       else if(0==strcmp(argv[2], "valid")) validate_recall(cfg, weights);
     */
}
