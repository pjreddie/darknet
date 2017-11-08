#include "darknet.h"

#include <sys/time.h>
#include <assert.h>

void extend_data_truth(data *d, int n, float val)
{
    int i, j;
    for(i = 0; i < d->y.rows; ++i){
        d->y.vals[i] = realloc(d->y.vals[i], (d->y.cols+n)*sizeof(float));
        for(j = 0; j < n; ++j){
            d->y.vals[i][d->y.cols + j] = val;
        }
    }
    d->y.cols += n;
}

matrix network_loss_data(network *net, data test)
{
    int i,b;
    int k = 1;
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net->batch*test.X.cols, sizeof(float));
    float *y = calloc(net->batch*test.y.cols, sizeof(float));
    for(i = 0; i < test.X.rows; i += net->batch){
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
            memcpy(y+b*test.y.cols, test.y.vals[i+b], test.y.cols*sizeof(float));
        }

        network orig = *net;
        net->input = X;
        net->truth = y;
        net->train = 0;
        net->delta = 0;
        forward_network(net);
        *net = orig;

        float *delta = net->layers[net->n-1].output;
        for(b = 0; b < net->batch; ++b){
            if(i+b == test.X.rows) break;
            int t = max_index(y + b*test.y.cols, 1000);
            float err = sum_array(delta + b*net->outputs, net->outputs);
            pred.vals[i+b][0] = -err;
            //pred.vals[i+b][0] = 1-delta[b*net->outputs + t];
        }
    }
    free(X);
    free(y);
    return pred;   
}

void train_attention(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    int i, j;

    float avg_cls_loss = -1;
    float avg_att_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    int classes = option_find_int(options, "classes", 2);

    char **labels = get_labels(label_list);
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    double time;

    int divs=3;
    int size=2;

    load_args args = {0};
    args.w = divs*net->w/size;
    args.h = divs*net->h/size;
    args.size = divs*net->w/size;
    args.threads = 32;
    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*args.w;
    args.max = net->max_ratio*args.w;
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    args.type = CLASSIFICATION_DATA;

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        time = what_time_is_it_now();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);
        data resized = resize_data(train, net->w, net->h);
        extend_data_truth(&resized, divs*divs, 0);
        data *tiles = tile_data(train, divs, size);

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time = what_time_is_it_now();

        float aloss = 0;
        float closs = 0;
        int z;
        for (i = 0; i < divs*divs/ngpus; ++i) {
#pragma omp parallel for
            for(j = 0; j < ngpus; ++j){
                int index = i*ngpus + j;
                extend_data_truth(tiles+index, divs*divs, SECRET_NUM);
                matrix deltas = network_loss_data(nets[j], tiles[index]);
                for(z = 0; z < resized.y.rows; ++z){
                    resized.y.vals[z][train.y.cols + index] = deltas.vals[z][0];
                }
                free_matrix(deltas);
            }
        }
        int *inds = calloc(resized.y.rows, sizeof(int));
        for(z = 0; z < resized.y.rows; ++z){
            int index = max_index(resized.y.vals[z] + train.y.cols, divs*divs);
            inds[z] = index;
            for(i = 0; i < divs*divs; ++i){
                resized.y.vals[z][train.y.cols + i] = (i == index)? 1 : 0;
            }
        }
        data best = select_data(tiles, inds);
        free(inds);
        #ifdef GPU
        if (ngpus == 1) {
            closs = train_network(net, best);
        } else {
            closs = train_networks(nets, ngpus, best, 4);
        }
        #endif
        for (i = 0; i < divs*divs; ++i) {
            printf("%.2f ", resized.y.vals[0][train.y.cols + i]);
            if((i+1)%divs == 0) printf("\n");
            free_data(tiles[i]);
        }
        free_data(best);
        printf("\n");
        image im = float_to_image(64,64,3,resized.X.vals[0]);
        //show_image(im, "orig");
        //cvWaitKey(100);
        /*
           image im1 = float_to_image(64,64,3,tiles[i].X.vals[0]);
           image im2 = float_to_image(64,64,3,resized.X.vals[0]);
           show_image(im1, "tile");
           show_image(im2, "res");
         */
#ifdef GPU
        if (ngpus == 1) {
            aloss = train_network(net, resized);
        } else {
            aloss = train_networks(nets, ngpus, resized, 4);
        }
#endif
        for(i = 0; i < divs*divs; ++i){
            printf("%f ", nets[0]->output[1000 + i]);
            if ((i+1) % divs == 0) printf("\n");
        }
        printf("\n");

        free_data(resized);
        free_data(train);
        if(avg_cls_loss == -1) avg_cls_loss = closs;
        if(avg_att_loss == -1) avg_att_loss = aloss;
        avg_cls_loss = avg_cls_loss*.9 + closs*.1;
        avg_att_loss = avg_att_loss*.9 + aloss*.1;

        printf("%ld, %.3f: Att: %f, %f avg, Class: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, aloss, avg_att_loss, closs, avg_cls_loss, get_current_rate(net), what_time_is_it_now()-time, *net->seen);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
            save_weights(net, buff);
        }
        if(get_current_batch(net)%1000 == 0){
            char buff[256];
            sprintf(buff, "%s/%s.backup",backup_directory,base);
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);
    pthread_join(load_thread, 0);

    free_network(net);
    free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void validate_attention_single(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net->hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));
    int divs = 4;
    int size = 2;
    int extra = 0;
    float *avgs = calloc(classes, sizeof(float));
    int *inds = calloc(divs*divs, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, net->w*divs/size);
        image crop = crop_image(resized, (resized.w - net->w*divs/size)/2, (resized.h - net->h*divs/size)/2, net->w*divs/size, net->h*divs/size);
        image rcrop = resize_image(crop, net->w, net->h);
        //show_image(im, "orig");
        //show_image(crop, "cropped");
        //cvWaitKey(0);
        float *pred = network_predict(net, rcrop.data);
        //pred[classes + 56] = 0;
        for(j = 0; j < divs*divs; ++j){
            printf("%.2f ", pred[classes + j]);
            if((j+1)%divs == 0) printf("\n");
        }
        printf("\n");
        copy_cpu(classes, pred, 1, avgs, 1);
        top_k(pred + classes, divs*divs, divs*divs, inds);
        show_image(crop, "crop");
        for(j = 0; j < extra; ++j){
            int index = inds[j];
            int row = index / divs;
            int col = index % divs;
            int y = row * crop.h / divs - (net->h - crop.h/divs)/2;
            int x = col * crop.w / divs - (net->w - crop.w/divs)/2;
            printf("%d %d %d %d\n", row, col, y, x);
            image tile = crop_image(crop, x, y, net->w, net->h);
            float *pred = network_predict(net, tile.data);
            axpy_cpu(classes, 1., pred, 1, avgs, 1);
            show_image(tile, "tile");
            //cvWaitKey(10);
        }
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        if(rcrop.data != resized.data) free_image(rcrop);
        if(resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void validate_attention_multi(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);
    int scales[] = {224, 288, 320, 352, 384};
    int nscales = sizeof(scales)/sizeof(scales[0]);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        float *pred = calloc(classes, sizeof(float));
        image im = load_image_color(paths[i], 0, 0);
        for(j = 0; j < nscales; ++j){
            image r = resize_min(im, scales[j]);
            resize_network(net, r.w, r.h);
            float *p = network_predict(net, r.data);
            if(net->hierarchy) hierarchy_predictions(p, net->outputs, net->hierarchy, 1 , 1);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            flip_image(r);
            p = network_predict(net, r.data);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            if(r.data != im.data) free_image(r);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void predict_attention(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    if(top == 0) top = option_find_int(options, "top", 1);

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        image r = letterbox_image(im, net->w, net->h);
        //resize_network(&net, r.w, r.h);
        //printf("%d %d\n", r.w, r.h);

        float *X = r.data;
        time=clock();
        float *predictions = network_predict(net, X);
        if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
        top_k(predictions, net->outputs, top, indexes);
        fprintf(stderr, "%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            //if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net->hierarchy->parent[index] >= 0) ? names[net->hierarchy->parent[index]] : "Root");
            //else printf("%s: %f\n",names[index], predictions[index]);
            printf("%5.2f%%: %s\n", predictions[index]*100, names[index]);
        }
        if(r.data != im.data) free_image(r);
        free_image(im);
        if (filename) break;
    }
}


void run_attention(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int ngpus;
    int *gpus = read_intlist(gpu_list, &ngpus, gpu_index);


    int top = find_int_arg(argc, argv, "-t", 0);
    int clear = find_arg(argc, argv, "-clear");
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    char *layer_s = (argc > 7) ? argv[7]: 0;
    if(0==strcmp(argv[2], "predict")) predict_attention(data, cfg, weights, filename, top);
    else if(0==strcmp(argv[2], "train")) train_attention(data, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) validate_attention_single(data, cfg, weights);
    else if(0==strcmp(argv[2], "validmulti")) validate_attention_multi(data, cfg, weights);
}


