#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"
#include "assert.h"
#include "classifier.h"
#include "dark_cuda.h"
#ifdef WIN32
#include <time.h>
#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif

float validate_classifier_single(char *datacfg, char *filename, char *weightfile, network *existing_net, int topk_custom);

float *get_regression_values(char **labels, int n)
{
    float* v = (float*)xcalloc(n, sizeof(float));
    int i;
    for(i = 0; i < n; ++i){
        char *p = strchr(labels[i], ' ');
        *p = 0;
        v[i] = atof(p+1);
    }
    return v;
}

void train_classifier(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int dontuse_opencv, int dont_show, int mjpeg_port, int calc_topk, int show_imgs, char* chart_path)
{
    int i;

    float avg_loss = -1;
    float avg_contrastive_acc = 0;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network* nets = (network*)xcalloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = parse_network_cfg(cfgfile);
        if(weightfile){
            load_weights(&nets[i], weightfile);
        }
        if (clear) {
            *nets[i].seen = 0;
            *nets[i].cur_iteration = 0;
        }
        nets[i].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];

    int imgs = net.batch * net.subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk_data = option_find_int(options, "top", 5);
    char topk_buff[10];
    sprintf(topk_buff, "top%d", topk_data);
    layer l = net.layers[net.n - 1];
    if (classes != l.outputs && (l.type == SOFTMAX || l.type == COST)) {
        printf("\n Error: num of filters = %d in the last conv-layer in cfg-file doesn't match to classes = %d in data-file \n",
            l.outputs, classes);
        getchar();
    }

    char **labels = get_labels(label_list);
    if (net.unsupervised) {
        free(labels);
        labels = NULL;
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int train_images_num = plist->size;
    clock_t time;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    args.threads = 32;
    if (net.contrastive && args.threads > net.batch/2) args.threads = net.batch / 2;
    args.hierarchy = net.hierarchy;

    args.contrastive = net.contrastive;
    args.dontuse_opencv = dontuse_opencv;
    args.min = net.min_crop;
    args.max = net.max_crop;
    args.flip = net.flip;
    args.blur = net.blur;
    args.angle = net.angle;
    args.aspect = net.aspect;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    args.size = net.w > net.h ? net.w : net.h;

    args.label_smooth_eps = net.label_smooth_eps;
    args.mixup = net.mixup;
    if (dont_show && show_imgs) show_imgs = 2;
    args.show_imgs = show_imgs;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = train_images_num;
    args.labels = labels;
    args.type = CLASSIFICATION_DATA;

#ifdef OPENCV
    //args.threads = 3;
    mat_cv* img = NULL;
    float max_img_loss = net.max_chart_loss;
    int number_of_lines = 100;
    int img_size = 1000;
    char windows_name[100];
    sprintf(windows_name, "chart_%s.png", base);
    if (!dontuse_opencv) img = draw_train_chart(windows_name, max_img_loss, net.max_batches, number_of_lines, img_size, dont_show, chart_path);
#endif  //OPENCV

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int iter_save = get_current_batch(net);
    int iter_save_last = get_current_batch(net);
    int iter_topk = get_current_batch(net);
    float topk = 0;

    int count = 0;
    double start, time_remaining, avg_time = -1, alpha_time = 0.01;
    start = what_time_is_it_now();

    while(get_current_batch(net) < net.max_batches || net.max_batches == 0){
        time=clock();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();

        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if(avg_loss == -1 || isnan(avg_loss) || isinf(avg_loss)) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        i = get_current_batch(net);

        int calc_topk_for_each = iter_topk + 2 * train_images_num / (net.batch * net.subdivisions);  // calculate TOPk for each 2 Epochs
        calc_topk_for_each = fmax(calc_topk_for_each, net.burn_in);
        calc_topk_for_each = fmax(calc_topk_for_each, 100);
        if (i % 10 == 0) {
            if (calc_topk) {
                fprintf(stderr, "\n (next TOP%d calculation at %d iterations) ", topk_data, calc_topk_for_each);
                if (topk > 0) fprintf(stderr, " Last accuracy TOP%d = %2.2f %% \n", topk_data, topk * 100);
            }

            if (net.cudnn_half) {
                if (i < net.burn_in * 3) fprintf(stderr, " Tensor Cores are disabled until the first %d iterations are reached.\n", 3 * net.burn_in);
                else fprintf(stderr, " Tensor Cores are used.\n");
            }
        }

        int draw_precision = 0;
        if (calc_topk && (i >= calc_topk_for_each || i == net.max_batches)) {
            iter_topk = i;
            if (net.contrastive && l.type != SOFTMAX && l.type != COST) {
                int k;
                for (k = 0; k < net.n; ++k) if (net.layers[k].type == CONTRASTIVE) break;
                topk = *(net.layers[k].loss) / 100;
                sprintf(topk_buff, "Contr");
            }
            else {
                topk = validate_classifier_single(datacfg, cfgfile, weightfile, &net, topk_data); // calc TOP-n
                printf("\n accuracy %s = %f \n", topk_buff, topk);
            }
            draw_precision = 1;
        }

        time_remaining = ((net.max_batches - i) / ngpus) * (what_time_is_it_now() - start) / 60 / 60;
        // set initial value, even if resume training from 10000 iteration
        if (avg_time < 0) avg_time = time_remaining;
        else avg_time = alpha_time * time_remaining + (1 -  alpha_time) * avg_time;
        start = what_time_is_it_now();
        printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images, %f hours left\n", get_current_batch(net), (float)(*net.seen)/ train_images_num, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen, avg_time);
#ifdef OPENCV
        if (net.contrastive) {
            float cur_con_acc = -1;
            int k;
            for (k = 0; k < net.n; ++k)
                if (net.layers[k].type == CONTRASTIVE) cur_con_acc = *net.layers[k].loss;
            if (cur_con_acc >= 0) avg_contrastive_acc = avg_contrastive_acc*0.99 + cur_con_acc * 0.01;
            printf("  avg_contrastive_acc = %f \n", avg_contrastive_acc);
        }
        if (!dontuse_opencv) draw_train_loss(windows_name, img, img_size, avg_loss, max_img_loss, i, net.max_batches, topk, draw_precision, topk_buff, avg_contrastive_acc / 100, dont_show, mjpeg_port, avg_time);
#endif  // OPENCV

        if (i >= (iter_save + 1000)) {
            iter_save = i;
#ifdef GPU
            if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }

        if (i >= (iter_save_last + 100)) {
            iter_save_last = i;
#ifdef GPU
            if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_last.weights", backup_directory, base);
            save_weights(net, buff);
        }
        free_data(train);
    }
#ifdef GPU
    if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);

#ifdef OPENCV
    release_mat(&img);
    destroy_all_windows_cv();
#endif

    pthread_join(load_thread, 0);
    free_data(buffer);

    //free_network(net);
    for (i = 0; i < ngpus; ++i) free_network(nets[i]);
    free(nets);

    //free_ptrs((void**)labels, classes);
    if(labels) free(labels);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);

    free_list_contents_kvp(options);
    free_list(options);

}


/*
   void train_classifier(char *datacfg, char *cfgfile, char *weightfile, int clear)
   {
   srand(time(0));
   float avg_loss = -1;
   char *base = basecfg(cfgfile);
   printf("%s\n", base);
   network net = parse_network_cfg(cfgfile);
   if(weightfile){
   load_weights(&net, weightfile);
   }
   if(clear) *net.seen = 0;

   int imgs = net.batch * net.subdivisions;

   printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
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
   clock_t time;

   load_args args = {0};
   args.w = net.w;
   args.h = net.h;
   args.threads = 8;

   args.min = net.min_crop;
   args.max = net.max_crop;
   args.flip = net.flip;
   args.angle = net.angle;
   args.aspect = net.aspect;
   args.exposure = net.exposure;
   args.saturation = net.saturation;
   args.hue = net.hue;
   args.size = net.w;
   args.hierarchy = net.hierarchy;

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

   int epoch = (*net.seen)/N;
   while(get_current_batch(net) < net.max_batches || net.max_batches == 0){
   time=clock();

   pthread_join(load_thread, 0);
   train = buffer;
   load_thread = load_data(args);

   printf("Loaded: %lf seconds\n", sec(clock()-time));
   time=clock();

#ifdef OPENCV
if(0){
int u;
for(u = 0; u < imgs; ++u){
    image im = float_to_image(net.w, net.h, 3, train.X.vals[u]);
    show_image(im, "loaded");
    cvWaitKey(0);
}
}
#endif

float loss = train_network(net, train);
free_data(train);

if(avg_loss == -1) avg_loss = loss;
avg_loss = avg_loss*.9 + loss*.1;
printf("%d, %.3f: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), (float)(*net.seen)/N, loss, avg_loss, get_current_rate(net), sec(clock()-time), *net.seen);
if(*net.seen/N > epoch){
    epoch = *net.seen/N;
    char buff[256];
    sprintf(buff, "%s/%s_%d.weights",backup_directory,base, epoch);
    save_weights(net, buff);
}
if(get_current_batch(net)%100 == 0){
    char buff[256];
    sprintf(buff, "%s/%s.backup",backup_directory,base);
    save_weights(net, buff);
}
}
char buff[256];
sprintf(buff, "%s/%s.weights", backup_directory, base);
save_weights(net, buff);

free_network(net);
free_ptrs((void**)labels, classes);
free_ptrs((void**)paths, plist->size);
free_list(plist);
free(base);
}
*/

void validate_classifier_crop(char *datacfg, char *filename, char *weightfile)
{
    int i = 0;
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);
    if (topk > classes) topk = classes;

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    clock_t time;
    float avg_acc = 0;
    float avg_topk = 0;
    int splits = m/1000;
    int num = (i+1)*m/splits - i*m/splits;

    data val, buffer;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;

    args.paths = paths;
    args.classes = classes;
    args.n = num;
    args.m = 0;
    args.labels = labels;
    args.d = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    for(i = 1; i <= splits; ++i){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        num = (i+1)*m/splits - i*m/splits;
        char **part = paths+(i*m/splits);
        if(i != splits){
            args.paths = part;
            load_thread = load_data_in_thread(args);
        }
        printf("Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
        float *acc = network_accuracies(net, val, topk);
        avg_acc += acc[0];
        avg_topk += acc[1];
        printf("%d: top 1: %f, top %d: %f, %lf seconds, %d images\n", i, avg_acc/i, topk, avg_topk/i, sec(clock()-time), val.X.rows);
        free_data(val);
    }
}

void validate_classifier_10(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network net = parse_network_cfg(filename);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);
    if (topk > classes) topk = classes;

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int* indexes = (int*)xcalloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class_id = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class_id = j;
                break;
            }
        }
        int w = net.w;
        int h = net.h;
        int shift = 32;
        image im = load_image_color(paths[i], w+shift, h+shift);
        image images[10];
        images[0] = crop_image(im, -shift, -shift, w, h);
        images[1] = crop_image(im, shift, -shift, w, h);
        images[2] = crop_image(im, 0, 0, w, h);
        images[3] = crop_image(im, -shift, shift, w, h);
        images[4] = crop_image(im, shift, shift, w, h);
        flip_image(im);
        images[5] = crop_image(im, -shift, -shift, w, h);
        images[6] = crop_image(im, shift, -shift, w, h);
        images[7] = crop_image(im, 0, 0, w, h);
        images[8] = crop_image(im, -shift, shift, w, h);
        images[9] = crop_image(im, shift, shift, w, h);
        float* pred = (float*)xcalloc(classes, sizeof(float));
        for(j = 0; j < 10; ++j){
            float *p = network_predict(net, images[j].data);
            if(net.hierarchy) hierarchy_predictions(p, net.outputs, net.hierarchy, 1);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            free_image(images[j]);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if(indexes[0] == class_id) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class_id) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
    free(indexes);
}

void validate_classifier_full(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network net = parse_network_cfg(filename);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);
    if (topk > classes) topk = classes;

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int* indexes = (int*)xcalloc(topk, sizeof(int));

    int size = net.w;
    for(i = 0; i < m; ++i){
        int class_id = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class_id = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, size);
        resize_network(&net, resized.w, resized.h);
        //show_image(im, "orig");
        //show_image(crop, "cropped");
        //cvWaitKey(0);
        float *pred = network_predict(net, resized.data);
        if(net.hierarchy) hierarchy_predictions(pred, net.outputs, net.hierarchy, 1);

        free_image(im);
        free_image(resized);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class_id) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class_id) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
    free(indexes);
}


float validate_classifier_single(char *datacfg, char *filename, char *weightfile, network *existing_net, int topk_custom)
{
    int i, j;
    network net;
    int old_batch = -1;
    if (existing_net) {
        net = *existing_net;    // for validation during training
        old_batch = net.batch;
        set_batch_network(&net, 1);
    }
    else {
        net = parse_network_cfg_custom(filename, 1, 0);
        if (weightfile) {
            load_weights(&net, weightfile);
        }
        //set_batch_network(&net, 1);
        fuse_conv_batchnorm(net);
        calculate_binary_weights(net);
    }
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net.hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);
    if (topk_custom > 0) topk = topk_custom;    // for validation during training
    if (topk > classes) topk = classes;
    printf(" TOP calculation...\n");

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int* indexes = (int*)xcalloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class_id = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class_id = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, net.w);
        image crop = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        //show_image(im, "orig");
        //show_image(crop, "cropped");
        //cvWaitKey(0);
        float *pred = network_predict(net, crop.data);
        if(net.hierarchy) hierarchy_predictions(pred, net.outputs, net.hierarchy, 1);

        if(resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class_id) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class_id) avg_topk += 1;
        }

        if (existing_net) printf("\r");
        else printf("\n");
        printf("%d: top 1: %f, top %d: %f", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
    free(indexes);
    if (existing_net) {
        set_batch_network(&net, old_batch);
    }
    float topk_result = avg_topk / i;
    return topk_result;
}

void validate_classifier_multi(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network net = parse_network_cfg(filename);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);
    if (topk > classes) topk = classes;

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);
    int scales[] = {224, 288, 320, 352, 384};
    int nscales = sizeof(scales)/sizeof(scales[0]);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int* indexes = (int*)xcalloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class_id = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class_id = j;
                break;
            }
        }
        float* pred = (float*)xcalloc(classes, sizeof(float));
        image im = load_image_color(paths[i], 0, 0);
        for(j = 0; j < nscales; ++j){
            image r = resize_min(im, scales[j]);
            resize_network(&net, r.w, r.h);
            float *p = network_predict(net, r.data);
            if(net.hierarchy) hierarchy_predictions(p, net.outputs, net.hierarchy, 1);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            flip_image(r);
            p = network_predict(net, r.data);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            if(r.data != im.data) free_image(r);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if(indexes[0] == class_id) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class_id) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
    free(indexes);
}

void try_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int layer_num)
{
    network net = parse_network_cfg_custom(cfgfile, 1, 0);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    int classes = option_find_int(options, "classes", 2);
    int top = option_find_int(options, "top", 1);
    if (top > classes) top = classes;

    char **names = get_labels(name_list);
    clock_t time;
    int* indexes = (int*)xcalloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) break;
            strtok(input, "\n");
        }
        image orig = load_image_color(input, 0, 0);
        image r = resize_min(orig, 256);
        image im = crop_image(r, (r.w - 224 - 1)/2 + 1, (r.h - 224 - 1)/2 + 1, 224, 224);
        float mean[] = {0.48263312050943, 0.45230225481413, 0.40099074308742};
        float std[] = {0.22590347483426, 0.22120921437787, 0.22103996251583};
        float var[3];
        var[0] = std[0]*std[0];
        var[1] = std[1]*std[1];
        var[2] = std[2]*std[2];

        normalize_cpu(im.data, mean, var, 1, 3, im.w*im.h);

        float *X = im.data;
        time=clock();
        float *predictions = network_predict(net, X);

        layer l = net.layers[layer_num];
        int i;
        for(i = 0; i < l.c; ++i){
            if(l.rolling_mean) printf("%f %f %f\n", l.rolling_mean[i], l.rolling_variance[i], l.scales[i]);
        }
#ifdef GPU
        cuda_pull_array(l.output_gpu, l.output, l.outputs);
#endif
        for(i = 0; i < l.outputs; ++i){
            printf("%f\n", l.output[i]);
        }
        /*

           printf("\n\nWeights\n");
           for(i = 0; i < l.n*l.size*l.size*l.c; ++i){
           printf("%f\n", l.filters[i]);
           }

           printf("\n\nBiases\n");
           for(i = 0; i < l.n; ++i){
           printf("%f\n", l.biases[i]);
           }
         */

        top_predictions(net, top, indexes);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            printf("%s: %f\n", names[index], predictions[index]);
        }
        free_image(im);
        if (filename) break;
    }
    free(indexes);
}

void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top)
{
    network net = parse_network_cfg_custom(cfgfile, 1, 0);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);

    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    int classes = option_find_int(options, "classes", 2);
    printf(" classes = %d, output in cfg = %d \n", classes, net.layers[net.n - 1].c);
    layer l = net.layers[net.n - 1];
    if (classes != l.outputs && (l.type == SOFTMAX || l.type == COST)) {
        printf("\n Error: num of filters = %d in the last conv-layer in cfg-file doesn't match to classes = %d in data-file \n",
            l.outputs, classes);
        getchar();
    }
    if (top == 0) top = option_find_int(options, "top", 1);
    if (top > classes) top = classes;

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int* indexes = (int*)xcalloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    //int size = net.w;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) break;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        image resized = resize_min(im, net.w);
        image cropped = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        printf("%d %d\n", cropped.w, cropped.h);

        float *X = cropped.data;

        double time = get_time_point();
        float *predictions = network_predict(net, X);
        printf("%s: Predicted in %lf milli-seconds.\n", input, ((double)get_time_point() - time) / 1000);

        if(net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 0);
        top_k(predictions, net.outputs, top, indexes);

        for(i = 0; i < top; ++i){
            int index = indexes[i];
            if(net.hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net.hierarchy->parent[index] >= 0) ? names[net.hierarchy->parent[index]] : "Root");
            else printf("%s: %f\n",names[index], predictions[index]);
        }

        free_image(cropped);
        if (resized.data != im.data) {
            free_image(resized);
        }
        free_image(im);

        if (filename) break;
    }
    free(indexes);
    free_network(net);
    free_list_contents_kvp(options);
    free_list(options);
}


void label_classifier(char *datacfg, char *filename, char *weightfile)
{
    int i;
    network net = parse_network_cfg(filename);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "names", "data/labels.list");
    char *test_list = option_find_str(options, "test", "data/train.list");
    int classes = option_find_int(options, "classes", 2);

    char **labels = get_labels(label_list);
    list *plist = get_paths(test_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    for(i = 0; i < m; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, net.w);
        image crop = crop_image(resized, (resized.w - net.w)/2, (resized.h - net.h)/2, net.w, net.h);
        float *pred = network_predict(net, crop.data);

        if(resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);
        int ind = max_index(pred, classes);

        printf("%s\n", labels[ind]);
    }
}


void test_classifier(char *datacfg, char *cfgfile, char *weightfile, int target_layer)
{
    int curr = 0;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);

    list *options = read_data_cfg(datacfg);

    char *test_list = option_find_str(options, "test", "data/test.list");
    int classes = option_find_int(options, "classes", 2);

    list *plist = get_paths(test_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    clock_t time;

    data val, buffer;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.classes = classes;
    args.n = net.batch;
    args.m = 0;
    args.labels = 0;
    args.d = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    for(curr = net.batch; curr < m; curr += net.batch){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        if(curr < m){
            args.paths = paths + curr;
            if (curr + net.batch > m) args.n = m - curr;
            load_thread = load_data_in_thread(args);
        }
        fprintf(stderr, "Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
        matrix pred = network_predict_data(net, val);

        int i, j;
        if (target_layer >= 0){
            //layer l = net.layers[target_layer];
        }

        for(i = 0; i < pred.rows; ++i){
            printf("%s", paths[curr-net.batch+i]);
            for(j = 0; j < pred.cols; ++j){
                printf("\t%g", pred.vals[i][j]);
            }
            printf("\n");
        }

        free_matrix(pred);

        fprintf(stderr, "%lf seconds, %d images, %d total\n", sec(clock()-time), val.X.rows, curr);
        free_data(val);
    }
}


void threat_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
    float threat = 0;
    float roll = .2;

    printf("Classifier Demo\n");
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    list *options = read_data_cfg(datacfg);

    srand(2222222);
    cap_cv * cap;

    if (filename) {
        //cap = cvCaptureFromFile(filename);
        cap = get_capture_video_stream(filename);
    }
    else {
        //cap = cvCaptureFromCAM(cam_index);
        cap = get_capture_webcam(cam_index);
    }

    int classes = option_find_int(options, "classes", 2);
    int top = option_find_int(options, "top", 1);
    if (top > classes) top = classes;

    char *name_list = option_find_str(options, "names", 0);
    char **names = get_labels(name_list);

    int* indexes = (int*)xcalloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.\n");
    create_window_cv("Threat", 0, 512, 512);
    float fps = 0;
    int i;

    int count = 0;

    while(1){
        ++count;
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        //image in = get_image_from_stream(cap);
        image in = get_image_from_stream_cpp(cap);
        if(!in.data) break;
        image in_s = resize_image(in, net.w, net.h);

        image out = in;
        int x1 = out.w / 20;
        int y1 = out.h / 20;
        int x2 = 2*x1;
        int y2 = out.h - out.h/20;

        int border = .01*out.h;
        int h = y2 - y1 - 2*border;
        int w = x2 - x1 - 2*border;

        float *predictions = network_predict(net, in_s.data);
        float curr_threat = 0;
        if(1){
            curr_threat = predictions[0] * 0 +
                predictions[1] * .6 +
                predictions[2];
        } else {
            curr_threat = predictions[218] +
                predictions[539] +
                predictions[540] +
                predictions[368] +
                predictions[369] +
                predictions[370];
        }
        threat = roll * curr_threat + (1-roll) * threat;

        draw_box_width(out, x2 + border, y1 + .02*h, x2 + .5 * w, y1 + .02*h + border, border, 0,0,0);
        if(threat > .97) {
            draw_box_width(out,  x2 + .5 * w + border,
                    y1 + .02*h - 2*border,
                    x2 + .5 * w + 6*border,
                    y1 + .02*h + 3*border, 3*border, 1,0,0);
        }
        draw_box_width(out,  x2 + .5 * w + border,
                y1 + .02*h - 2*border,
                x2 + .5 * w + 6*border,
                y1 + .02*h + 3*border, .5*border, 0,0,0);
        draw_box_width(out, x2 + border, y1 + .42*h, x2 + .5 * w, y1 + .42*h + border, border, 0,0,0);
        if(threat > .57) {
            draw_box_width(out,  x2 + .5 * w + border,
                    y1 + .42*h - 2*border,
                    x2 + .5 * w + 6*border,
                    y1 + .42*h + 3*border, 3*border, 1,1,0);
        }
        draw_box_width(out,  x2 + .5 * w + border,
                y1 + .42*h - 2*border,
                x2 + .5 * w + 6*border,
                y1 + .42*h + 3*border, .5*border, 0,0,0);

        draw_box_width(out, x1, y1, x2, y2, border, 0,0,0);
        for(i = 0; i < threat * h ; ++i){
            float ratio = (float) i / h;
            float r = (ratio < .5) ? (2*(ratio)) : 1;
            float g = (ratio < .5) ? 1 : 1 - 2*(ratio - .5);
            draw_box_width(out, x1 + border, y2 - border - i, x2 - border, y2 - border - i, 1, r, g, 0);
        }
        top_predictions(net, top, indexes);
        char buff[256];
        sprintf(buff, "tmp/threat_%06d", count);
        //save_image(out, buff);

#ifndef _WIN32
        printf("\033[2J");
        printf("\033[1;1H");
#endif
        printf("\nFPS:%.0f\n",fps);

        for(i = 0; i < top; ++i){
            int index = indexes[i];
            printf("%.1f%%: %s\n", predictions[index]*100, names[index]);
        }

        if(1){
            show_image(out, "Threat");
            wait_key_cv(10);
        }
        free_image(in_s);
        free_image(in);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}


void gun_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV_DISABLE
    int bad_cats[] = {218, 539, 540, 1213, 1501, 1742, 1911, 2415, 4348, 19223, 368, 369, 370, 1133, 1200, 1306, 2122, 2301, 2537, 2823, 3179, 3596, 3639, 4489, 5107, 5140, 5289, 6240, 6631, 6762, 7048, 7171, 7969, 7984, 7989, 8824, 8927, 9915, 10270, 10448, 13401, 15205, 18358, 18894, 18895, 19249, 19697};

    printf("Classifier Demo\n");
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    list *options = read_data_cfg(datacfg);

    srand(2222222);
    CvCapture * cap;

    if (filename) {
        //cap = cvCaptureFromFile(filename);
        cap = get_capture_video_stream(filename);
    }
    else {
        //cap = cvCaptureFromCAM(cam_index);
        cap = get_capture_webcam(cam_index);
    }

    int classes = option_find_int(options, "classes", 2);
    int top = option_find_int(options, "top", 1);
    if (top > classes) top = classes;

    char *name_list = option_find_str(options, "names", 0);
    char **names = get_labels(name_list);

    int* indexes = (int*)xcalloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow("Threat Detection", CV_WINDOW_NORMAL);
    cvResizeWindow("Threat Detection", 512, 512);
    float fps = 0;
    int i;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        //image in = get_image_from_stream(cap);
        image in = get_image_from_stream_cpp(cap);
        image in_s = resize_image(in, net.w, net.h);
        show_image(in, "Threat Detection");

        float *predictions = network_predict(net, in_s.data);
        top_predictions(net, top, indexes);

        printf("\033[2J");
        printf("\033[1;1H");

        int threat = 0;
        for(i = 0; i < sizeof(bad_cats)/sizeof(bad_cats[0]); ++i){
            int index = bad_cats[i];
            if(predictions[index] > .01){
                printf("Threat Detected!\n");
                threat = 1;
                break;
            }
        }
        if(!threat) printf("Scanning...\n");
        for(i = 0; i < sizeof(bad_cats)/sizeof(bad_cats[0]); ++i){
            int index = bad_cats[i];
            if(predictions[index] > .01){
                printf("%s\n", names[index]);
            }
        }

        free_image(in_s);
        free_image(in);

        cvWaitKey(10);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}

void demo_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int benchmark, int benchmark_layers)
{
#ifdef OPENCV
    printf("Classifier Demo\n");
    network net = parse_network_cfg_custom(cfgfile, 1, 0);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    net.benchmark_layers = benchmark_layers;
    set_batch_network(&net, 1);
    list *options = read_data_cfg(datacfg);

    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);

    srand(2222222);
    cap_cv * cap;

    if(filename){
        cap = get_capture_video_stream(filename);
    }else{
        cap = get_capture_webcam(cam_index);
    }

    int classes = option_find_int(options, "classes", 2);
    int top = option_find_int(options, "top", 1);
    if (top > classes) top = classes;

    char *name_list = option_find_str(options, "names", 0);
    char **names = get_labels(name_list);

    int* indexes = (int*)xcalloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.\n");
    if (!benchmark) create_window_cv("Classifier", 0, 512, 512);
    float fps = 0;
    int i;

    double start_time = get_time_point();
    float avg_fps = 0;
    int frame_counter = 0;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        //image in = get_image_from_stream(cap);
        image in_s, in;
        if (!benchmark) {
            in = get_image_from_stream_cpp(cap);
            in_s = resize_image(in, net.w, net.h);
            show_image(in, "Classifier");
        }
        else {
            static image tmp;
            if (!tmp.data) tmp = make_image(net.w, net.h, 3);
            in_s = tmp;
        }

        double time = get_time_point();
        float *predictions = network_predict(net, in_s.data);
        double frame_time_ms = (get_time_point() - time)/1000;
        frame_counter++;

        if(net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 1);
        top_predictions(net, top, indexes);

#ifndef _WIN32
        printf("\033[2J");
        printf("\033[1;1H");
#endif


        if (!benchmark) {
            printf("\rFPS: %.2f  (use -benchmark command line flag for correct measurement)\n", fps);
            for (i = 0; i < top; ++i) {
                int index = indexes[i];
                printf("%.1f%%: %s\n", predictions[index] * 100, names[index]);
            }
            printf("\n");

            free_image(in_s);
            free_image(in);

            int c = wait_key_cv(10);// cvWaitKey(10);
            if (c == 27 || c == 1048603) break;
        }
        else {
            printf("\rFPS: %.2f \t AVG_FPS = %.2f ", fps, avg_fps);
        }

        //gettimeofday(&tval_after, NULL);
        //timersub(&tval_after, &tval_before, &tval_result);
        //float curr = 1000000.f/((long int)tval_result.tv_usec);
        float curr = 1000.f / frame_time_ms;
        if (fps == 0) fps = curr;
        else fps = .9*fps + .1*curr;

        float spent_time = (get_time_point() - start_time) / 1000000;
        if (spent_time >= 3.0f) {
            //printf(" spent_time = %f \n", spent_time);
            avg_fps = frame_counter / spent_time;
            frame_counter = 0;
            start_time = get_time_point();
        }
    }
#endif
}


void run_classifier(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    int mjpeg_port = find_int_arg(argc, argv, "-mjpeg_port", -1);
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = (int*)xcalloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int dont_show = find_arg(argc, argv, "-dont_show");
    int benchmark = find_arg(argc, argv, "-benchmark");
    int benchmark_layers = find_arg(argc, argv, "-benchmark_layers");
    if (benchmark_layers) benchmark = 1;
    int dontuse_opencv = find_arg(argc, argv, "-dontuse_opencv");
    int show_imgs = find_arg(argc, argv, "-show_imgs");
    int calc_topk = find_arg(argc, argv, "-topk");
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int top = find_int_arg(argc, argv, "-t", 0);
    int clear = find_arg(argc, argv, "-clear");
    char *data = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    char *layer_s = (argc > 7) ? argv[7]: 0;
    int layer = layer_s ? atoi(layer_s) : -1;
    char* chart_path = find_char_arg(argc, argv, "-chart", 0);
    if(0==strcmp(argv[2], "predict")) predict_classifier(data, cfg, weights, filename, top);
    else if(0==strcmp(argv[2], "try")) try_classifier(data, cfg, weights, filename, atoi(layer_s));
    else if(0==strcmp(argv[2], "train")) train_classifier(data, cfg, weights, gpus, ngpus, clear, dontuse_opencv, dont_show, mjpeg_port, calc_topk, show_imgs, chart_path);
    else if(0==strcmp(argv[2], "demo")) demo_classifier(data, cfg, weights, cam_index, filename, benchmark, benchmark_layers);
    else if(0==strcmp(argv[2], "gun")) gun_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "threat")) threat_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "test")) test_classifier(data, cfg, weights, layer);
    else if(0==strcmp(argv[2], "label")) label_classifier(data, cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_classifier_single(data, cfg, weights, NULL, -1);
    else if(0==strcmp(argv[2], "validmulti")) validate_classifier_multi(data, cfg, weights);
    else if(0==strcmp(argv[2], "valid10")) validate_classifier_10(data, cfg, weights);
    else if(0==strcmp(argv[2], "validcrop")) validate_classifier_crop(data, cfg, weights);
    else if(0==strcmp(argv[2], "validfull")) validate_classifier_full(data, cfg, weights);

    if (gpus && gpu_list && ngpus > 1) free(gpus);
}
