#include <stdlib.h>
#include "darknet.h"
#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"

#ifndef __COMPAR_FN_T
#define __COMPAR_FN_T
typedef int (*__compar_fn_t)(const void*, const void*);
#ifdef __USE_GNU
typedef __compar_fn_t comparison_fn_t;
#endif
#endif

#include "http_stream.h"

int check_mistakes = 0;

static int coco_ids[] = { 1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90 };

void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int dont_show, int calc_map, int mjpeg_port, int show_imgs, int benchmark_layers, char* chart_path)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.txt");
    char *valid_images = option_find_str(options, "valid", train_images);
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    network net_map;
    if (calc_map) {
        FILE* valid_file = fopen(valid_images, "r");
        if (!valid_file) {
            printf("\n Error: There is no %s file for mAP calculation!\n Don't use -map flag.\n Or set valid=%s in your %s file. \n", valid_images, train_images, datacfg);
            getchar();
            exit(-1);
        }
        else fclose(valid_file);

        cuda_set_device(gpus[0]);
        printf(" Prepare additional network for mAP calculation...\n");
        net_map = parse_network_cfg_custom(cfgfile, 1, 1);
        net_map.benchmark_layers = benchmark_layers;
        const int net_classes = net_map.layers[net_map.n - 1].classes;

        int k;  // free memory unnecessary arrays
        for (k = 0; k < net_map.n - 1; ++k) free_layer_custom(net_map.layers[k], 1);

        char *name_list = option_find_str(options, "names", "data/names.list");
        int names_size = 0;
        char **names = get_labels_custom(name_list, &names_size);
        if (net_classes != names_size) {
            printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
                name_list, names_size, net_classes, cfgfile);
            if (net_classes > names_size) getchar();
        }
        free_ptrs((void**)names, net_map.layers[net_map.n - 1].classes);
    }

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network* nets = (network*)xcalloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int k;
    for (k = 0; k < ngpus; ++k) {
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[k]);
#endif
        nets[k] = parse_network_cfg(cfgfile);
        nets[k].benchmark_layers = benchmark_layers;
        if (weightfile) {
            load_weights(&nets[k], weightfile);
        }
        if (clear) {
            *nets[k].seen = 0;
            *nets[k].cur_iteration = 0;
        }
        nets[k].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];

    const int actual_batch_size = net.batch * net.subdivisions;
    if (actual_batch_size == 1) {
        printf("\n Error: You set incorrect value batch=1 for Training! You should set batch=64 subdivision=64 \n");
        getchar();
    }
    else if (actual_batch_size < 8) {
        printf("\n Warning: You set batch=%d lower than 64! It is recommended to set batch=64 subdivision=64 \n", actual_batch_size);
    }

    int imgs = net.batch * net.subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    data train, buffer;

    layer l = net.layers[net.n - 1];

    int classes = l.classes;

    list *plist = get_paths(train_images);
    int train_images_num = plist->size;
    char **paths = (char **)list_to_array(plist);

    const int init_w = net.w;
    const int init_h = net.h;
    const int init_b = net.batch;
    int iter_save, iter_save_last, iter_map;
    iter_save = get_current_iteration(net);
    iter_save_last = get_current_iteration(net);
    iter_map = get_current_iteration(net);
    float mean_average_precision = -1;
    float best_map = mean_average_precision;

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.flip = net.flip;
    args.jitter = l.jitter;
    args.resize = l.resize;
    args.num_boxes = l.max_boxes;
    net.num_boxes = args.num_boxes;
    net.train_images_num = train_images_num;
    args.d = &buffer;
    args.type = DETECTION_DATA;
    args.threads = 64;    // 16 or 64

    args.angle = net.angle;
    args.gaussian_noise = net.gaussian_noise;
    args.blur = net.blur;
    args.mixup = net.mixup;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    args.letter_box = net.letter_box;
    if (dont_show && show_imgs) show_imgs = 2;
    args.show_imgs = show_imgs;

#ifdef OPENCV
    //int num_threads = get_num_threads();
    //if(num_threads > 2) args.threads = get_num_threads() - 2;
    args.threads = 6 * ngpus;   // 3 for - Amazon EC2 Tesla V100: p3.2xlarge (8 logical cores) - p3.16xlarge
    //args.threads = 12 * ngpus;    // Ryzen 7 2700X (16 logical cores)
    mat_cv* img = NULL;
    float max_img_loss = net.max_chart_loss;
    int number_of_lines = 100;
    int img_size = 1000;
    char windows_name[100];
    sprintf(windows_name, "chart_%s.png", base);
    img = draw_train_chart(windows_name, max_img_loss, net.max_batches, number_of_lines, img_size, dont_show, chart_path);
#endif    //OPENCV
    if (net.track) {
        args.track = net.track;
        args.augment_speed = net.augment_speed;
        if (net.sequential_subdivisions) args.threads = net.sequential_subdivisions * ngpus;
        else args.threads = net.subdivisions * ngpus;
        args.mini_batch = net.batch / net.time_steps;
        printf("\n Tracking! batch = %d, subdiv = %d, time_steps = %d, mini_batch = %d \n", net.batch, net.subdivisions, net.time_steps, args.mini_batch);
    }
    //printf(" imgs = %d \n", imgs);

    pthread_t load_thread = load_data(args);

    int count = 0;
    double time_remaining, avg_time = -1, alpha_time = 0.01;

    //while(i*imgs < N*120){
    while (get_current_iteration(net) < net.max_batches) {
        if (l.random && count++ % 10 == 0) {
            float rand_coef = 1.4;
            if (l.random != 1.0) rand_coef = l.random;
            printf("Resizing, random_coef = %.2f \n", rand_coef);
            float random_val = rand_scale(rand_coef);    // *x or /x
            int dim_w = roundl(random_val*init_w / net.resize_step + 1) * net.resize_step;
            int dim_h = roundl(random_val*init_h / net.resize_step + 1) * net.resize_step;
            if (random_val < 1 && (dim_w > init_w || dim_h > init_h)) dim_w = init_w, dim_h = init_h;

            int max_dim_w = roundl(rand_coef*init_w / net.resize_step + 1) * net.resize_step;
            int max_dim_h = roundl(rand_coef*init_h / net.resize_step + 1) * net.resize_step;

            // at the beginning (check if enough memory) and at the end (calc rolling mean/variance)
            if (avg_loss < 0 || get_current_iteration(net) > net.max_batches - 100) {
                dim_w = max_dim_w;
                dim_h = max_dim_h;
            }

            if (dim_w < net.resize_step) dim_w = net.resize_step;
            if (dim_h < net.resize_step) dim_h = net.resize_step;
            int dim_b = (init_b * max_dim_w * max_dim_h) / (dim_w * dim_h);
            int new_dim_b = (int)(dim_b * 0.8);
            if (new_dim_b > init_b) dim_b = new_dim_b;

            args.w = dim_w;
            args.h = dim_h;

            int k;
            if (net.dynamic_minibatch) {
                for (k = 0; k < ngpus; ++k) {
                    (*nets[k].seen) = init_b * net.subdivisions * get_current_iteration(net); // remove this line, when you will save to weights-file both: seen & cur_iteration
                    nets[k].batch = dim_b;
                    int j;
                    for (j = 0; j < nets[k].n; ++j)
                        nets[k].layers[j].batch = dim_b;
                }
                net.batch = dim_b;
                imgs = net.batch * net.subdivisions * ngpus;
                args.n = imgs;
                printf("\n %d x %d  (batch = %d) \n", dim_w, dim_h, net.batch);
            }
            else
                printf("\n %d x %d \n", dim_w, dim_h);

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            for (k = 0; k < ngpus; ++k) {
                resize_network(nets + k, dim_w, dim_h);
            }
            net = nets[0];
        }
        double time = what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        if (net.track) {
            net.sequential_subdivisions = get_current_seq_subdivisions(net);
            args.threads = net.sequential_subdivisions * ngpus;
            printf(" sequential_subdivisions = %d, sequence = %d \n", net.sequential_subdivisions, get_sequence_value(net));
        }
        load_thread = load_data(args);

        /*
        int k;
        for(k = 0; k < l.max_boxes; ++k){
        box b = float_to_box(train.y.vals[10] + 1 + k*5);
        if(!b.x) break;
        printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
        }
        image im = float_to_image(448, 448, 3, train.X.vals[10]);
        int k;
        for(k = 0; k < l.max_boxes; ++k){
        box b = float_to_box(train.y.vals[10] + 1 + k*5);
        printf("%d %d %d %d\n", truth.x, truth.y, truth.w, truth.h);
        draw_bbox(im, b, 8, 1,0,0);
        }
        save_image(im, "truth11");
        */

        const double load_time = (what_time_is_it_now() - time);
        printf("Loaded: %lf seconds", load_time);
        if (load_time > 0.1 && avg_loss > 0) printf(" - performance bottleneck on CPU or Disk HDD/SSD");
        printf("\n");

        time = what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if (ngpus == 1) {
            int wait_key = (dont_show) ? 0 : 1;
            loss = train_network_waitkey(net, train, wait_key);
        }
        else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0 || avg_loss != avg_loss) avg_loss = loss;    // if(-inf or nan)
        avg_loss = avg_loss*.9 + loss*.1;

        const int iteration = get_current_iteration(net);
        //i = get_current_batch(net);

        int calc_map_for_each = 4 * train_images_num / (net.batch * net.subdivisions);  // calculate mAP for each 4 Epochs
        calc_map_for_each = fmax(calc_map_for_each, 100);
        int next_map_calc = iter_map + calc_map_for_each;
        next_map_calc = fmax(next_map_calc, net.burn_in);
        //next_map_calc = fmax(next_map_calc, 400);
        if (calc_map) {
            printf("\n (next mAP calculation at %d iterations) ", next_map_calc);
            if (mean_average_precision > 0) printf("\n Last accuracy mAP@0.5 = %2.2f %%, best = %2.2f %% ", mean_average_precision * 100, best_map * 100);
        }

        if (net.cudnn_half) {
            if (iteration < net.burn_in * 3) fprintf(stderr, "\n Tensor Cores are disabled until the first %d iterations are reached.\n", 3 * net.burn_in);
            else fprintf(stderr, "\n Tensor Cores are used.\n");
            fflush(stderr);
        }
        printf("\n %d: %f, %f avg loss, %f rate, %lf seconds, %d images, %f hours left\n", iteration, loss, avg_loss, get_current_rate(net), (what_time_is_it_now() - time), iteration*imgs, avg_time);
        fflush(stdout);

        int draw_precision = 0;
        if (calc_map && (iteration >= next_map_calc || iteration == net.max_batches)) {
            if (l.random) {
                printf("Resizing to initial size: %d x %d ", init_w, init_h);
                args.w = init_w;
                args.h = init_h;
                int k;
                if (net.dynamic_minibatch) {
                    for (k = 0; k < ngpus; ++k) {
                        for (k = 0; k < ngpus; ++k) {
                            nets[k].batch = init_b;
                            int j;
                            for (j = 0; j < nets[k].n; ++j)
                                nets[k].layers[j].batch = init_b;
                        }
                    }
                    net.batch = init_b;
                    imgs = init_b * net.subdivisions * ngpus;
                    args.n = imgs;
                    printf("\n %d x %d  (batch = %d) \n", init_w, init_h, init_b);
                }
                pthread_join(load_thread, 0);
                free_data(train);
                train = buffer;
                load_thread = load_data(args);
                for (k = 0; k < ngpus; ++k) {
                    resize_network(nets + k, init_w, init_h);
                }
                net = nets[0];
            }

            copy_weights_net(net, &net_map);

            // combine Training and Validation networks
            //network net_combined = combine_train_valid_networks(net, net_map);

            iter_map = iteration;
            mean_average_precision = validate_detector_map(datacfg, cfgfile, weightfile, 0.25, 0.5, 0, net.letter_box, &net_map);// &net_combined);
            printf("\n mean_average_precision (mAP@0.5) = %f \n", mean_average_precision);
            if (mean_average_precision > best_map) {
                best_map = mean_average_precision;
                printf("New best mAP!\n");
                char buff[256];
                sprintf(buff, "%s/%s_best.weights", backup_directory, base);
                save_weights(net, buff);
            }

            draw_precision = 1;
        }
        time_remaining = ((net.max_batches - iteration) / ngpus)*(what_time_is_it_now() - time + load_time) / 60 / 60;
        // set initial value, even if resume training from 10000 iteration
        if (avg_time < 0) avg_time = time_remaining;
        else avg_time = alpha_time * time_remaining + (1 -  alpha_time) * avg_time;
#ifdef OPENCV
        draw_train_loss(windows_name, img, img_size, avg_loss, max_img_loss, iteration, net.max_batches, mean_average_precision, draw_precision, "mAP%", dont_show, mjpeg_port, avg_time);
#endif    // OPENCV

        //if (i % 1000 == 0 || (i < 1000 && i % 100 == 0)) {
        //if (i % 100 == 0) {
        if (iteration >= (iter_save + 1000) || iteration % 1000 == 0) {
            iter_save = iteration;
#ifdef GPU
            if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, iteration);
            save_weights(net, buff);
        }

        if (iteration >= (iter_save_last + 100) || (iteration % 100 == 0 && iteration > 1)) {
            iter_save_last = iteration;
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

    // free memory
    pthread_join(load_thread, 0);
    free_data(buffer);

    free_load_threads(&args);

    free(base);
    free(paths);
    free_list_contents(plist);
    free_list(plist);

    free_list_contents_kvp(options);
    free_list(options);

    for (k = 0; k < ngpus; ++k) free_network(nets[k]);
    free(nets);
    //free_network(net);

    if (calc_map) {
        net_map.n = 0;
        free_network(net_map);
    }
}


static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if (c) p = c;
    return atoi(p + 1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    //int image_id = get_coco_image_id(image_path);
    char *p = basecfg(image_path);
    int image_id = atoi(p);
    for (i = 0; i < num_boxes; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for (j = 0; j < classes; ++j) {
            if (dets[i].prob[j] > 0) {
                char buff[1024];
                sprintf(buff, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
                fprintf(fp, buff);
                //printf("%s", buff);
            }
        }
    }
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for (i = 0; i < total; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for (j = 0; j < classes; ++j) {
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for (i = 0; i < total; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for (j = 0; j < classes; ++j) {
            int myclass = j;
            if (dets[i].prob[myclass] > 0) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j + 1, dets[i].prob[myclass],
                xmin, ymin, xmax, ymax);
        }
    }
}

static void print_kitti_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h, char *outfile, char *prefix)
{
    char *kitti_ids[] = { "car", "pedestrian", "cyclist" };
    FILE *fpd = 0;
    char buffd[1024];
    snprintf(buffd, 1024, "%s/%s/data/%s.txt", prefix, outfile, id);

    fpd = fopen(buffd, "w");
    int i, j;
    for (i = 0; i < total; ++i)
    {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for (j = 0; j < classes; ++j)
        {
            //if (dets[i].prob[j]) fprintf(fpd, "%s 0 0 0 %f %f %f %f -1 -1 -1 -1 0 0 0 %f\n", kitti_ids[j], xmin, ymin, xmax, ymax, dets[i].prob[j]);
            if (dets[i].prob[j]) fprintf(fpd, "%s -1 -1 -10 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 %f\n", kitti_ids[j], xmin, ymin, xmax, ymax, dets[i].prob[j]);
        }
    }
    fclose(fpd);
}

static void eliminate_bdd(char *buf, char *a)
{
    int n = 0;
    int i, k;
    for (i = 0; buf[i] != '\0'; i++)
    {
        if (buf[i] == a[n])
        {
            k = i;
            while (buf[i] == a[n])
            {
                if (a[++n] == '\0')
                {
                    for (k; buf[k + n] != '\0'; k++)
                    {
                        buf[k] = buf[k + n];
                    }
                    buf[k] = '\0';
                    break;
                }
                i++;
            }
            n = 0; i--;
        }
    }
}

static void get_bdd_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    eliminate_bdd(p, ".jpg");
    eliminate_bdd(p, "/");
    strcpy(filename, p);
}

static void print_bdd_detections(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    char *bdd_ids[] = { "bike" , "bus" , "car" , "motor" ,"person", "rider", "traffic light", "traffic sign", "train", "truck" };
    get_bdd_image_id(image_path);
    int i, j;

    for (i = 0; i < num_boxes; ++i)
    {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx1 = xmin;
        float by1 = ymin;
        float bx2 = xmax;
        float by2 = ymax;

        for (j = 0; j < classes; ++j)
        {
            if (dets[i].prob[j])
            {
                fprintf(fp, "\t{\n\t\t\"name\":\"%s\",\n\t\t\"category\":\"%s\",\n\t\t\"bbox\":[%f, %f, %f, %f],\n\t\t\"score\":%f\n\t},\n", image_path, bdd_ids[j], bx1, by1, bx2, by2, dets[i].prob[j]);
            }
        }
    }
}

void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    //set_batch_network(&net, 1);
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n - 1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    int bdd = 0;
    int kitti = 0;

    if (0 == strcmp(type, "coco")) {
        if (!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    }
    else if (0 == strcmp(type, "bdd")) {
        if (!outfile) outfile = "bdd_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        bdd = 1;
    }
    else if (0 == strcmp(type, "kitti")) {
        char buff2[1024];
        if (!outfile) outfile = "kitti_results";
        printf("%s\n", outfile);
        snprintf(buff, 1024, "%s/%s", prefix, outfile);
        int mkd = make_directory(buff, 0777);
        snprintf(buff2, 1024, "%s/%s/data", prefix, outfile);
        int mkd2 = make_directory(buff2, 0777);
        kitti = 1;
    }
    else if (0 == strcmp(type, "imagenet")) {
        if (!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    }
    else {
        if (!outfile) outfile = "comp4_det_test_";
        fps = (FILE**) xcalloc(classes, sizeof(FILE *));
        for (j = 0; j < classes; ++j) {
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    int m = plist->size;
    int i = 0;
    int t;

    float thresh = .001;
    float nms = .45;

    int nthreads = 4;
    if (m < 4) nthreads = m;
    image* val = (image*)xcalloc(nthreads, sizeof(image));
    image* val_resized = (image*)xcalloc(nthreads, sizeof(image));
    image* buf = (image*)xcalloc(nthreads, sizeof(image));
    image* buf_resized = (image*)xcalloc(nthreads, sizeof(image));
    pthread_t* thr = (pthread_t*)xcalloc(nthreads, sizeof(pthread_t));

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    args.type = IMAGE_DATA;
    const int letter_box = net.letter_box;
    if (letter_box) args.type = LETTERBOX_DATA;

    for (t = 0; t < nthreads; ++t) {
        args.path = paths[i + t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for (i = nthreads; i < m + nthreads; i += nthreads) {
        fprintf(stderr, "%d\n", i);
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for (t = 0; t < nthreads && i + t < m; ++t) {
            args.path = paths[i + t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            char *path = paths[i + t - nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(&net, w, h, thresh, .5, map, 0, &nboxes, letter_box);
            if (nms) {
                if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
                else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }

            if (coco) {
                print_cocos(fp, path, dets, nboxes, classes, w, h);
            }
            else if (imagenet) {
                print_imagenet_detections(fp, i + t - nthreads + 1, dets, nboxes, classes, w, h);
            }
            else if (bdd) {
                print_bdd_detections(fp, path, dets, nboxes, classes, w, h);
            }
            else if (kitti) {
                print_kitti_detections(fps, id, dets, nboxes, classes, w, h, outfile, prefix);
            }
            else {
                print_detector_detections(fps, id, dets, nboxes, classes, w, h);
            }

            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    if (fps) {
        for (j = 0; j < classes; ++j) {
            fclose(fps[j]);
        }
        free(fps);
    }
    if (coco) {
#ifdef WIN32
        fseek(fp, -3, SEEK_CUR);
#else
        fseek(fp, -2, SEEK_CUR);
#endif
        fprintf(fp, "\n]\n");
    }

    if (bdd) {
#ifdef WIN32
        fseek(fp, -3, SEEK_CUR);
#else
        fseek(fp, -2, SEEK_CUR);
#endif
        fprintf(fp, "\n]\n");
        fclose(fp);
    }

    if (fp) fclose(fp);

    if (val) free(val);
    if (val_resized) free(val_resized);
    if (thr) free(thr);
    if (buf) free(buf);
    if (buf_resized) free(buf_resized);

    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)time(0) - start);
}

void validate_detector_recall(char *datacfg, char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    //set_batch_network(&net, 1);
    fuse_conv_batchnorm(net);
    srand(time(0));

    //list *plist = get_paths("data/coco_val_5k.list");
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.txt");
    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    //layer l = net.layers[net.n - 1];

    int j, k;

    int m = plist->size;
    int i = 0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = .4;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for (i = 0; i < m; ++i) {
        char *path = paths[i];
        image orig = load_image(path, 0, 0, net.c);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        int nboxes = 0;
        int letterbox = 0;
        detection *dets = get_network_boxes(&net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes, letterbox);
        if (nms) do_nms_obj(dets, nboxes, 1, nms);

        char labelpath[4096];
        replace_image_to_label(path, labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for (k = 0; k < nboxes; ++k) {
            if (dets[k].objectness > thresh) {
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
            float best_iou = 0;
            for (k = 0; k < nboxes; ++k) {
                float iou = box_iou(dets[k].bbox, t);
                if (dets[k].objectness > thresh && iou > best_iou) {
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if (best_iou > iou_thresh) {
                ++correct;
            }
        }
        //fprintf(stderr, " %s - %s - ", paths[i], labelpath);
        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.*correct / total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

typedef struct {
    box b;
    float p;
    int class_id;
    int image_index;
    int truth_flag;
    int unique_truth_index;
} box_prob;

int detections_comparator(const void *pa, const void *pb)
{
    box_prob a = *(const box_prob *)pa;
    box_prob b = *(const box_prob *)pb;
    float diff = a.p - b.p;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

float validate_detector_map(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou, const float iou_thresh, const int map_points, int letter_box, network *existing_net)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.txt");
    char *difficult_valid_images = option_find_str(options, "difficult", NULL);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);
    //char *mapf = option_find_str(options, "map", 0);
    //int *map = 0;
    //if (mapf) map = read_map(mapf);
    FILE* reinforcement_fd = NULL;

    network net;
    //int initial_batch;
    if (existing_net) {
        char *train_images = option_find_str(options, "train", "data/train.txt");
        valid_images = option_find_str(options, "valid", train_images);
        net = *existing_net;
        remember_network_recurrent_state(*existing_net);
        free_network_recurrent_state(*existing_net);
    }
    else {
        net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
        if (weightfile) {
            load_weights(&net, weightfile);
        }
        //set_batch_network(&net, 1);
        fuse_conv_batchnorm(net);
        calculate_binary_weights(net);
    }
    if (net.layers[net.n - 1].classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        getchar();
    }
    srand(time(0));
    printf("\n calculation mAP (mean average precision)...\n");

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    char **paths_dif = NULL;
    if (difficult_valid_images) {
        list *plist_dif = get_paths(difficult_valid_images);
        paths_dif = (char **)list_to_array(plist_dif);
    }


    layer l = net.layers[net.n - 1];
    int classes = l.classes;

    int m = plist->size;
    int i = 0;
    int t;

    const float thresh = .005;
    const float nms = .45;
    //const float iou_thresh = 0.5;

    int nthreads = 4;
    if (m < 4) nthreads = m;
    image* val = (image*)xcalloc(nthreads, sizeof(image));
    image* val_resized = (image*)xcalloc(nthreads, sizeof(image));
    image* buf = (image*)xcalloc(nthreads, sizeof(image));
    image* buf_resized = (image*)xcalloc(nthreads, sizeof(image));
    pthread_t* thr = (pthread_t*)xcalloc(nthreads, sizeof(pthread_t));

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    if (letter_box) args.type = LETTERBOX_DATA;
    else args.type = IMAGE_DATA;

    //const float thresh_calc_avg_iou = 0.24;
    float avg_iou = 0;
    int tp_for_thresh = 0;
    int fp_for_thresh = 0;

    box_prob* detections = (box_prob*)xcalloc(1, sizeof(box_prob));
    int detections_count = 0;
    int unique_truth_count = 0;

    int* truth_classes_count = (int*)xcalloc(classes, sizeof(int));

    // For multi-class precision and recall computation
    float *avg_iou_per_class = (float*)xcalloc(classes, sizeof(float));
    int *tp_for_thresh_per_class = (int*)xcalloc(classes, sizeof(int));
    int *fp_for_thresh_per_class = (int*)xcalloc(classes, sizeof(int));

    for (t = 0; t < nthreads; ++t) {
        args.path = paths[i + t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for (i = nthreads; i < m + nthreads; i += nthreads) {
        fprintf(stderr, "\r%d", i);
        for (t = 0; t < nthreads && (i + t - nthreads) < m; ++t) {
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for (t = 0; t < nthreads && (i + t) < m; ++t) {
            args.path = paths[i + t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            const int image_index = i + t - nthreads;
            char *path = paths[image_index];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);

            int nboxes = 0;
            float hier_thresh = 0;
            detection *dets;
            if (args.type == LETTERBOX_DATA) {
                dets = get_network_boxes(&net, val[t].w, val[t].h, thresh, hier_thresh, 0, 1, &nboxes, letter_box);
            }
            else {
                dets = get_network_boxes(&net, 1, 1, thresh, hier_thresh, 0, 0, &nboxes, letter_box);
            }
            //detection *dets = get_network_boxes(&net, val[t].w, val[t].h, thresh, hier_thresh, 0, 1, &nboxes, letter_box); // for letter_box=1
            if (nms) {
                if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
                else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }
            //if (nms) do_nms_obj(dets, nboxes, l.classes, nms);

            char labelpath[4096];
            replace_image_to_label(path, labelpath);
            int num_labels = 0;
            box_label *truth = read_boxes(labelpath, &num_labels);
            int j;
            for (j = 0; j < num_labels; ++j) {
                truth_classes_count[truth[j].id]++;
            }

            // difficult
            box_label *truth_dif = NULL;
            int num_labels_dif = 0;
            if (paths_dif)
            {
                char *path_dif = paths_dif[image_index];

                char labelpath_dif[4096];
                replace_image_to_label(path_dif, labelpath_dif);

                truth_dif = read_boxes(labelpath_dif, &num_labels_dif);
            }

            const int checkpoint_detections_count = detections_count;

            int i;
            for (i = 0; i < nboxes; ++i) {

                int class_id;
                for (class_id = 0; class_id < classes; ++class_id) {
                    float prob = dets[i].prob[class_id];
                    if (prob > 0) {
                        detections_count++;
                        detections = (box_prob*)xrealloc(detections, detections_count * sizeof(box_prob));
                        detections[detections_count - 1].b = dets[i].bbox;
                        detections[detections_count - 1].p = prob;
                        detections[detections_count - 1].image_index = image_index;
                        detections[detections_count - 1].class_id = class_id;
                        detections[detections_count - 1].truth_flag = 0;
                        detections[detections_count - 1].unique_truth_index = -1;

                        int truth_index = -1;
                        float max_iou = 0;
                        for (j = 0; j < num_labels; ++j)
                        {
                            box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
                            //printf(" IoU = %f, prob = %f, class_id = %d, truth[j].id = %d \n",
                            //    box_iou(dets[i].bbox, t), prob, class_id, truth[j].id);
                            float current_iou = box_iou(dets[i].bbox, t);
                            if (current_iou > iou_thresh && class_id == truth[j].id) {
                                if (current_iou > max_iou) {
                                    max_iou = current_iou;
                                    truth_index = unique_truth_count + j;
                                }
                            }
                        }

                        // best IoU
                        if (truth_index > -1) {
                            detections[detections_count - 1].truth_flag = 1;
                            detections[detections_count - 1].unique_truth_index = truth_index;
                        }
                        else {
                            // if object is difficult then remove detection
                            for (j = 0; j < num_labels_dif; ++j) {
                                box t = { truth_dif[j].x, truth_dif[j].y, truth_dif[j].w, truth_dif[j].h };
                                float current_iou = box_iou(dets[i].bbox, t);
                                if (current_iou > iou_thresh && class_id == truth_dif[j].id) {
                                    --detections_count;
                                    break;
                                }
                            }
                        }

                        // calc avg IoU, true-positives, false-positives for required Threshold
                        if (prob > thresh_calc_avg_iou) {
                            int z, found = 0;
                            for (z = checkpoint_detections_count; z < detections_count - 1; ++z) {
                                if (detections[z].unique_truth_index == truth_index) {
                                    found = 1; break;
                                }
                            }

                            if (truth_index > -1 && found == 0) {
                                avg_iou += max_iou;
                                ++tp_for_thresh;
                                avg_iou_per_class[class_id] += max_iou;
                                tp_for_thresh_per_class[class_id]++;
                            }
                            else{
                                fp_for_thresh++;
                                fp_for_thresh_per_class[class_id]++;
                            }
                        }
                    }
                }
            }

            unique_truth_count += num_labels;

            //static int previous_errors = 0;
            //int total_errors = fp_for_thresh + (unique_truth_count - tp_for_thresh);
            //int errors_in_this_image = total_errors - previous_errors;
            //previous_errors = total_errors;
            //if(reinforcement_fd == NULL) reinforcement_fd = fopen("reinforcement.txt", "wb");
            //char buff[1000];
            //sprintf(buff, "%s\n", path);
            //if(errors_in_this_image > 0) fwrite(buff, sizeof(char), strlen(buff), reinforcement_fd);

            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }

    //for (t = 0; t < nthreads; ++t) {
    //    pthread_join(thr[t], 0);
    //}

    if ((tp_for_thresh + fp_for_thresh) > 0)
        avg_iou = avg_iou / (tp_for_thresh + fp_for_thresh);

    int class_id;
    for(class_id = 0; class_id < classes; class_id++){
        if ((tp_for_thresh_per_class[class_id] + fp_for_thresh_per_class[class_id]) > 0)
            avg_iou_per_class[class_id] = avg_iou_per_class[class_id] / (tp_for_thresh_per_class[class_id] + fp_for_thresh_per_class[class_id]);
    }

    // SORT(detections)
    qsort(detections, detections_count, sizeof(box_prob), detections_comparator);

    typedef struct {
        double precision;
        double recall;
        int tp, fp, fn;
    } pr_t;

    // for PR-curve
    pr_t** pr = (pr_t**)xcalloc(classes, sizeof(pr_t*));
    for (i = 0; i < classes; ++i) {
        pr[i] = (pr_t*)xcalloc(detections_count, sizeof(pr_t));
    }
    printf("\n detections_count = %d, unique_truth_count = %d  \n", detections_count, unique_truth_count);


    int* detection_per_class_count = (int*)xcalloc(classes, sizeof(int));
    for (j = 0; j < detections_count; ++j) {
        detection_per_class_count[detections[j].class_id]++;
    }

    int* truth_flags = (int*)xcalloc(unique_truth_count, sizeof(int));

    int rank;
    for (rank = 0; rank < detections_count; ++rank) {
        if (rank % 100 == 0)
            printf(" rank = %d of ranks = %d \r", rank, detections_count);

        if (rank > 0) {
            int class_id;
            for (class_id = 0; class_id < classes; ++class_id) {
                pr[class_id][rank].tp = pr[class_id][rank - 1].tp;
                pr[class_id][rank].fp = pr[class_id][rank - 1].fp;
            }
        }

        box_prob d = detections[rank];
        // if (detected && isn't detected before)
        if (d.truth_flag == 1) {
            if (truth_flags[d.unique_truth_index] == 0)
            {
                truth_flags[d.unique_truth_index] = 1;
                pr[d.class_id][rank].tp++;    // true-positive
            } else
                pr[d.class_id][rank].fp++;
        }
        else {
            pr[d.class_id][rank].fp++;    // false-positive
        }

        for (i = 0; i < classes; ++i)
        {
            const int tp = pr[i][rank].tp;
            const int fp = pr[i][rank].fp;
            const int fn = truth_classes_count[i] - tp;    // false-negative = objects - true-positive
            pr[i][rank].fn = fn;

            if ((tp + fp) > 0) pr[i][rank].precision = (double)tp / (double)(tp + fp);
            else pr[i][rank].precision = 0;

            if ((tp + fn) > 0) pr[i][rank].recall = (double)tp / (double)(tp + fn);
            else pr[i][rank].recall = 0;

            if (rank == (detections_count - 1) && detection_per_class_count[i] != (tp + fp)) {    // check for last rank
                    printf(" class_id: %d - detections = %d, tp+fp = %d, tp = %d, fp = %d \n", i, detection_per_class_count[i], tp+fp, tp, fp);
            }
        }
    }

    free(truth_flags);


    double mean_average_precision = 0;

    for (i = 0; i < classes; ++i) {
        double avg_precision = 0;

        // MS COCO - uses 101-Recall-points on PR-chart.
        // PascalVOC2007 - uses 11-Recall-points on PR-chart.
        // PascalVOC2010-2012 - uses Area-Under-Curve on PR-chart.
        // ImageNet - uses Area-Under-Curve on PR-chart.

        // correct mAP calculation: ImageNet, PascalVOC 2010-2012
        if (map_points == 0)
        {
            double last_recall = pr[i][detections_count - 1].recall;
            double last_precision = pr[i][detections_count - 1].precision;
            for (rank = detections_count - 2; rank >= 0; --rank)
            {
                double delta_recall = last_recall - pr[i][rank].recall;
                last_recall = pr[i][rank].recall;

                if (pr[i][rank].precision > last_precision) {
                    last_precision = pr[i][rank].precision;
                }

                avg_precision += delta_recall * last_precision;
            }
        }
        // MSCOCO - 101 Recall-points, PascalVOC - 11 Recall-points
        else
        {
            int point;
            for (point = 0; point < map_points; ++point) {
                double cur_recall = point * 1.0 / (map_points-1);
                double cur_precision = 0;
                for (rank = 0; rank < detections_count; ++rank)
                {
                    if (pr[i][rank].recall >= cur_recall) {    // > or >=
                        if (pr[i][rank].precision > cur_precision) {
                            cur_precision = pr[i][rank].precision;
                        }
                    }
                }
                //printf("class_id = %d, point = %d, cur_recall = %.4f, cur_precision = %.4f \n", i, point, cur_recall, cur_precision);

                avg_precision += cur_precision;
            }
            avg_precision = avg_precision / map_points;
        }

        printf("class_id = %d, name = %s, ap = %2.2f%%   \t (TP = %d, FP = %d) \n",
            i, names[i], avg_precision * 100, tp_for_thresh_per_class[i], fp_for_thresh_per_class[i]);

        float class_precision = (float)tp_for_thresh_per_class[i] / ((float)tp_for_thresh_per_class[i] + (float)fp_for_thresh_per_class[i]);
        float class_recall = (float)tp_for_thresh_per_class[i] / ((float)tp_for_thresh_per_class[i] + (float)(truth_classes_count[i] - tp_for_thresh_per_class[i]));
        //printf("Precision = %1.2f, Recall = %1.2f, avg IOU = %2.2f%% \n\n", class_precision, class_recall, avg_iou_per_class[i]);

        mean_average_precision += avg_precision;
    }

    const float cur_precision = (float)tp_for_thresh / ((float)tp_for_thresh + (float)fp_for_thresh);
    const float cur_recall = (float)tp_for_thresh / ((float)tp_for_thresh + (float)(unique_truth_count - tp_for_thresh));
    const float f1_score = 2.F * cur_precision * cur_recall / (cur_precision + cur_recall);
    printf("\n for conf_thresh = %1.2f, precision = %1.2f, recall = %1.2f, F1-score = %1.2f \n",
        thresh_calc_avg_iou, cur_precision, cur_recall, f1_score);

    printf(" for conf_thresh = %0.2f, TP = %d, FP = %d, FN = %d, average IoU = %2.2f %% \n",
        thresh_calc_avg_iou, tp_for_thresh, fp_for_thresh, unique_truth_count - tp_for_thresh, avg_iou * 100);

    mean_average_precision = mean_average_precision / classes;
    printf("\n IoU threshold = %2.0f %%, ", iou_thresh * 100);
    if (map_points) printf("used %d Recall-points \n", map_points);
    else printf("used Area-Under-Curve for each unique Recall \n");

    printf(" mean average precision (mAP@%0.2f) = %f, or %2.2f %% \n", iou_thresh, mean_average_precision, mean_average_precision * 100);

    for (i = 0; i < classes; ++i) {
        free(pr[i]);
    }
    free(pr);
    free(detections);
    free(truth_classes_count);
    free(detection_per_class_count);

    free(avg_iou_per_class);
    free(tp_for_thresh_per_class);
    free(fp_for_thresh_per_class);

    fprintf(stderr, "Total Detection Time: %d Seconds\n", (int)(time(0) - start));
    printf("\nSet -points flag:\n");
    printf(" `-points 101` for MS COCO \n");
    printf(" `-points 11` for PascalVOC 2007 (uncomment `difficult` in voc.data) \n");
    printf(" `-points 0` (AUC) for ImageNet, PascalVOC 2010-2012, your custom dataset\n");
    if (reinforcement_fd != NULL) fclose(reinforcement_fd);

    // free memory
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    if (existing_net) {
        //set_batch_network(&net, initial_batch);
        //free_network_recurrent_state(*existing_net);
        restore_network_recurrent_state(*existing_net);
        //randomize_network_recurrent_state(*existing_net);
    }
    else {
        free_network(net);
    }
    if (val) free(val);
    if (val_resized) free(val_resized);
    if (thr) free(thr);
    if (buf) free(buf);
    if (buf_resized) free(buf_resized);

    return mean_average_precision;
}

typedef struct {
    float w, h;
} anchors_t;

int anchors_comparator(const void *pa, const void *pb)
{
    anchors_t a = *(const anchors_t *)pa;
    anchors_t b = *(const anchors_t *)pb;
    float diff = b.w*b.h - a.w*a.h;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

int anchors_data_comparator(const float **pa, const float **pb)
{
    float *a = (float *)*pa;
    float *b = (float *)*pb;
    float diff = b[0] * b[1] - a[0] * a[1];
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}


void calc_anchors(char *datacfg, int num_of_clusters, int width, int height, int show)
{
    printf("\n num_of_clusters = %d, width = %d, height = %d \n", num_of_clusters, width, height);
    if (width < 0 || height < 0) {
        printf("Usage: darknet detector calc_anchors data/voc.data -num_of_clusters 9 -width 416 -height 416 \n");
        printf("Error: set width and height \n");
        return;
    }

    //float pointsdata[] = { 1,1, 2,2, 6,6, 5,5, 10,10 };
    float* rel_width_height_array = (float*)xcalloc(1000, sizeof(float));


    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    list *plist = get_paths(train_images);
    int number_of_images = plist->size;
    char **paths = (char **)list_to_array(plist);

    int classes = option_find_int(options, "classes", 1);
    int* counter_per_class = (int*)xcalloc(classes, sizeof(int));

    srand(time(0));
    int number_of_boxes = 0;
    printf(" read labels from %d images \n", number_of_images);

    int i, j;
    for (i = 0; i < number_of_images; ++i) {
        char *path = paths[i];
        char labelpath[4096];
        replace_image_to_label(path, labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        //printf(" new path: %s \n", labelpath);
        char *buff = (char*)xcalloc(6144, sizeof(char));
        for (j = 0; j < num_labels; ++j)
        {
            if (truth[j].x > 1 || truth[j].x <= 0 || truth[j].y > 1 || truth[j].y <= 0 ||
                truth[j].w > 1 || truth[j].w <= 0 || truth[j].h > 1 || truth[j].h <= 0)
            {
                printf("\n\nWrong label: %s - j = %d, x = %f, y = %f, width = %f, height = %f \n",
                    labelpath, j, truth[j].x, truth[j].y, truth[j].w, truth[j].h);
                sprintf(buff, "echo \"Wrong label: %s - j = %d, x = %f, y = %f, width = %f, height = %f\" >> bad_label.list",
                    labelpath, j, truth[j].x, truth[j].y, truth[j].w, truth[j].h);
                system(buff);
                if (check_mistakes) getchar();
            }
            if (truth[j].id >= classes) {
                classes = truth[j].id + 1;
                counter_per_class = (int*)xrealloc(counter_per_class, classes * sizeof(int));
            }
            counter_per_class[truth[j].id]++;

            number_of_boxes++;
            rel_width_height_array = (float*)xrealloc(rel_width_height_array, 2 * number_of_boxes * sizeof(float));

            rel_width_height_array[number_of_boxes * 2 - 2] = truth[j].w * width;
            rel_width_height_array[number_of_boxes * 2 - 1] = truth[j].h * height;
            printf("\r loaded \t image: %d \t box: %d", i + 1, number_of_boxes);
        }
        free(buff);
    }
    printf("\n all loaded. \n");
    printf("\n calculating k-means++ ...");

    matrix boxes_data;
    model anchors_data;
    boxes_data = make_matrix(number_of_boxes, 2);

    printf("\n");
    for (i = 0; i < number_of_boxes; ++i) {
        boxes_data.vals[i][0] = rel_width_height_array[i * 2];
        boxes_data.vals[i][1] = rel_width_height_array[i * 2 + 1];
        //if (w > 410 || h > 410) printf("i:%d,  w = %f, h = %f \n", i, w, h);
    }

    // Is used: distance(box, centroid) = 1 - IoU(box, centroid)

    // K-means
    anchors_data = do_kmeans(boxes_data, num_of_clusters);

    qsort((void*)anchors_data.centers.vals, num_of_clusters, 2 * sizeof(float), (__compar_fn_t)anchors_data_comparator);

    //gen_anchors.py = 1.19, 1.99, 2.79, 4.60, 4.53, 8.92, 8.06, 5.29, 10.32, 10.66
    //float orig_anch[] = { 1.19, 1.99, 2.79, 4.60, 4.53, 8.92, 8.06, 5.29, 10.32, 10.66 };

    printf("\n");
    float avg_iou = 0;
    for (i = 0; i < number_of_boxes; ++i) {
        float box_w = rel_width_height_array[i * 2]; //points->data.fl[i * 2];
        float box_h = rel_width_height_array[i * 2 + 1]; //points->data.fl[i * 2 + 1];
                                                         //int cluster_idx = labels->data.i[i];
        int cluster_idx = 0;
        float min_dist = FLT_MAX;
        float best_iou = 0;
        for (j = 0; j < num_of_clusters; ++j) {
            float anchor_w = anchors_data.centers.vals[j][0];   // centers->data.fl[j * 2];
            float anchor_h = anchors_data.centers.vals[j][1];   // centers->data.fl[j * 2 + 1];
            float min_w = (box_w < anchor_w) ? box_w : anchor_w;
            float min_h = (box_h < anchor_h) ? box_h : anchor_h;
            float box_intersect = min_w*min_h;
            float box_union = box_w*box_h + anchor_w*anchor_h - box_intersect;
            float iou = box_intersect / box_union;
            float distance = 1 - iou;
            if (distance < min_dist) {
              min_dist = distance;
              cluster_idx = j;
              best_iou = iou;
            }
        }

        float anchor_w = anchors_data.centers.vals[cluster_idx][0]; //centers->data.fl[cluster_idx * 2];
        float anchor_h = anchors_data.centers.vals[cluster_idx][1]; //centers->data.fl[cluster_idx * 2 + 1];
        if (best_iou > 1 || best_iou < 0) { // || box_w > width || box_h > height) {
            printf(" Wrong label: i = %d, box_w = %f, box_h = %f, anchor_w = %f, anchor_h = %f, iou = %f \n",
                i, box_w, box_h, anchor_w, anchor_h, best_iou);
        }
        else avg_iou += best_iou;
    }

    char buff[1024];
    FILE* fwc = fopen("counters_per_class.txt", "wb");
    if (fwc) {
        sprintf(buff, "counters_per_class = ");
        printf("\n%s", buff);
        fwrite(buff, sizeof(char), strlen(buff), fwc);
        for (i = 0; i < classes; ++i) {
            sprintf(buff, "%d", counter_per_class[i]);
            printf("%s", buff);
            fwrite(buff, sizeof(char), strlen(buff), fwc);
            if (i < classes - 1) {
                fwrite(", ", sizeof(char), 2, fwc);
                printf(", ");
            }
        }
        printf("\n");
        fclose(fwc);
    }
    else {
        printf(" Error: file counters_per_class.txt can't be open \n");
    }

    avg_iou = 100 * avg_iou / number_of_boxes;
    printf("\n avg IoU = %2.2f %% \n", avg_iou);


    FILE* fw = fopen("anchors.txt", "wb");
    if (fw) {
        printf("\nSaving anchors to the file: anchors.txt \n");
        printf("anchors = ");
        for (i = 0; i < num_of_clusters; ++i) {
            float anchor_w = anchors_data.centers.vals[i][0]; //centers->data.fl[i * 2];
            float anchor_h = anchors_data.centers.vals[i][1]; //centers->data.fl[i * 2 + 1];
            if (width > 32) sprintf(buff, "%3.0f,%3.0f", anchor_w, anchor_h);
            else sprintf(buff, "%2.4f,%2.4f", anchor_w, anchor_h);
            printf("%s", buff);
            fwrite(buff, sizeof(char), strlen(buff), fw);
            if (i + 1 < num_of_clusters) {
                fwrite(", ", sizeof(char), 2, fw);
                printf(", ");
            }
        }
        printf("\n");
        fclose(fw);
    }
    else {
        printf(" Error: file anchors.txt can't be open \n");
    }

    if (show) {
#ifdef OPENCV
        show_acnhors(number_of_boxes, num_of_clusters, rel_width_height_array, anchors_data, width, height);
#endif // OPENCV
    }
    free(rel_width_height_array);
    free(counter_per_class);

    getchar();
}


void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    net.benchmark_layers = benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        if (net.layers[net.n - 1].classes > names_size) getchar();
    }
    srand(2222222);
    char buff[256];
    char *input = buff;
    char *json_buf = NULL;
    int json_image_id = 0;
    FILE* json_file = NULL;
    if (outfile) {
        json_file = fopen(outfile, "wb");
        if(!json_file) {
          error("fopen failed");
        }
        char *tmp = "[\n";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
    }
    int j;
    float nms = .45;    // 0.4F
    while (1) {
        if (filename) {
            strncpy(input, filename, 256);
            if (strlen(input) > 0)
                if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
        }
        else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input) break;
            strtok(input, "\n");
        }
        //image im;
        //image sized = load_image_resize(input, net.w, net.h, net.c, &im);
        image im = load_image(input, 0, 0, net.c);
        image sized;
        if(letter_box) sized = letterbox_image(im, net.w, net.h);
        else sized = resize_image(im, net.w, net.h);
        layer l = net.layers[net.n - 1];

        //box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        //float **probs = calloc(l.w*l.h*l.n, sizeof(float*));
        //for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)xcalloc(l.classes, sizeof(float));

        float *X = sized.data;

        //time= what_time_is_it_now();
        double time = get_time_point();
        network_predict(net, X);
        //network_predict_image(&net, im); letterbox = 1;
        printf("%s: Predicted in %lf milli-seconds.\n", input, ((double)get_time_point() - time) / 1000);
        //printf("%s: Predicted in %f seconds.\n", input, (what_time_is_it_now()-time));

        int nboxes = 0;
        detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letter_box);
        if (nms) {
            if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
            else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
        }
        draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes, ext_output);
        save_image(im, "predictions");
        if (!dont_show) {
            show_image(im, "predictions");
        }

        if (json_file) {
            if (json_buf) {
                char *tmp = ", \n";
                fwrite(tmp, sizeof(char), strlen(tmp), json_file);
            }
            ++json_image_id;
            json_buf = detection_to_json(dets, nboxes, l.classes, names, json_image_id, input);

            fwrite(json_buf, sizeof(char), strlen(json_buf), json_file);
            free(json_buf);
        }

        // pseudo labeling concept - fast.ai
        if (save_labels)
        {
            char labelpath[4096];
            replace_image_to_label(input, labelpath);

            FILE* fw = fopen(labelpath, "wb");
            int i;
            for (i = 0; i < nboxes; ++i) {
                char buff[1024];
                int class_id = -1;
                float prob = 0;
                for (j = 0; j < l.classes; ++j) {
                    if (dets[i].prob[j] > thresh && dets[i].prob[j] > prob) {
                        prob = dets[i].prob[j];
                        class_id = j;
                    }
                }
                if (class_id >= 0) {
                    sprintf(buff, "%d %2.4f %2.4f %2.4f %2.4f\n", class_id, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);
                    fwrite(buff, sizeof(char), strlen(buff), fw);
                }
            }
            fclose(fw);
        }

        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);

        if (!dont_show) {
            wait_until_press_key_cv();
            destroy_all_windows_cv();
        }

        if (filename) break;
    }

    if (json_file) {
        char *tmp = "\n]";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
        fclose(json_file);
    }

    // free memory
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    int i;
    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);

    free_network(net);
}

#if defined(OPENCV) && defined(GPU)

// adversarial attack dnn
void draw_object(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, int dont_show, int it_num,
    int letter_box, int benchmark_layers)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);// parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
    net.adversarial = 1;
    set_batch_network(&net, 1);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    net.benchmark_layers = benchmark_layers;
    //fuse_conv_batchnorm(net);
    //calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        if (net.layers[net.n - 1].classes > names_size) getchar();
    }

    srand(2222222);
    char buff[256];
    char *input = buff;

    int j;
    float nms = .45;    // 0.4F
    while (1) {
        if (filename) {
            strncpy(input, filename, 256);
            if (strlen(input) > 0)
                if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
        }
        else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input) break;
            strtok(input, "\n");
        }
        //image im;
        //image sized = load_image_resize(input, net.w, net.h, net.c, &im);
        image im = load_image(input, 0, 0, net.c);
        image sized;
        if (letter_box) sized = letterbox_image(im, net.w, net.h);
        else sized = resize_image(im, net.w, net.h);

        image src_sized = copy_image(sized);

        layer l = net.layers[net.n - 1];
        net.num_boxes = l.max_boxes;
        int num_truth = l.truths;
        float *truth_cpu = (float *)xcalloc(num_truth, sizeof(float));

        int *it_num_set = (int *)xcalloc(1, sizeof(int));
        float *lr_set = (float *)xcalloc(1, sizeof(float));
        int *boxonly = (int *)xcalloc(1, sizeof(int));

        cv_draw_object(sized, truth_cpu, net.num_boxes, num_truth, it_num_set, lr_set, boxonly, l.classes, names);

        net.learning_rate = *lr_set;
        it_num = *it_num_set;

        float *X = sized.data;

        mat_cv* img = NULL;
        float max_img_loss = 5;
        int number_of_lines = 100;
        int img_size = 1000;
        char windows_name[100];
        char *base = basecfg(cfgfile);
        sprintf(windows_name, "chart_%s.png", base);
        img = draw_train_chart(windows_name, max_img_loss, it_num, number_of_lines, img_size, dont_show, NULL);

        int iteration;
        for (iteration = 0; iteration < it_num; ++iteration)
        {
            forward_backward_network_gpu(net, X, truth_cpu);

            float avg_loss = get_network_cost(net);
            draw_train_loss(windows_name, img, img_size, avg_loss, max_img_loss, iteration, it_num, 0, 0, "mAP%", dont_show, 0, 0);

            float inv_loss = 1.0 / max_val_cmp(0.01, avg_loss);
            //net.learning_rate = *lr_set * inv_loss;

            if (*boxonly) {
                int dw = truth_cpu[2] * sized.w, dh = truth_cpu[3] * sized.h;
                int dx = truth_cpu[0] * sized.w - dw / 2, dy = truth_cpu[1] * sized.h - dh / 2;
                image crop = crop_image(sized, dx, dy, dw, dh);
                copy_image_inplace(src_sized, sized);
                embed_image(crop, sized, dx, dy);
            }

            show_image_cv(sized, "image_optimization");
            wait_key_cv(20);
        }

        net.train = 0;
        quantize_image(sized);
        network_predict(net, X);

        save_image_png(sized, "drawn");
        //sized = load_image("drawn.png", 0, 0, net.c);

        int nboxes = 0;
        detection *dets = get_network_boxes(&net, sized.w, sized.h, thresh, 0, 0, 1, &nboxes, letter_box);
        if (nms) {
            if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
            else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
        }
        draw_detections_v3(sized, dets, nboxes, thresh, names, alphabet, l.classes, 1);
        save_image(sized, "pre_predictions");
        if (!dont_show) {
            show_image(sized, "pre_predictions");
        }

        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);
        free_image(src_sized);

        if (!dont_show) {
            wait_until_press_key_cv();
            destroy_all_windows_cv();
        }

        free(lr_set);
        free(it_num_set);

        if (filename) break;
    }

    // free memory
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    int i;
    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);

    free_network(net);
}
#else // defined(OPENCV) && defined(GPU)
void draw_object(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, int dont_show, int it_num,
    int letter_box, int benchmark_layers)
{
    printf(" ./darknet detector draw ... can't be used without OpenCV and CUDA! \n");
    getchar();
}
#endif // defined(OPENCV) && defined(GPU)

void run_detector(int argc, char **argv)
{
    int dont_show = find_arg(argc, argv, "-dont_show");
    int benchmark = find_arg(argc, argv, "-benchmark");
    int benchmark_layers = find_arg(argc, argv, "-benchmark_layers");
    //if (benchmark_layers) benchmark = 1;
    if (benchmark) dont_show = 1;
    int show = find_arg(argc, argv, "-show");
    int letter_box = find_arg(argc, argv, "-letter_box");
    int calc_map = find_arg(argc, argv, "-map");
    int map_points = find_int_arg(argc, argv, "-points", 0);
    check_mistakes = find_arg(argc, argv, "-check_mistakes");
    int show_imgs = find_arg(argc, argv, "-show_imgs");
    int mjpeg_port = find_int_arg(argc, argv, "-mjpeg_port", -1);
    int avgframes = find_int_arg(argc, argv, "-avgframes", 3);
    int dontdraw_bbox = find_arg(argc, argv, "-dontdraw_bbox");
    int json_port = find_int_arg(argc, argv, "-json_port", -1);
    char *http_post_host = find_char_arg(argc, argv, "-http_post_host", 0);
    int time_limit_sec = find_int_arg(argc, argv, "-time_limit_sec", 0);
    char *out_filename = find_char_arg(argc, argv, "-out_filename", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .25);    // 0.24
    float iou_thresh = find_float_arg(argc, argv, "-iou_thresh", .5);    // 0.5 for mAP
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int num_of_clusters = find_int_arg(argc, argv, "-num_of_clusters", 5);
    int width = find_int_arg(argc, argv, "-width", -1);
    int height = find_int_arg(argc, argv, "-height", -1);
    // extended output in test mode (output of rect bound coords)
    // and for recall mode (extended output table-like format with results for best_class fit)
    int ext_output = find_arg(argc, argv, "-ext_output");
    int save_labels = find_arg(argc, argv, "-save_labels");
    char* chart_path = find_char_arg(argc, argv, "-chart", 0);
    if (argc < 4) {
        fprintf(stderr, "usage: %s %s [train/test/valid/demo/map] [data] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if (gpu_list) {
        printf("%s\n", gpu_list);
        int len = (int)strlen(gpu_list);
        ngpus = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = (int*)xcalloc(ngpus, sizeof(int));
        for (i = 0; i < ngpus; ++i) {
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',') + 1;
        }
    }
    else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    if (weights)
        if (strlen(weights) > 0)
            if (weights[strlen(weights) - 1] == 0x0d) weights[strlen(weights) - 1] = 0;
    char *filename = (argc > 6) ? argv[6] : 0;
    if (0 == strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, dont_show, ext_output, save_labels, outfile, letter_box, benchmark_layers);
    else if (0 == strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear, dont_show, calc_map, mjpeg_port, show_imgs, benchmark_layers, chart_path);
    else if (0 == strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
    else if (0 == strcmp(argv[2], "recall")) validate_detector_recall(datacfg, cfg, weights);
    else if (0 == strcmp(argv[2], "map")) validate_detector_map(datacfg, cfg, weights, thresh, iou_thresh, map_points, letter_box, NULL);
    else if (0 == strcmp(argv[2], "calc_anchors")) calc_anchors(datacfg, num_of_clusters, width, height, show);
    else if (0 == strcmp(argv[2], "draw")) {
        int it_num = 100;
        draw_object(datacfg, cfg, weights, filename, thresh, dont_show, it_num, letter_box, benchmark_layers);
    }
    else if (0 == strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        if (filename)
            if (strlen(filename) > 0)
                if (filename[strlen(filename) - 1] == 0x0d) filename[strlen(filename) - 1] = 0;
        demo(cfg, weights, thresh, hier_thresh, cam_index, filename, names, classes, avgframes, frame_skip, prefix, out_filename,
            mjpeg_port, dontdraw_bbox, json_port, dont_show, ext_output, letter_box, time_limit_sec, http_post_host, benchmark, benchmark_layers);

        free_list_contents_kvp(options);
        free_list(options);
    }
    else printf(" There isn't such command: %s", argv[2]);

    if (gpus && gpu_list && ngpus > 1) free(gpus);
}
