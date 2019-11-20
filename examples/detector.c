#include "darknet.h"
#include <unistd.h>
#include <termio.h> 
#include <fcntl.h>
#include <curl/curl.h>
#include <time.h>
#include <dirent.h>


struct people{
    int camera;
    int count;
};

int kbhit(void)
{
  struct termios oldt, newt;
  int ch;
  int oldf;

  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

  ch = getchar();

  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, oldf);

  if(ch != EOF)
  {
    ungetc(ch, stdin);
    return 1;
  }

  return 0;
}

int getch(void) 
{ 
    int ch;
    struct termios old;
    struct termios new; 
    tcgetattr(0, &old); 
    new = old; 
    new.c_lflag &= ~(ICANON|ECHO); 
    new.c_cc[VMIN] = 1; 
    new.c_cc[VTIME] = 0; 
    tcsetattr(0, TCSAFLUSH, &new); 
    ch = getchar(); 
    tcsetattr(0, TCSAFLUSH, &old); return ch; 
}

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};
int cando = 0;

//train_detector-1 start
void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)// ??��?��?��??��?��?�� ??��?��?��??��?��?��
{ // we must add validation data set training while train the training data set.
    list *options = read_data_cfg(datacfg);// read_data_cfg() function can find in option_list.c
    char *train_images = option_find_str(options, "train", "data/train.list"); // default == data/train.list
    char *backup_directory = option_find_str(options, "backup", "/backup/"); // default == /backup/

    srand(time(0));
    char *base = basecfg(cfgfile); // obj.cfg 
    //printf("%s\n", base);
    float avg_loss = -1;
    network **nets = calloc(ngpus, sizeof(network)); // network struct 
    // network??��?��?��湲곗?�� ?��????��?��?��?��?? ngpus?��?? ?????��?��?��??��?��?�� ??��?��?�� ??��?��?��??��?��?�� ??�듦�? ??��?��?��??��?��?�� 
    //train_detector-1 end
    srand(time(0));
    int seed = rand();
    int i;
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);// each layers information add in nets[] index.
        nets[i]->learning_rate *= ngpus; 
    }
    /*
        load_network() function -> parse_network_cfg() function -> read_cfg() function ->
        fgetl() function = get line in file  / strip() function = if line have a ' ' or '\t' or '\n' change to '\0'
        -> meet '[] list_insert() function[list.c] ->  make_network() -> parse_net_option() function ->
        -> insert information in the struct -> each parse_networklayer 
    */
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus; // training image numbers
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    data train, buffer;

    layer l = net->layers[net->n - 1]; // net -> n = layer's total number
    int classes = l.classes; // our cfg file's classes is 1
    float jitter = l.jitter; // our cfg file's jitter is 0.3

    //load_args 초기?�� ?��?��
    list *plist = get_paths(train_images); // ?��?�� ?��?���? �??��?���? list?�� �? node�? ????��?��?��
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist); // load_args?�� paths�? 2차원 ?��?��?��?���? ?��문에 list -> char**�? �??��

    load_args args = get_base_args(net);
    
    args.coords = l.coords;
    args.paths = paths; 
    args.n = imgs; // args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = l.max_boxes;
    args.d = &buffer; // net's d
    args.type = DETECTION_DATA;
    //args.type = INSTANCE_DATA;
    args.threads = 64;
    // train_detecot-2 start
    pthread_t load_thread = load_data(args); // load_data()'s return type is pthread_t
    /*load_data() -> load_threads() -> load_data_in_thread() -> load_thread() ->
    * -> { load_data_detection() -> get_random_paths() take a random path -> make_matrix() [matrix.c] ->
    * -> load_image_color() [image.c] -> load_image() = take a image -> load_image_cv() [image_opencv.cpp] this function use imread() and call mat_to_image() function  if can't read file add to the bad.list->
    * -> mat_to_image() -> ipl_to_image() image data is ( im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.; ) i < h = height , k < c = nChannels , j < w = width
    * -> rgbgr_image() [image.c] swap data[i] , data[i+w*h*2] end load_image  ->  place_image() there use 3 for iteration -> bilinear_interpolate() -> 
    *  val = (1-dy) * (1-dx) * get_pixel_extend(im, ix, iy, c) + 
        dy     * (1-dx) * get_pixel_extend(im, ix, iy+1, c) + 
        (1-dy) *   dx   * get_pixel_extend(im, ix+1, iy, c) +
        dy     *   dx   * get_pixel_extend(im, ix+1, iy+1, c);
        there are in the bilinear_interpolate() function what is it means?? 
        -> set_pixel() [image.c] ->  random_distort_image() [image.c] -> distort_image() -> fill_truth_detection() -> return data d end load_data_detection() fucntion }
        load_args a = 留ㅺ컻蹂???��?뱾�??? 二쇱?��?��? ??�듭????�?�?? ?븣臾몄뿉 寃곌?���?? ??��??���?? �?? 泥섏?��?�? �?? 留ㅺ컻蹂???��??�� args?�? ?????��

    */
    // train_detector-2 end

    double time;
    int count = 0;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < 100000){
    //while(get_current_batch(net) < net->max_batches){ // net_max_batches = 500200
	printf("get_current_batch : %ld , net->max_batches : %d\n",get_current_batch(net),net->max_batches);
        if(l.random && count++%10 == 0){
            printf("Resizing\n");
            int dim = (rand() % 10 + 10) * 32; // ((0 ~ 9)+10)*32 ==> 320 ~ 608 resize
            if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim); // current size print
            args.w = dim; // net's width = dim
            args.h = dim; // net's height = dim
			  // dix * dim resolution
            pthread_join(load_thread, 0); // waiting load_thread
            train = buffer; // what is a buffer ? data = train ( train_network() )
            free_data(train);
            load_thread = load_data(args); // ngpus�? 1개�?? ?��?�� 경우?�� ?��?��?��  thread

            #pragma omp parallel for
            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim); // network resize 
            } // resize_network() [network.c] -> resize each type layer
            net = nets[0];
        }
        time=what_time_is_it_now();
        pthread_join(load_thread, 0); // waiting load_thread
        train = buffer;
        load_thread = load_data(args);// ?��?�� 반복문에?�� ?��?��?�� ?��미�??�? �??��?��?�� thread
        
        /*
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[10] + 1 + k*5);
           if(!b.x) break;
           printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
           }
         */
        /*
           int zz;
           for(zz = 0; zz < train.X.cols; ++zz){
           image im = float_to_image(net->w, net->h, 3, train.X.vals[zz]);
           int k;
           for(k = 0; k < l.max_boxes; ++k){
           box b = float_to_box(train.y.vals[zz] + k*5, 1);
           printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);
           draw_bbox(im, b, 1, 1,0,0);
           }
           show_image(im, "truth11");
           cvWaitKey(0);
           save_image(im, "truth11");
           }
         */

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);

        time=what_time_is_it_now();
        float loss = 0; // this is mean the loss
#ifdef GPU
        if(ngpus == 1){ 
            loss = train_network(net, train); // call train_network() function if you can use a GPU
        } else {
            loss = train_networks(nets, ngpus, train, 4); 
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        i = get_current_batch(net);
        printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);
        if(i%100==0){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff); // save weights file 
        }
	//if(i%1000 == 0 || (i < 1000 && i%100 == 0)){
	//if(i%10000==0 || (i < 1000 && i%100 == 0)){
	if(i%10000==0 || (i%1000==0 && i <10000)){
#ifdef GPU
            if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);// current training weight print
            save_weights(net, buff); // save weights file
        }
        free_data(train);
    } // while end
#ifdef GPU
    if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff); // save final weight
}
//end train_detector

static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if(c) p = c;
    return atoi(p+1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for(i = 0; i < num_boxes; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
        }
    }
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = dets[i].bbox.x - dets[i].bbox.w/2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w/2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h/2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            int class = j;
            if (dets[i].prob[class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j+1, dets[i].prob[class],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_detector_flip(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
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

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 2);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    image input = make_image(net->w, net->h, net->c*2);

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data, 1);
            flip_image(val_resized[t]);
            copy_cpu(net->w*net->h*net->c, val_resized[t].data, 1, input.data + net->w*net->h*net->c, 1);

            network_predict(net, input.data);
            int w = val[t].w;
            int h = val[t].h;
            int num = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &num);
            if (nms) do_nms_sort(dets, num, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, num, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, num, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, num, classes, w, h);
            }
            free_detections(dets, num);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
}


void validate_detector(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{ // arg is valid
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    if(0==strcmp(type, "coco")){
        if(!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    } else if(0==strcmp(type, "imagenet")){
        if(!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    } else {
        if(!outfile) outfile = "comp4_det_test_";
        fps = calloc(classes, sizeof(FILE *));
        for(j = 0; j < classes; ++j){
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    int m = plist->size;
    int i=0;
    int t;

    float thresh = .005;
    float nms = .45;

    int nthreads = 4;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    //args.type = IMAGE_DATA;
    args.type = LETTERBOX_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    double start = what_time_is_it_now();
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(net, w, h, thresh, .5, map, 0, &nboxes);
            if (nms) do_nms_sort(dets, nboxes, classes, nms);
            if (coco){
                print_cocos(fp, path, dets, nboxes, classes, w, h);
            } else if (imagenet){
                print_imagenet_detections(fp, i+t-nthreads+1, dets, nboxes, classes, w, h);
            } else {
                print_detector_detections(fps, id, dets, nboxes, classes, w, h);
            }
            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    for(j = 0; j < classes; ++j){
        if(fps) fclose(fps[j]);
    }
    if(coco){
        fseek(fp, -2, SEEK_CUR); 
        fprintf(fp, "\n]\n");
        fclose(fp);
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", what_time_is_it_now() - start);
} // end validate_detector() function

void validate_detector_recall(char *cfgfile, char *weightfile)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    srand(time(0));

    list *plist = get_paths("data/coco_val_5k.list");
    char **paths = (char **)list_to_array(plist);

    layer l = net->layers[net->n-1];

    int j, k;

    int m = plist->size;
    int i=0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = .4;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net->w, net->h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes);
        if (nms) do_nms_obj(dets, nboxes, 1, nms);

        char labelpath[4096];
        find_replace(path, "images", "labels", labelpath);
        find_replace(labelpath, "JPEGImages", "labels", labelpath);
        find_replace(labelpath, ".jpg", ".txt", labelpath);
        find_replace(labelpath, ".JPEG", ".txt", labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < nboxes; ++k){
            if(dets[k].objectness > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < l.w*l.h*l.n; ++k){
                float iou = box_iou(dets[k].bbox, t);
                if(dets[k].objectness > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}


void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net->w, net->h);
        //image sized = resize_image(im, net->w, net->h);
        //image sized2 = resize_max(im, net->w);
        //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
        //resize_network(net, sized.w, sized.h);
        layer l = net->layers[net->n-1];


        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
        //printf("%d\n", nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
        free_detections(dets, nboxes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            make_window("predictions", 512, 512, 0);
            show_image(im, "predictions", 0);
#endif
        }

        free_image(im);
        free_image(sized);
        if (filename) break;
    }
}


/*
void censor_detector(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int class, float thresh, int skip)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    int w = 1280;
    int h = 720;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow(base, CV_WINDOW_NORMAL); 
    cvResizeWindow(base, 512, 512);
    float fps = 0;
    int i;
    float nms = .45;

    while(1){
        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);
        layer l = net->layers[net->n-1];

        float *X = in_s.data;
        network_predict(net, X);
        int nboxes = 0;
        detection *dets = get_network_boxes(net, in.w, in.h, thresh, 0, 0, 0, &nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        for(i = 0; i < nboxes; ++i){
            if(dets[i].prob[class] > thresh){
                box b = dets[i].bbox;
                int left  = b.x-b.w/2.;
                int top   = b.y-b.h/2.;
                censor_image(in, left, top, b.w, b.h);
            }
        }
        show_image(in, base);
        cvWaitKey(10);
        free_detections(dets, nboxes);


        free_image(in_s);
        free_image(in);


        float curr = 0;
        fps = .9*fps + .1*curr;
        for(i = 0; i < skip; ++i){
            image in = get_image_from_stream(cap);
            free_image(in);
        }
    }
    #endif
}

void extract_detector(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename, int class, float thresh, int skip)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    srand(2222222);
    CvCapture * cap;

    int w = 1280;
    int h = 720;

    if(filename){
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(w){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
    }
    if(h){
        cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
    }

    if(!cap) error("Couldn't connect to webcam.\n");
    cvNamedWindow(base, CV_WINDOW_NORMAL); 
    cvResizeWindow(base, 512, 512);
    float fps = 0;
    int i;
    int count = 0;
    float nms = .45;

    while(1){
        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);
        layer l = net->layers[net->n-1];

        show_image(in, base);

        int nboxes = 0;
        float *X = in_s.data;
        network_predict(net, X);
        detection *dets = get_network_boxes(net, in.w, in.h, thresh, 0, 0, 1, &nboxes);
        //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);

        for(i = 0; i < nboxes; ++i){
            if(dets[i].prob[class] > thresh){
                box b = dets[i].bbox;
                int size = b.w*in.w > b.h*in.h ? b.w*in.w : b.h*in.h;
                int dx  = b.x*in.w-size/2.;
                int dy  = b.y*in.h-size/2.;
                image bim = crop_image(in, dx, dy, size, size);
                char buff[2048];
                sprintf(buff, "results/extract/%07d", count);
                ++count;
                save_image(bim, buff);
                free_image(bim);
            }
        }
        free_detections(dets, nboxes);


        free_image(in_s);
        free_image(in);


        float curr = 0;
        fps = .9*fps + .1*curr;
        for(i = 0; i < skip; ++i){
            image in = get_image_from_stream(cap);
            free_image(in);
        }
    }
    #endif
}
*/

/*
void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets)
{
    network_predict_image(net, im);
    layer l = net->layers[net->n-1];
    int nboxes = num_boxes(net);
    fill_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 0, dets);
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
}
*/

//�߰� �Լ� ( ������ ���� ���� �� ���� �б⸦ ����ϴ� �Լ� )
void detector_run(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double times;
    char buff[256];
    char *input = buff;
    float nms=.45;
    int wait = 100;
    int j = 0;
    int i;
    int z= 0;
    Points pointArray[10];//좌표 정보 저장[] 2차원 배열을 사용하여 해결하는 방법 생각해보기
    unsigned int seqkey;
    /*기존의 경우에는 Points 구조체 배열을 10개의 크기를 만들어서
    각 인덱스마다 카메라의 배열 리스트를 가지고 있는 변수를 만들었다.
    
    수정해야되는 부분은 각 카메라 마다 다중 구역을 검출해야되기 때문에
    방법을 수정할 필요가 있다.
    1. Points 배열을 10개 만들어서 각 카메라마다 각각의 배열을 매칭시켜 ArrayList를 저장하도록 할 것
    2. Points 구조체를 배열로 가지는 구조체를 추가적으로 만들어서 이를 사용할 것

    둘 중 하나의 방법을 활용하여 해결할 것

    */
    for(i = 0 ; i < 10 ; i++)
    {
        pointArray[i].size = 0;
    }
    while(1)
    {
        if(kbhit()==1) // 키 입력 확인
        {
            int key = getch();
            switch(key){
                case '1':
                    printf("change 1 picture\n");
                    load_one_image(input, &pointArray[key-'1'],key-'0');
                    break;
                case '2':
                    printf("change 2 picture\n");
                    load_one_image(input, &pointArray[key-'1'],key-'0');
                    break;
                case '3':
                    printf("change 3 picture\n");
                    load_one_image(input, &pointArray[key-'1'],key-'0');
                    break;
                case '4':
                    printf("change 4 picture\n");
                    load_one_image(input, &pointArray[key-'1'],key-'0');
                    break;
                case '5':
                    printf("change 5 picture\n");
                    load_one_image(input, &pointArray[key-'1'],key-'0');
                    break;
                case '6':
                    printf("change 6 picture\n");
                    load_one_image(input, &pointArray[key-'1'],key-'0');
                    break;
                case '7':
                    printf("change 7 picture\n");
                    load_one_image(input, &pointArray[key-'1'],key-'0');
                    break;
                case '8':
                    printf("change 8 picture\n");
                    load_one_image(input, &pointArray[key-'1'],key-'0');
                    break;
                case '9':
                    printf("change 9 picture\n");
                    load_one_image(input, &pointArray[key-'1'],key-'0');
                    break;
                case '0':
                    printf("change 0 picture\n");
                    load_one_image(input, &pointArray[9],10);
                    break;
                case 's': case 'S':
                    printf("\ncheck Area\n");
                    for(j = 1 ; j <= 10 ; j++){
                        sprintf(input,"/home/kdy/information/TestImage/Test_%d.jpg",j);
                        //sprintf(input,"/var/lib/tomcat8/webapps/UploadServer/resources/upload/img%d%d.jpg",j/10,j%10);
                                              
                        load_mat_image_point(input,j,&pointArray[j-1]); // pointArray.size가 0일 경우 전체 화면에 대해서 검출하는 식으로 진행
                        for(i = 0 ; i < pointArray[j-1].size ; i++)
                        {
                            printf("pointArray[%d].X : %d , pointArray[%d].Y : %d\n",j,pointArray[j-1].x[i],j,pointArray[j-1].y[i]);
                        }
                    }// end for function
                    break;
                case 27 :
                    printf("Program Exit\n");
                    return;
                    break;
                }
        }//end if function
        //non error
        if(wait == 0)
        {
            image im;
            image sized;
            int count;
            for(j = 1 ; j <= 10 ; j++)
            {
                sprintf(input,"/home/kdy/information/TestImage/Test_%d.jpg",j);
                //sprintf(input,"/var/lib/tomcat8/webapps/UploadServer/resources/upload/img%d%d.jpg",j/10,j%10);
                char sudoText[512];
                sprintf(sudoText,"sudo chmod 777 %s",input);
                system(sudoText);
                im = load_image_color(input,0,0);
                sized = letterbox_image(im, net->w, net->h);

                layer l = net->layers[net->n-1];

                float *X = sized.data;
                times=what_time_is_it_now();
                if(cando == 1)
                {
                    network_predict(net, X);
                    printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-times);
                    int nboxes = 0;
                    detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
                    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
                    printf("pointArray[%d].size = %d\n",j-1,pointArray[j-1].size);
                    if(pointArray[j-1].size >= 3)
                    {
                        printf("111\n");
                        im = draw_detections_area_count(im, dets, nboxes, thresh, names, alphabet, l.classes,&pointArray[j-1],&count);
                    }
                    else
                    {
                        draw_detections_count(im, dets, nboxes, thresh, names, alphabet, l.classes,&count);
                    }
                    free_detections(dets, nboxes);
                }
                if(outfile){
                    save_image(im, outfile);
                }
                else
                {
                save_image(im, "predictions");
                /*
                #ifdef OPENCV
                    usleep(1000*100);
                    make_window("predictions", 512, 512, 0);
                    show_image(im, "predictions", 0);

                #endif
                 */
                }
                free_image(im);
                free_image(sized);
                //Get
                char url[512];
                struct people pp;
                //pp.camera = j;
                //pp.count = count;
                //sprintf(url,"http://210.115.230.164:8080/People/Update?camera=%d&count=%d",j,count);
                //sprintf(url,"curl -X POST --header 'Content-Type: application/json' --header 'Accept: application/json' -d '{\"camera\": %d,\"count\": %d}' 'http://210.115.230.164:8080/People/UpdatePost'",j,count);
                
                 /*// system()사용 
                sprintf(url,"curl -X POST --header 'Content-Type: application/x-www-form-urlencoded' -d \"fname=테스트&poi=x18&su=%d\" 'http://121.187.239.177:8080/poipeoplesu'",z++);
                if(z>=100)
                {
                    z = 0;
                }
                system(url);
                printf("%d\n",z);
                */

                sprintf(url, "http://121.187.239.177:8080/poipeoplesu");
                //sprintf(url,"http://210.115.230.164:8080/People/UpdatePost");
                char data[512];
                char poi[512];
                time_t timer;
                struct tm *t;
                char days[512];
                timer = time(NULL);
                if (j < 10)
                    sprintf(poi, "x0%d", j);
                else
                {
                    sprintf(poi, "x%d", j);
                }
                printf("poi : %s, su : %d seqkey : %d\n",poi,count,seqkey);
                sprintf(data, "fname=테스트&poi=%s&su=%d&seqkey=%d", poi, count,seqkey);
                CURL *curl;
                CURLcode res;
                struct curl_slist *list = NULL;
                curl = curl_easy_init();
                if (curl)
                {
                    curl_easy_setopt(curl, CURLOPT_URL, url);
                    list = curl_slist_append(list, "Content-Type: application/x-www-form-urlencoded");
                    //list = curl_slist_append(list, "Content-Type: application/json");
                    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);   // content-type 설정
                    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L); // 값을 false 하면 에러가 떠서 공식 문서 참고함
                    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 1L); // 값을 false 하면 에러가 떠서 공식 문서 참고함
                    curl_easy_setopt(curl, CURLOPT_POST, 1L);           //POST option
                    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data);

                    res = curl_easy_perform(curl);
                    if (res == CURLE_OK)
                        printf("POST success = %s\n", url);
                    else
                        printf("POST fault\n");

                    curl_easy_cleanup(curl);
                }
                else
                {
                    printf("CURL fault\n");
                }

            }// end for functionls
            seqkey+=1;
            wait = 300;
            //wait = 50; // 5초
            usleep(100*1000);
        }//end if function
        
        usleep(100*1000);
        wait -= 1;
        if (filename) break;
    }// end while function
    free(pointArray);
}

void detector_runs(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double times;
    char buff[256];
    char *input = buff;
    float nms=.45;
    int wait = 100;
    int j = 0;
    int i;
    int z= 0;
    NumPoints pointArrays[10];
    unsigned int seqkey;
    /*기존의 경우에는 Points 구조체 배열을 10개의 크기를 만들어서
    각 인덱스마다 카메라의 배열 리스트를 가지고 있는 변수를 만들었다.
    
    수정해야되는 부분은 각 카메라 마다 다중 구역을 검출해야되기 때문에
    방법을 수정할 필요가 있다.
    1. Points 배열을 10개 만들어서 각 카메라마다 각각의 배열을 매칭시켜 ArrayList를 저장하도록 할 것
    2. Points 구조체를 배열로 가지는 구조체를 추가적으로 만들어서 이를 사용할 것

    둘 중 하나의 방법을 활용하여 해결할 것

    */
    for(i = 0 ; i < 10 ; i++)
    {
        pointArrays[i].size = 0;
    }
    while(1)
    {
        if(kbhit()==1) // 키 입력 확인
        {
            int key = getch();
            switch(key){
                case '1':
                    printf("change 1 picture\n");
                    load_one_image_Array(input, &pointArrays[key-'1'], key-'0');
                    for(i = 0 ; i < pointArrays[key-'1'].size ; i++)
                    {
                        for(j = 0 ; j < pointArrays[key-'1'].P[i].size ; j++)
                        printf("pointArrays[0]->x[%d] = %d, pointArrays[0]->y[%d] = %d\n",i,pointArrays[key-'1'].P[i].x[j],i,pointArrays[key-'1'].P[i].y[j]);
                    }
                    break;
                case '2':
                    printf("change 2 picture\n");
                    load_one_image_Array(input, &pointArrays[key-'1'],key-'0');
                    break;
                case '3':
                    printf("change 3 picture\n");
                    load_one_image_Array(input, &pointArrays[key-'1'],key-'0');
                    break;
                case '4':
                    printf("change 4 picture\n");
                    load_one_image_Array(input, &pointArrays[key-'1'],key-'0');
                    break;
                case '5':
                    printf("change 5 picture\n");
                    load_one_image_Array(input, &pointArrays[key-'1'],key-'0');
                    break;
                case '6':
                    printf("change 6 picture\n");
                    load_one_image_Array(input, &pointArrays[key-'1'],key-'0');
                    break;
                case '7':
                    printf("change 7 picture\n");
                    load_one_image_Array(input, &pointArrays[key-'1'],key-'0');
                    break;
                case '8':
                    printf("change 8 picture\n");
                    load_one_image_Array(input, &pointArrays[key-'1'],key-'0');
                    break;
                case '9':
                    printf("change 9 picture\n");
                    load_one_image_Array(input, &pointArrays[key-'1'],key-'0');
                    break;
                case '0':
                    printf("change 0 picture\n");
                    load_one_image_Array(input, &pointArrays[9],10);
                    break;
                    
                case 27 :
                    printf("Program Exit\n");
                    return;
                    break;
                }
        }//end if function
        usleep(100*1000);
        wait -= 1;
        if (filename) break;
    }// end while function
    free(pointArrays);
}

void detector_directory(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);
    char buff[256];
    char *input = buff;
    DIR *d;
    struct dirent *dir;
    printf("Enter Directory path: ");
    fflush(stdout);
    input = fgets(input, 256, stdin);
    if(!input) return;
    strtok(input, "\n");
    d = opendir(input);
    

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    float nms=.45;
    int wait = 100;
    int j = 0;
    int i;
    int z= 0;

    if(d)
    {
        while((dir = readdir(d)) != NULL)
        {
            
            image im;
            image sized;
            char buf[512];
            int count;
            int a;
            if((a = strstr(dir->d_name,".jpg")) > 0 )
            {
            printf("%d\n",a);
            printf("%s\n",dir->d_name);
            sprintf(buf,"%s/%s",input,dir->d_name);
            printf("%s\n",buf);
            im = load_image_color(buf,0,0);
            sized = letterbox_image(im, net->w, net->h);

            layer l = net->layers[net->n-1];

            float *X = sized.data;
            time=what_time_is_it_now();
            if(cando == 1)
            {
                network_predict(net, X);
                printf("%s: Predicted in %f seconds.\n", buf, what_time_is_it_now()-time);
                int nboxes = 0;
                detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
                if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
                draw_detections_count(im, dets, nboxes, thresh, names, alphabet, l.classes,&count);
                free_detections(dets, nboxes);
            }
            if(outfile){
                save_image(im, outfile);
            }
            else
            {
            save_image(im, "predictions");

            #ifdef OPENCV
                usleep(1000*100);
                make_window("predictions", 512, 512, 0);
                show_image(im, "predictions", 0);

            #endif

            }
            free_image(im);
            free_image(sized);
            }
            else{

            }
        }//end if function
    }
    else
    {
        printf("Can't open directory\n");
    }
    closedir(d);
}

void detector_blur_directory(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);
    char buff[256];
    char *input = buff;
    DIR *d;
    struct dirent *dir;
    printf("Enter Directory path: ");
    fflush(stdout);
    input = fgets(input, 256, stdin);
    if(!input) return;
    strtok(input, "\n");
    d = opendir(input);
    

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    float nms=.45;
    int wait = 100;
    int j = 0;
    int i;
    int z= 0;

    if(d)
    {
        while((dir = readdir(d)) != NULL)
        {
            
            image im;
            image sized;
            char buf[512];
            int count;
            int a;
            if((a = strstr(dir->d_name,".jpg")) > 0 )
            {
            printf("%d\n",a);
            printf("%s\n",dir->d_name);
            sprintf(buf,"%s/%s",input,dir->d_name);
            printf("%s\n",buf);
            im = load_image_color_blur(buf,0,0);
            sized = letterbox_image(im, net->w, net->h);

            layer l = net->layers[net->n-1];

            float *X = sized.data;
            time=what_time_is_it_now();
            if(cando == 1)
            {
                network_predict(net, X);
                printf("%s: Predicted in %f seconds.\n", buf, what_time_is_it_now()-time);
                int nboxes = 0;
                detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
                if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
                draw_detections_count(im, dets, nboxes, thresh, names, alphabet, l.classes,&count);
                free_detections(dets, nboxes);
            }
            if(outfile){
                save_image(im, outfile);
            }
            else
            {
            save_image(im, "predictions");

            #ifdef OPENCV
                usleep(1000*100);
                make_window("predictions", 512, 512, 0);
                show_image(im, "predictions", 0);

            #endif

            }
            free_image(im);
            free_image(sized);
            }
            else{

            }
        }//end if function
    }
    else
    {
        printf("Can't open directory\n");
    }
    closedir(d);
}

void run_detector(int argc, char **argv) // argv[1] == detector ??��?��?�� 寃쎌?�� 
{
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .5);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int avg = find_int_arg(argc, argv, "-avg", 3);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
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
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    int width = find_int_arg(argc, argv, "-w", 0);
    int height = find_int_arg(argc, argv, "-h", 0);
    int fps = find_int_arg(argc, argv, "-fps", 0);
    //int class = find_int_arg(argc, argv, "-class", 0);

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);// ??��?��?��??��?��?��
    else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "valid2")) validate_detector_flip(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "recall")) validate_detector_recall(cfg, weights);
    else if(0==strcmp(argv[2], "directory")) detector_directory(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "directory2")) detector_blur_directory(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "demo")) { // detect
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
    }
    else if(0==strcmp(argv[2],"run"))
    {
        detector_run(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    }
    else if(0==strcmp(argv[2],"runs"))
    {
        detector_runs(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    }
    //else if(0==strcmp(argv[2], "extract")) extract_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
    //else if(0==strcmp(argv[2], "censor")) censor_detector(datacfg, cfg, weights, cam_index, filename, class, thresh, frame_skip);
}

void load_one_image(char *input, Points* ary,int j)
{
    int i =0;        
    sprintf(input,"/home/kdy/information/TestImage/Test_%d.jpg",j);
    //sprintf(input,"/var/lib/tomcat8/webapps/UploadServer/resources/upload/img%d%d.jpg",j/10,j%10);
    char sudoText[512];
    //sprintf(sudoText,"sudo chmod 777 %s",input);
    //system(sudoText);
    //sprintf(input,"/home/kdy/information/TestImage/Test_%d.jpg",j);
    load_mat_image_point(input,j,ary); // pointArray.size가 0일 경우 전체 화면에 대해서 검출하는 식으로 진행
    for(i = 0 ; i <ary->size ; i++)
    {
        printf("pointArray[%d].X : %d , pointArray[%d].Y : %d\n",j,ary->x[i],j,ary->y[i]);
    }
}

void load_one_image_Array(char *input, NumPoints *ary,int j)
{
    int i = 0;        
    int k = 0;
    //sprintf(input,"/var/lib/tomcat8/webapps/UploadServer/resources/upload/img%d%d.jpg",j/10,j%10);
    char sudoText[512];
    //sprintf(sudoText,"sudo chmod 777 %s",input);
    //system(sudoText);
    sprintf(input,"/home/kdy/information/TestImage/Test_%d.jpg",j);
    load_mat_image_points(input,j,ary); // pointArray.size가 0일 경우 전체 화면에 대해서 검출하는 식으로 진행
    for(i = 0 ; i <ary->size ; i++)
    {
        for(k = 0 ; k < ary->P[i].size ; k++)
        printf("pointArray[%d][%d].X : %d , pointArray[%d][%d].Y : %d\n",i,k,ary->P[i].x[k],i,k,ary->P[i].y[k]);
    }
}