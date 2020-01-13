#include <stdio.h>

#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"

char *coco_classes[] = {"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

void train_coco(char *cfgfile, char *weightfile)
{
    //char *train_images = "/home/pjreddie/data/voc/test/train.txt";
    //char *train_images = "/home/pjreddie/data/coco/train.txt";
    char *train_images = "data/coco.trainval.txt";
    //char *train_images = "data/bags.train.list";
    char* backup_directory = "backup/";
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;
    data train, buffer;


    layer l = net.layers[net.n - 1];

    int side = l.side;
    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA;

    args.angle = net.angle;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        /*
           image im = float_to_image(net.w, net.h, 3, train.X.vals[113]);
           image copy = copy_image(im);
           draw_coco(copy, train.y.vals[113], 7, "truth");
           cvWaitKey(0);
           free_image(copy);
         */

        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        if(i%100==0){
            char buff[256];
            sprintf(buff, "%s/%s.backup", backup_directory, base);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void print_cocos(FILE *fp, int image_id, box *boxes, float **probs, int num_boxes, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < num_boxes; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, probs[i][j]);
        }
    }
}

int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '_');
    return atoi(p+1);
}

void validate_coco(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/";
    list *plist = get_paths("data/coco_val_5k.list");
    //list *plist = get_paths("/home/pjreddie/data/people-art/test.txt");
    //list *plist = get_paths("/home/pjreddie/data/voc/test/2007_test.txt");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int side = l.side;

    int j;
    char buff[1024];
    snprintf(buff, 1024, "%s/coco_results.json", base);
    FILE *fp = fopen(buff, "w");
    fprintf(fp, "[\n");

    box* boxes = (box*)xcalloc(side * side * l.n, sizeof(box));
    float** probs = (float**)xcalloc(side * side * l.n, sizeof(float*));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = (float*)xcalloc(classes, sizeof(float));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .01;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 8;
    image* val = (image*)xcalloc(nthreads, sizeof(image));
    image* val_resized = (image*)xcalloc(nthreads, sizeof(image));
    image* buf = (image*)xcalloc(nthreads, sizeof(image));
    image* buf_resized = (image*)xcalloc(nthreads, sizeof(image));
    pthread_t* thr = (pthread_t*)xcalloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
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
            int image_id = get_coco_image_id(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            get_detection_boxes(l, w, h, thresh, probs, boxes, 0);
            if (nms) do_nms_sort_v2(boxes, probs, side*side*l.n, classes, iou_thresh);
            print_cocos(fp, image_id, boxes, probs, side*side*l.n, classes, w, h);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
#ifdef WIN32
    fseek(fp, -3, SEEK_CUR);
#else
    fseek(fp, -2, SEEK_CUR);
#endif
    fprintf(fp, "\n]\n");
    fclose(fp);

    if (val) free(val);
    if (val_resized) free(val_resized);
    if (buf) free(buf);
    if (buf_resized) free(buf_resized);
    if (thr) free(thr);

    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void validate_coco_recall(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    list* plist = get_paths("data/voc/test/2007_test.txt");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int side = l.side;

    int j, k;
    /* unused code,why?
    FILE** fps = (FILE**)xcalloc(classes, sizeof(FILE*));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, coco_classes[j]);
        fps[j] = fopen(buff, "w");
    }
    */
    box* boxes = (box*)xcalloc(side * side * l.n, sizeof(box));
    float** probs = (float**)xcalloc(side * side * l.n, sizeof(float*));
    for(j = 0; j < side*side*l.n; ++j) {
      probs[j] = (float*)xcalloc(classes, sizeof(float));
    }

    int m = plist->size;
    int i=0;

    float thresh = .001;
    int nms = 0;
    float iou_thresh = .5;
    float nms_thresh = .5;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        get_detection_boxes(l, 1, 1, thresh, probs, boxes, 1);
        if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms_thresh);

        char labelpath[4096];
		replace_image_to_label(path, labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < side*side*l.n; ++k){
            if(probs[k][0] > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < side*side*l.n; ++k){
                float iou = box_iou(boxes[k], t);
                if(probs[k][0] > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);

        //if (fps) free(fps);
        if (id) free(id);
        free_image(orig);
        free_image(sized);
    }
    free(boxes);
    for(j = 0; j < side*side*l.n; ++j) {
        free(probs[j]);
    }
    free(probs);
}

void test_coco(char *cfgfile, char *weightfile, char *filename, float thresh)
{
    image **alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);
    float nms = .4;
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    box* boxes = (box*)xcalloc(l.side * l.side * l.n, sizeof(box));
    float** probs = (float**)xcalloc(l.side * l.side * l.n, sizeof(float*));
    for(j = 0; j < l.side*l.side*l.n; ++j) {
      probs[j] = (float*)xcalloc(l.classes, sizeof(float));
    }
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) break;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        get_detection_boxes(l, 1, 1, thresh, probs, boxes, 0);
        if (nms) do_nms_sort_v2(boxes, probs, l.side*l.side*l.n, l.classes, nms);
        draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, coco_classes, alphabet, 80);
        save_image(im, "prediction");
        show_image(im, "predictions");
        free_image(im);
        free_image(sized);

        wait_until_press_key_cv();
        destroy_all_windows_cv();

        if (filename) break;
    }
    free(boxes);
    for(j = 0; j < l.side*l.side*l.n; ++j) {
        free(probs[j]);
    }
    free(probs);
}

void run_coco(int argc, char **argv)
{
	int dont_show = find_arg(argc, argv, "-dont_show");
	int mjpeg_port = find_int_arg(argc, argv, "-mjpeg_port", -1);
    int json_port = find_int_arg(argc, argv, "-json_port", -1);
	char *out_filename = find_char_arg(argc, argv, "-out_filename", 0);
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .2);
	float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
	int ext_output = find_arg(argc, argv, "-ext_output");

    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) test_coco(cfg, weights, filename, thresh);
    else if(0==strcmp(argv[2], "train")) train_coco(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_coco(cfg, weights);
    else if(0==strcmp(argv[2], "recall")) validate_coco_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, hier_thresh, cam_index, filename, coco_classes, 80, frame_skip,
		prefix, out_filename, mjpeg_port, json_port, dont_show, ext_output, 0, 0, 0, 0, 0);
}
