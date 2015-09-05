#include <stdio.h>

#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

char *coco_classes[] = {"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

void draw_coco(image im, float *pred, int side, char *label)
{
    int classes = 1;
    int elems = 4+classes;
    int j;
    int r, c;

    for(r = 0; r < side; ++r){
        for(c = 0; c < side; ++c){
            j = (r*side + c) * elems;
            int class = max_index(pred+j, classes);
            if (pred[j+class] > 0.2){
                int width = pred[j+class]*5 + 1;
                printf("%f %s\n", pred[j+class], "object"); //coco_classes[class-1]);
                float red = get_color(0,class,classes);
                float green = get_color(1,class,classes);
                float blue = get_color(2,class,classes);

                j += classes;

                box predict = {pred[j+0], pred[j+1], pred[j+2], pred[j+3]};
                predict.x = (predict.x+c)/side;
                predict.y = (predict.y+r)/side;
                
                draw_bbox(im, predict, width, red, green, blue);
            }
        }
    }
    show_image(im, label);
}

void train_coco(char *cfgfile, char *weightfile)
{
    //char *train_images = "/home/pjreddie/data/coco/train.txt";
    char *train_images = "/home/pjreddie/data/voc/test/train.txt";
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    data_seed = time(0);
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = 128;
    int i = *net.seen/imgs;
    data train, buffer;


    layer l = net.layers[net.n - 1];

    int side = l.side;
    int classes = l.classes;

    list *plist = get_paths(train_images);
    int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    while(i*imgs < N*120){
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

        printf("%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock()-time), i*imgs);
        if((i-1)*imgs <= N && i*imgs > N){
            fprintf(stderr, "First stage done\n");
            net.learning_rate *= 10;
            char buff[256];
            sprintf(buff, "%s/%s_first_stage.weights", backup_directory, base);
            save_weights(net, buff);
        }

        if((i-1)*imgs <= 80*N && i*imgs > N*80){
            fprintf(stderr, "Second stage done.\n");
            char buff[256];
            sprintf(buff, "%s/%s_second_stage.weights", backup_directory, base);
            save_weights(net, buff);
        }
        if(i%1000==0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void get_probs(float *predictions, int total, int classes, int inc, float **probs)
{
    int i,j;
    for (i = 0; i < total; ++i){
        int index = i*inc;
        float scale = predictions[index];
        probs[i][0] = scale;
        for(j = 0; j < classes; ++j){
            probs[i][j] = scale*predictions[index+j+1];
        }
    }
}
void get_boxes(float *predictions, int n, int num_boxes, int per_box, box *boxes)
{
    int i,j;
    for (i = 0; i < num_boxes*num_boxes; ++i){
        for(j = 0; j < n; ++j){
            int index = i*n+j;
            int offset = index*per_box;
            int row = i / num_boxes;
            int col = i % num_boxes;
            boxes[index].x = (predictions[offset + 0] + col) / num_boxes;
            boxes[index].y = (predictions[offset + 1] + row) / num_boxes;
            boxes[index].w = predictions[offset + 2];
            boxes[index].h = predictions[offset + 3];
        }
    }
}

void convert_cocos(float *predictions, int classes, int num_boxes, int num, int w, int h, float thresh, float **probs, box *boxes)
{
    int i,j;
    int per_box = 4+classes;
    for (i = 0; i < num_boxes*num_boxes*num; ++i){
        int offset = i*per_box;
        for(j = 0; j < classes; ++j){
            float prob = predictions[offset+j];
            probs[i][j] = (prob > thresh) ? prob : 0;
        }
        int row = i / num_boxes;
        int col = i % num_boxes;
        offset += classes;
        boxes[i].x = (predictions[offset + 0] + col) / num_boxes;
        boxes[i].y = (predictions[offset + 1] + row) / num_boxes;
        boxes[i].w = predictions[offset + 2];
        boxes[i].h = predictions[offset + 3];
    }
}

void print_cocos(FILE *fp, int image_id, box *boxes, float **probs, int num_boxes, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < num_boxes*num_boxes; ++i){
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

void validate_recall(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *val_images = "/home/pjreddie/data/voc/test/2007_test.txt";
    list *plist = get_paths(val_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n - 1];

    int num_boxes = l.side;
    int num = l.n;
    int classes = l.classes;

    int j;

    box *boxes = calloc(num_boxes*num_boxes*num, sizeof(box));
    float **probs = calloc(num_boxes*num_boxes*num, sizeof(float *));
    for(j = 0; j < num_boxes*num_boxes*num; ++j) probs[j] = calloc(classes+1, sizeof(float *));

    int N = plist->size;
    int i=0;
    int k;

    float iou_thresh = .5;
    float thresh = .1;
    int total = 0;
    int correct = 0;
    float avg_iou = 0;
    int nms = 1;
    int proposals = 0;
    int save = 1;

    for (i = 0; i < N; ++i) {
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image resized = resize_image(orig, net.w, net.h);

        float *X = resized.data;
        float *predictions = network_predict(net, X);
        get_boxes(predictions+1+classes, num, num_boxes, 5+classes, boxes);
        get_probs(predictions, num*num_boxes*num_boxes, classes, 5+classes, probs);
        if (nms) do_nms(boxes, probs, num*num_boxes*num_boxes, (classes>0) ? classes : 1, iou_thresh);

        char *labelpath = find_replace(path, "images", "labels");
        labelpath = find_replace(labelpath, "JPEGImages", "labels");
        labelpath = find_replace(labelpath, ".jpg", ".txt");
        labelpath = find_replace(labelpath, ".JPEG", ".txt");

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < num_boxes*num_boxes*num; ++k){
            if(probs[k][0] > thresh){
                ++proposals;
                if(save){
                    char buff[256];
                    sprintf(buff, "/data/extracted/nms_preds/%d", proposals);
                    int dx = (boxes[k].x - boxes[k].w/2) * orig.w;
                    int dy = (boxes[k].y - boxes[k].h/2) * orig.h;
                    int w = boxes[k].w * orig.w;
                    int h = boxes[k].h * orig.h;
                    image cropped = crop_image(orig, dx, dy, w, h);
                    image sized = resize_image(cropped, 224, 224);
#ifdef OPENCV
                    save_image_jpg(sized, buff);
#endif
                    free_image(sized);
                    free_image(cropped);
                    sprintf(buff, "/data/extracted/nms_pred_boxes/%d.txt", proposals);
                    char *im_id = basecfg(path);
                    FILE *fp = fopen(buff, "w");
                    fprintf(fp, "%s %d %d %d %d\n", im_id, dx, dy, dx+w, dy+h);
                    fclose(fp);
                    free(im_id);
                }
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < num_boxes*num_boxes*num; ++k){
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
        free(truth);
        free_image(orig);
        free_image(resized);
        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
    }
}

void extract_boxes(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *val_images = "/home/pjreddie/data/voc/test/train.txt";
    list *plist = get_paths(val_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n - 1];

    int num_boxes = l.side;
    int num = l.n;
    int classes = l.classes;

    int j;

    box *boxes = calloc(num_boxes*num_boxes*num, sizeof(box));
    float **probs = calloc(num_boxes*num_boxes*num, sizeof(float *));
    for(j = 0; j < num_boxes*num_boxes*num; ++j) probs[j] = calloc(classes+1, sizeof(float *));

    int N = plist->size;
    int i=0;
    int k;

    int count = 0;
    float iou_thresh = .3;

    for (i = 0; i < N; ++i) {
        fprintf(stderr, "%5d %5d\n", i, count);
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image resized = resize_image(orig, net.w, net.h);

        float *X = resized.data;
        float *predictions = network_predict(net, X);
        get_boxes(predictions+1+classes, num, num_boxes, 5+classes, boxes);
        get_probs(predictions, num*num_boxes*num_boxes, classes, 5+classes, probs);

        char *labelpath = find_replace(path, "images", "labels");
        labelpath = find_replace(labelpath, "JPEGImages", "labels");
        labelpath = find_replace(labelpath, ".jpg", ".txt");
        labelpath = find_replace(labelpath, ".JPEG", ".txt");

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        FILE *label = stdin;
        for(k = 0; k < num_boxes*num_boxes*num; ++k){
            int overlaps = 0;
            for (j = 0; j < num_labels; ++j) {
                box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
                float iou = box_iou(boxes[k], t);
                if (iou > iou_thresh){
                    if (!overlaps) {
                        char buff[256];
                        sprintf(buff, "/data/extracted/labels/%d.txt", count);
                        label = fopen(buff, "w");
                        overlaps = 1;
                    }
                    fprintf(label, "%d %f\n", truth[j].id, iou);
                }
            }
            if (overlaps) {
                char buff[256];
                sprintf(buff, "/data/extracted/imgs/%d", count++);
                int dx = (boxes[k].x - boxes[k].w/2) * orig.w;
                int dy = (boxes[k].y - boxes[k].h/2) * orig.h;
                int w = boxes[k].w * orig.w;
                int h = boxes[k].h * orig.h;
                image cropped = crop_image(orig, dx, dy, w, h);
                image sized = resize_image(cropped, 224, 224);
#ifdef OPENCV
                save_image_jpg(sized, buff);
#endif
                free_image(sized);
                free_image(cropped);
                fclose(label);
            }
        }
        free(truth);
        free_image(orig);
        free_image(resized);
    }
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

    char *base = "/home/pjreddie/backup/";
    list *plist = get_paths("data/coco_val_5k.list");
    char **paths = (char **)list_to_array(plist);

    int num_boxes = 9;
    int num = 4;
    int classes = 1;

    int j;
    char buff[1024];
    snprintf(buff, 1024, "%s/coco_results.json", base);
    FILE *fp = fopen(buff, "w");
    fprintf(fp, "[\n");

    box *boxes = calloc(num_boxes*num_boxes*num, sizeof(box));
    float **probs = calloc(num_boxes*num_boxes*num, sizeof(float *));
    for(j = 0; j < num_boxes*num_boxes*num; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .01;
    int nms = 1;
    float iou_thresh = .5;

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    int nthreads = 8;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));
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
            float *predictions = network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            convert_cocos(predictions, classes, num_boxes, num, w, h, thresh, probs, boxes);
            if (nms) do_nms(boxes, probs, num_boxes, classes, iou_thresh);
            print_cocos(fp, image_id, boxes, probs, num_boxes, classes, w, h);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fseek(fp, -2, SEEK_CUR); 
    fprintf(fp, "\n]\n");
    fclose(fp);
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void test_coco(char *cfgfile, char *weightfile, char *filename)
{

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char input[256];
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            fgets(input, 256, stdin);
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        draw_coco(im, predictions, 7, "predictions");
        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}

void run_coco(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) test_coco(cfg, weights, filename);
    else if(0==strcmp(argv[2], "train")) train_coco(cfg, weights);
    else if(0==strcmp(argv[2], "extract")) extract_boxes(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_recall(cfg, weights);
}
