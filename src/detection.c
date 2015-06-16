#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"


char *class_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

void draw_detection(image im, float *box, int side, char *label)
{
    int classes = 20;
    int elems = 4+classes;
    int j;
    int r, c;

    for(r = 0; r < side; ++r){
        for(c = 0; c < side; ++c){
            j = (r*side + c) * elems;
            int class = max_index(box+j, classes);
            if(box[j+class] > 0.2){
                printf("%f %s\n", box[j+class], class_names[class]);
                float red = get_color(0,class,classes);
                float green = get_color(1,class,classes);
                float blue = get_color(2,class,classes);

                j += classes;
                float x = box[j+0];
                float y = box[j+1];
                x = (x+c)/side;
                y = (y+r)/side;
                float w = box[j+2]; //*maxwidth;
                float h = box[j+3]; //*maxheight;
                h = h*h;
                w = w*w;

                int left  = (x-w/2)*im.w;
                int right = (x+w/2)*im.w;
                int top   = (y-h/2)*im.h;
                int bot   = (y+h/2)*im.h;
                draw_box(im, left, top, right, bot, red, green, blue);
                draw_box(im, left+1, top+1, right+1, bot+1, red, green, blue);
                draw_box(im, left-1, top-1, right-1, bot-1, red, green, blue);
            }
        }
    }
    show_image(im, label);
}

void train_detection(char *cfgfile, char *weightfile)
{
    srand(time(0));
    data_seed = time(0);
    int imgnet = 0;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer layer = get_network_detection_layer(net);
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = 128;
    int i = net.seen/imgs;
    data train, buffer;

    int classes = layer.classes;
    int background = (layer.background || layer.objectness);
    printf("%d\n", background);
    int side = sqrt(get_detection_layer_locations(layer));

    char **paths;
    list *plist;
    if (imgnet){
        plist = get_paths("/home/pjreddie/data/imagenet/det.train.list");
    }else{
        //plist = get_paths("/home/pjreddie/data/voc/no_2012_val.txt");
        //plist = get_paths("/home/pjreddie/data/voc/no_2007_test.txt");
        //plist = get_paths("/home/pjreddie/data/voc/val_2012.txt");
        //plist = get_paths("/home/pjreddie/data/voc/no_2007_test.txt");
        //plist = get_paths("/home/pjreddie/data/coco/trainval.txt");
        plist = get_paths("/home/pjreddie/data/voc/all2007-2012.txt");
    }
    paths = (char **)list_to_array(plist);
    pthread_t load_thread = load_data_detection_thread(imgs, paths, plist->size, classes, net.w, net.h, side, side, background, &buffer);
    clock_t time;
    while(1){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_detection_thread(imgs, paths, plist->size, classes, net.w, net.h, side, side, background, &buffer);

/*
           image im = float_to_image(net.w, net.h, 3, train.X.vals[114]);
           image copy = copy_image(im);
           draw_detection(copy, train.y.vals[114], 7, "truth");
           cvWaitKey(0);
           free_image(copy);
           */

        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
        net.seen += imgs;
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock()-time), i*imgs);
        if(i == 100){
            net.learning_rate *= 10;
        }
        if(i%1000==0){
            char buff[256];
            sprintf(buff, "/home/pjreddie/imagenet_backup/%s_%d.weights",base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
}

void convert_detections(float *predictions, int classes, int objectness, int background, int num_boxes, int w, int h, float thresh, float **probs, box *boxes)
{
    int i,j;
    int per_box = 4+classes+(background || objectness);
    for (i = 0; i < num_boxes*num_boxes; ++i){
        float scale = 1;
        if(objectness) scale = 1-predictions[i*per_box];
        int offset = i*per_box+(background||objectness);
        for(j = 0; j < classes; ++j){
            float prob = scale*predictions[offset+j];
            probs[i][j] = (prob > thresh) ? prob : 0;
        }
        int row = i / num_boxes;
        int col = i % num_boxes;
        offset += classes;
        boxes[i].x = (predictions[offset + 0] + col) / num_boxes * w;
        boxes[i].y = (predictions[offset + 1] + row) / num_boxes * h;
        boxes[i].w = pow(predictions[offset + 2], 2) * w;
        boxes[i].h = pow(predictions[offset + 3], 2) * h;
    }
}

void do_nms(box *boxes, float **probs, int num_boxes, int classes, float thresh)
{
    int i, j, k;
    for(i = 0; i < num_boxes*num_boxes; ++i){
        int any = 0;
        for(k = 0; k < classes; ++k) any = any || (probs[i][k] > 0);
        if(!any) {
            continue;
        }
        for(j = i+1; j < num_boxes*num_boxes; ++j){
            if (box_iou(boxes[i], boxes[j]) > thresh){
                for(k = 0; k < classes; ++k){
                    if (probs[i][k] < probs[j][k]) probs[i][k] = 0;
                    else probs[j][k] = 0;
                }
            }
        }
    }
}

void print_detections(FILE **fps, char *id, box *boxes, float **probs, int num_boxes, int classes, int w, int h)
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
        
        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_detection(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    detection_layer layer = get_network_detection_layer(net);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    list *plist = get_paths("data/voc.2012test.list");
    char **paths = (char **)list_to_array(plist);

    int classes = layer.classes;
    int objectness = layer.objectness;
    int background = layer.background;
    int num_boxes = sqrt(get_detection_layer_locations(layer));

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, class_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(num_boxes*num_boxes, sizeof(box));
    float **probs = calloc(num_boxes*num_boxes, sizeof(float *));
    for(j = 0; j < num_boxes*num_boxes; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 8;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));
    for(t = 0; t < nthreads; ++t){
        thr[t] = load_image_thread(paths[i+t], &buf[t], &buf_resized[t], net.w, net.h);
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
            thr[t] = load_image_thread(paths[i+t], &buf[t], &buf_resized[t], net.w, net.h);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            float *predictions = network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            convert_detections(predictions, classes, objectness, background, num_boxes, w, h, thresh, probs, boxes);
            if (nms) do_nms(boxes, probs, num_boxes, classes, iou_thresh);
            print_detections(fps, id, boxes, probs, num_boxes, classes, w, h);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void test_detection(char *cfgfile, char *weightfile, char *filename)
{

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer layer = get_network_detection_layer(net);
    if (!layer.joint) fprintf(stderr, "Detection layer should use joint prediction to draw correctly.\n");
    int im_size = 448;
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
        image sized = resize_image(im, im_size, im_size);
        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        draw_detection(im, predictions, 7, "predictions");
        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}

void run_detection(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) test_detection(cfg, weights, filename);
    else if(0==strcmp(argv[2], "train")) train_detection(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_detection(cfg, weights);
}
