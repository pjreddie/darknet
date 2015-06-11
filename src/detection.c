#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"


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
            if(box[j+class] > 0){
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

void predict_detections(network net, data d, float threshold, int offset, int classes, int objectness, int background, int num_boxes, int per_box)
{
    matrix pred = network_predict_data(net, d);
    int j, k, class;
    for(j = 0; j < pred.rows; ++j){
        for(k = 0; k < pred.cols; k += per_box){
            float scale = 1.;
            int index = k/per_box;
            int row = index / num_boxes;
            int col = index % num_boxes;
            if (objectness) scale = 1.-pred.vals[j][k];
            for (class = 0; class < classes; ++class){
                int ci = k+classes+(background || objectness);
                float x = (pred.vals[j][ci + 0] + col)/num_boxes;
                float y = (pred.vals[j][ci + 1] + row)/num_boxes;
                float w = pred.vals[j][ci + 2]; // distance_from_edge(row, num_boxes);
                float h = pred.vals[j][ci + 3]; // distance_from_edge(col, num_boxes);
                w = pow(w, 2);
                h = pow(h, 2);
                float prob = scale*pred.vals[j][k+class+(background || objectness)];
                if(prob < threshold) continue;
                printf("%d %d %f %f %f %f %f\n", offset +  j, class, prob, x, y, w, h);
            }
        }
    }
    free_matrix(pred);
}

void validate_detection(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer layer = get_network_detection_layer(net);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    list *plist = get_paths("/home/pjreddie/data/voc/test.txt");
    char **paths = (char **)list_to_array(plist);

    int classes = layer.classes;
    int objectness = layer.objectness;
    int background = layer.background;
    int num_boxes = sqrt(get_detection_layer_locations(layer));

    int per_box = 4+classes+(background || objectness);
    int num_output = num_boxes*num_boxes*per_box;

    int m = plist->size;
    int i = 0;
    int splits = 100;

    int nthreads = 4;
    int t;
    data *val = calloc(nthreads, sizeof(data));
    data *buf = calloc(nthreads, sizeof(data));
    pthread_t *thr = calloc(nthreads, sizeof(data));

    time_t start = time(0);

    for(t = 0; t < nthreads; ++t){
        int num = (i+1+t)*m/splits - (i+t)*m/splits;
        char **part = paths+((i+t)*m/splits);
        thr[t] = load_data_thread(part, num, 0, 0, num_output, net.w, net.h, &(buf[t]));
    }

    for(i = nthreads; i <= splits; i += nthreads){
        for(t = 0; t < nthreads; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
        }
        for(t = 0; t < nthreads && i < splits; ++t){
            int num = (i+1+t)*m/splits - (i+t)*m/splits;
            char **part = paths+((i+t)*m/splits);
            thr[t] = load_data_thread(part, num, 0, 0, num_output, net.w, net.h, &(buf[t]));
        }

        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads; ++t){
            predict_detections(net, val[t], .001, (i-nthreads+t)*m/splits, classes, objectness, background, num_boxes, per_box);
            free_data(val[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void test_detection(char *cfgfile, char *weightfile)
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
    char filename[256];
    while(1){
        printf("Image Path: ");
        fflush(stdout);
        fgets(filename, 256, stdin);
        strtok(filename, "\n");
        image im = load_image_color(filename,0,0);
        image sized = resize_image(im, im_size, im_size);
        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", filename, sec(clock()-time));
        draw_detection(im, predictions, 7, "predictions");
        free_image(im);
        free_image(sized);
        #ifdef OPENCV
        cvWaitKey(0);
        #endif
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
    if(0==strcmp(argv[2], "test")) test_detection(cfg, weights);
    else if(0==strcmp(argv[2], "train")) train_detection(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_detection(cfg, weights);
}
