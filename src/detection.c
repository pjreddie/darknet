#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"


char *class_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
char *inet_class_names[] = {"bg", "accordion", "airplane", "ant", "antelope", "apple", "armadillo", "artichoke", "axe", "baby bed", "backpack", "bagel", "balance beam", "banana", "band aid", "banjo", "baseball", "basketball", "bathing cap", "beaker", "bear", "bee", "bell pepper", "bench", "bicycle", "binder", "bird", "bookshelf", "bow tie", "bow", "bowl", "brassiere", "burrito", "bus", "butterfly", "camel", "can opener", "car", "cart", "cattle", "cello", "centipede", "chain saw", "chair", "chime", "cocktail shaker", "coffee maker", "computer keyboard", "computer mouse", "corkscrew", "cream", "croquet ball", "crutch", "cucumber", "cup or mug", "diaper", "digital clock", "dishwasher", "dog", "domestic cat", "dragonfly", "drum", "dumbbell", "electric fan", "elephant", "face powder", "fig", "filing cabinet", "flower pot", "flute", "fox", "french horn", "frog", "frying pan", "giant panda", "goldfish", "golf ball", "golfcart", "guacamole", "guitar", "hair dryer", "hair spray", "hamburger", "hammer", "hamster", "harmonica", "harp", "hat with a wide brim", "head cabbage", "helmet", "hippopotamus", "horizontal bar", "horse", "hotdog", "iPod", "isopod", "jellyfish", "koala bear", "ladle", "ladybug", "lamp", "laptop", "lemon", "lion", "lipstick", "lizard", "lobster", "maillot", "maraca", "microphone", "microwave", "milk can", "miniskirt", "monkey", "motorcycle", "mushroom", "nail", "neck brace", "oboe", "orange", "otter", "pencil box", "pencil sharpener", "perfume", "person", "piano", "pineapple", "ping-pong ball", "pitcher", "pizza", "plastic bag", "plate rack", "pomegranate", "popsicle", "porcupine", "power drill", "pretzel", "printer", "puck", "punching bag", "purse", "rabbit", "racket", "ray", "red panda", "refrigerator", "remote control", "rubber eraser", "rugby ball", "ruler", "salt or pepper shaker", "saxophone", "scorpion", "screwdriver", "seal", "sheep", "ski", "skunk", "snail", "snake", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sofa", "spatula", "squirrel", "starfish", "stethoscope", "stove", "strainer", "strawberry", "stretcher", "sunglasses", "swimming trunks", "swine", "syringe", "table", "tape player", "tennis ball", "tick", "tie", "tiger", "toaster", "traffic light", "train", "trombone", "trumpet", "turtle", "tv or monitor", "unicycle", "vacuum", "violin", "volleyball", "waffle iron", "washer", "water bottle", "watercraft", "whale", "wine bottle", "zebra"};
#define AMNT 3
void draw_detection(image im, float *box, int side, char *label)
{
    int classes = 20;
    int elems = 4+classes;
    int j;
    int r, c;

    for(r = 0; r < side; ++r){
        for(c = 0; c < side; ++c){
            j = (r*side + c) * elems;
            //printf("%d\n", j);
            //printf("Prob: %f\n", box[j]);
            int class = max_index(box+j, classes);
            if(box[j+class] > .05){
                //int z;
                //for(z = 0; z < classes; ++z) printf("%f %s\n", box[j+z], class_names[z]);
                printf("%f %s\n", box[j+class], class_names[class]);
                float red = get_color(0,class,classes);
                float green = get_color(1,class,classes);
                float blue = get_color(2,class,classes);

                //float maxheight = distance_from_edge(r, side);
                //float maxwidth  = distance_from_edge(c, side);
                j += classes;
                float x = box[j+0];
                float y = box[j+1];
                x = (x+c)/side;
                y = (y+r)/side;
                float w = box[j+2]; //*maxwidth;
                float h = box[j+3]; //*maxheight;
                h = h*h;
                w = w*w;
                //printf("coords %f %f %f %f\n", x, y, w, h);

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
    //printf("Done\n");
    show_image(im, label);
}

void draw_localization(image im, float *box)
{
    int classes = 20;
    int class;
    for(class = 0; class < classes; ++class){
        //int z;
        //for(z = 0; z < classes; ++z) printf("%f %s\n", box[j+z], class_names[z]);
        float red = get_color(0,class,classes);
        float green = get_color(1,class,classes);
        float blue = get_color(2,class,classes);

        int j = class*4;
        float x = box[j+0];
        float y = box[j+1];
        float w = box[j+2]; //*maxheight;
        float h = box[j+3]; //*maxwidth;
        //printf("coords %f %f %f %f\n", x, y, w, h);

        int left  = (x-w/2)*im.w;
        int right = (x+w/2)*im.w;
        int top   = (y-h/2)*im.h;
        int bot   = (y+h/2)*im.h;
        draw_box(im, left, top, right, bot, red, green, blue);
    }
    //printf("Done\n");
}

void train_localization(char *cfgfile, char *weightfile)
{
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
    int classes = 20;
    int i = net.seen/imgs;
    data train, buffer;

    char **paths;
    list *plist;
    plist = get_paths("/home/pjreddie/data/voc/loc.2012val.txt");
    paths = (char **)list_to_array(plist);
    pthread_t load_thread = load_data_localization_thread(imgs, paths, plist->size, classes, net.w, net.h, &buffer);
    clock_t time;
    while(1){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_localization_thread(imgs, paths, plist->size, classes, net.w, net.h, &buffer);

        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);

        //TODO
        #ifdef GPU
        float *out = get_network_output_gpu(net);
        #else
        float *out = get_network_output(net);
        #endif
        image im = float_to_image(net.w, net.h, 3, train.X.vals[127]);
        image copy = copy_image(im);
        draw_localization(copy, &(out[63*80]));
        draw_localization(copy, train.y.vals[127]);
        show_image(copy, "box");
        cvWaitKey(0);
        free_image(copy);

        net.seen += imgs;
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock()-time), i*imgs);
        if(i%100==0){
            char buff[256];
            sprintf(buff, "/home/pjreddie/imagenet_backup/%s_%d.weights",base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
}

void train_detection_teststuff(char *cfgfile, char *weightfile)
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
    net.learning_rate = 0;
    net.decay = 0;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = 128;
    int i = net.seen/imgs;
    data train, buffer;

    int classes = layer.classes;
    int background = layer.background;
    int side = sqrt(get_detection_layer_locations(layer));

    char **paths;
    list *plist;
    if (imgnet){
        plist = get_paths("/home/pjreddie/data/imagenet/det.train.list");
    }else{
        plist = get_paths("/home/pjreddie/data/voc/val_2012.txt");
        //plist = get_paths("/home/pjreddie/data/voc/no_2007_test.txt");
        //plist = get_paths("/home/pjreddie/data/coco/trainval.txt");
        //plist = get_paths("/home/pjreddie/data/voc/all2007-2012.txt");
    }
    paths = (char **)list_to_array(plist);
    pthread_t load_thread = load_data_detection_thread(imgs, paths, plist->size, classes, net.w, net.h, side, side, background, &buffer);
    clock_t time;
    cost_layer clayer = net.layers[net.n-1];
    while(1){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_detection_thread(imgs, paths, plist->size, classes, net.w, net.h, side, side, background, &buffer);

        /*
           image im = float_to_image(net.w, net.h, 3, train.X.vals[114]);
           image copy = copy_image(im);
           draw_detection(copy, train.y.vals[114], 7);
           free_image(copy);
         */

        int z;
        int count = 0;
        float sx, sy, sw, sh;
        sx = sy = sw = sh = 0;
        for(z = 0; z < clayer.batch*clayer.inputs; z += 24){
            if(clayer.delta[z+20]){
                ++count;
                sx += fabs(clayer.delta[z+20])*64;
                sy += fabs(clayer.delta[z+21])*64;
                sw += fabs(clayer.delta[z+22])*448;
                sh += fabs(clayer.delta[z+23])*448;
            }
        }
        printf("Avg error: %f, %f, %f x %f\n", sx/count, sy/count, sw/count, sh/count);

        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
        net.seen += imgs;
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock()-time), i*imgs);
        if(i == 100){
            //net.learning_rate *= 10;
        }
        if(i%100==0){
            char buff[256];
            sprintf(buff, "/home/pjreddie/imagenet_backup/%s_%d.weights",base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
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
    int background = layer.background;
    int side = sqrt(get_detection_layer_locations(layer));

    char **paths;
    list *plist;
    if (imgnet){
        plist = get_paths("/home/pjreddie/data/imagenet/det.train.list");
    }else{
        //plist = get_paths("/home/pjreddie/data/voc/no_2012_val.txt");
        //plist = get_paths("/home/pjreddie/data/voc/no_2007_test.txt");
        //plist = get_paths("/home/pjreddie/data/voc/val_2012.txt");
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

void predict_detections(network net, data d, float threshold, int offset, int classes, int nuisance, int background, int num_boxes, int per_box)
{
    matrix pred = network_predict_data(net, d);
    int j, k, class;
    for(j = 0; j < pred.rows; ++j){
        for(k = 0; k < pred.cols; k += per_box){
            float scale = 1.;
            int index = k/per_box;
            int row = index / num_boxes;
            int col = index % num_boxes;
            if (nuisance) scale = 1.-pred.vals[j][k];
            for (class = 0; class < classes; ++class){
                int ci = k+classes+background+nuisance;
                float x = (pred.vals[j][ci + 0] + col)/num_boxes;
                float y = (pred.vals[j][ci + 1] + row)/num_boxes;
                float w = pred.vals[j][ci + 2]; // distance_from_edge(row, num_boxes);
                float h = pred.vals[j][ci + 3]; // distance_from_edge(col, num_boxes);
                w = pow(w, 2);
                h = pow(h, 2);
                float prob = scale*pred.vals[j][k+class+background+nuisance];
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

    //list *plist = get_paths("/home/pjreddie/data/voc/test_2007.txt");
    //list *plist = get_paths("/home/pjreddie/data/voc/val_2012.txt");
    list *plist = get_paths("/home/pjreddie/data/voc/test.txt");
    //list *plist = get_paths("/home/pjreddie/data/voc/val.expanded.txt");
    //list *plist = get_paths("/home/pjreddie/data/voc/train.txt");
    char **paths = (char **)list_to_array(plist);

    int classes = layer.classes;
    int nuisance = layer.nuisance;
    int background = (layer.background && !nuisance);
    int num_boxes = sqrt(get_detection_layer_locations(layer));

    int per_box = 4+classes+background+nuisance;
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

    //clock_t time;
    for(i = nthreads; i <= splits; i += nthreads){
        //time=clock();
        for(t = 0; t < nthreads; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
        }
        for(t = 0; t < nthreads && i < splits; ++t){
            int num = (i+1+t)*m/splits - (i+t)*m/splits;
            char **part = paths+((i+t)*m/splits);
            thr[t] = load_data_thread(part, num, 0, 0, num_output, net.w, net.h, &(buf[t]));
        }

        //fprintf(stderr, "%d: Loaded: %lf seconds\n", i, sec(clock()-time));
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads; ++t){
            predict_detections(net, val[t], .001, (i-nthreads+t)*m/splits, classes, nuisance, background, num_boxes, per_box);
            free_data(val[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void do_mask(network net, data d, int offset, int classes, int nuisance, int background, int num_boxes, int per_box)
{
    matrix pred = network_predict_data(net, d);
    int j, k;
    for(j = 0; j < pred.rows; ++j){
        printf("%d ", offset +  j);
        for(k = 0; k < pred.cols; k += per_box){
            float scale = 1.-pred.vals[j][k];
            printf("%f ", scale);
        }
        printf("\n");
    }
    free_matrix(pred);
}

void mask_detection(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer layer = get_network_detection_layer(net);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    list *plist = get_paths("/home/pjreddie/data/voc/test_2007.txt");
    //list *plist = get_paths("/home/pjreddie/data/voc/val_2012.txt");
    //list *plist = get_paths("/home/pjreddie/data/voc/test.txt");
    //list *plist = get_paths("/home/pjreddie/data/voc/val.expanded.txt");
    //list *plist = get_paths("/home/pjreddie/data/voc/train.txt");
    char **paths = (char **)list_to_array(plist);

    int classes = layer.classes;
    int nuisance = layer.nuisance;
    int background = (layer.background && !nuisance);
    int num_boxes = sqrt(get_detection_layer_locations(layer));

    int per_box = 4+classes+background+nuisance;
    int num_output = num_boxes*num_boxes*per_box;

    int m = plist->size;
    int i = 0;
    int splits = 100;

    int nthreads = 4;
    int t;
    data *val = calloc(nthreads, sizeof(data));
    data *buf = calloc(nthreads, sizeof(data));
    pthread_t *thr = calloc(nthreads, sizeof(data));
    for(t = 0; t < nthreads; ++t){
        int num = (i+1+t)*m/splits - (i+t)*m/splits;
        char **part = paths+((i+t)*m/splits);
        thr[t] = load_data_thread(part, num, 0, 0, num_output, net.w, net.h, &(buf[t]));
    }

    clock_t time;
    for(i = nthreads; i <= splits; i += nthreads){
        time=clock();
        for(t = 0; t < nthreads; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
        }
        for(t = 0; t < nthreads && i < splits; ++t){
            int num = (i+1+t)*m/splits - (i+t)*m/splits;
            char **part = paths+((i+t)*m/splits);
            thr[t] = load_data_thread(part, num, 0, 0, num_output, net.w, net.h, &(buf[t]));
        }

        fprintf(stderr, "%d: Loaded: %lf seconds\n", i, sec(clock()-time));
        for(t = 0; t < nthreads; ++t){
            do_mask(net, val[t], (i-nthreads+t)*m/splits, classes, nuisance, background, num_boxes, per_box);
            free_data(val[t]);
        }
        time=clock();
    }
}

void validate_detection_post(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);

    network post = parse_network_cfg("cfg/localize.cfg");
    load_weights(&post, "/home/pjreddie/imagenet_backup/localize_1000.weights");
    set_batch_network(&post, 1);

    detection_layer layer = get_network_detection_layer(net);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    //list *plist = get_paths("/home/pjreddie/data/voc/test_2007.txt");
    list *plist = get_paths("/home/pjreddie/data/voc/val_2012.txt");
    //list *plist = get_paths("/home/pjreddie/data/voc/test.txt");
    //list *plist = get_paths("/home/pjreddie/data/voc/val.expanded.txt");
    //list *plist = get_paths("/home/pjreddie/data/voc/train.txt");
    char **paths = (char **)list_to_array(plist);

    int classes = layer.classes;
    int nuisance = layer.nuisance;
    int background = (layer.background && !nuisance);
    int num_boxes = sqrt(get_detection_layer_locations(layer));

    int per_box = 4+classes+background+nuisance;

    int m = plist->size;
    int i = 0;
    float threshold = .01;

    clock_t time = clock();
    for(i = 0; i < m; ++i){
        image im = load_image_color(paths[i], 0, 0);
        if(i % 100 == 0) {
            fprintf(stderr, "%d: Loaded: %lf seconds\n", i, sec(clock()-time));
            time = clock();
        }
        image sized = resize_image(im, net.w, net.h);
        float *out = network_predict(net, sized.data);
        free_image(sized);
        int k, class;
        //show_image(im, "original");
        int num_output = num_boxes*num_boxes*per_box;
        //image cp1 = copy_image(im);
        //draw_detection(cp1, out, 7, "before");
        for(k = 0; k < num_output; k += per_box){
            float *post_out = 0;
            float scale = 1.;
            int index = k/per_box;
            int row = index / num_boxes;
            int col = index % num_boxes;
            if (nuisance) scale = 1.-out[k];
            for (class = 0; class < classes; ++class){
                int ci = k+classes+background+nuisance;
                float x = (out[ci + 0] + col)/num_boxes;
                float y = (out[ci + 1] + row)/num_boxes;
                float w = out[ci + 2]; //* distance_from_edge(row, num_boxes);
                float h = out[ci + 3]; //* distance_from_edge(col, num_boxes);
                w = w*w;
                h = h*h;
                float prob = scale*out[k+class+background+nuisance];
                if (prob >= threshold) {
                    x *= im.w;
                    y *= im.h;
                    w *= im.w;
                    h *= im.h;
                    w += 32;
                    h += 32;
                    int left = (x - w/2);
                    int top = (y - h/2);
                    int right = (x + w/2);
                    int bot = (y+h/2);
                    if (left < 0) left = 0;
                    if (right > im.w) right = im.w;
                    if (top < 0) top = 0;
                    if (bot > im.h) bot = im.h;

                    image crop = crop_image(im, left, top, right-left, bot-top);
                    image resize = resize_image(crop, post.w, post.h);
                    if (!post_out){
                        post_out = network_predict(post, resize.data);
                    }
                    /*
                    draw_localization(resize, post_out);
                    show_image(resize, "second");
                    fprintf(stderr, "%s\n", class_names[class]);
                    cvWaitKey(0);
                    */
                    int index = 4*class;
                    float px = post_out[index+0];
                    float py = post_out[index+1];
                    float pw = post_out[index+2];
                    float ph = post_out[index+3];
                    px = (px * crop.w + left) / im.w;
                    py = (py * crop.h + top) / im.h;
                    pw = (pw * crop.w) / im.w;
                    ph = (ph * crop.h) / im.h;

                    out[ci + 0] = px*num_boxes - col;
                    out[ci + 1] = py*num_boxes - row;
                    out[ci + 2] = sqrt(pw);
                    out[ci + 3] = sqrt(ph);
                    /*
                       show_image(crop, "cropped");
                       cvWaitKey(0);
                     */
                    free_image(crop);
                    free_image(resize);
                    printf("%d %d %f %f %f %f %f\n", i, class, prob, px, py, pw, ph);
                }
            }
        }
        /*
        image cp2 = copy_image(im);
        draw_detection(cp2, out, 7, "after");
        cvWaitKey(0);
        */
    }
}

void test_detection(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    int im_size = 448;
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char filename[256];
    while(1){
        fgets(filename, 256, stdin);
        strtok(filename, "\n");
        image im = load_image_color(filename,0,0);
        image sized = resize_image(im, im_size, im_size);
        printf("%d %d %d\n", im.h, im.w, im.c);
        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", filename, sec(clock()-time));
        draw_detection(im, predictions, 7, "YOLO#SWAG#BLAZEIT");
        free_image(im);
        free_image(sized);
        cvWaitKey(0);
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
    else if(0==strcmp(argv[2], "teststuff")) train_detection_teststuff(cfg, weights);
    else if(0==strcmp(argv[2], "trainloc")) train_localization(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_detection(cfg, weights);
    else if(0==strcmp(argv[2], "mask")) mask_detection(cfg, weights);
    else if(0==strcmp(argv[2], "validpost")) validate_detection_post(cfg, weights);
}
