#include "network.h"
#include "detection_layer.h"
#include "utils.h"
#include "parser.h"


char *class_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
char *inet_class_names[] = {"bg", "accordion", "airplane", "ant", "antelope", "apple", "armadillo", "artichoke", "axe", "baby bed", "backpack", "bagel", "balance beam", "banana", "band aid", "banjo", "baseball", "basketball", "bathing cap", "beaker", "bear", "bee", "bell pepper", "bench", "bicycle", "binder", "bird", "bookshelf", "bow tie", "bow", "bowl", "brassiere", "burrito", "bus", "butterfly", "camel", "can opener", "car", "cart", "cattle", "cello", "centipede", "chain saw", "chair", "chime", "cocktail shaker", "coffee maker", "computer keyboard", "computer mouse", "corkscrew", "cream", "croquet ball", "crutch", "cucumber", "cup or mug", "diaper", "digital clock", "dishwasher", "dog", "domestic cat", "dragonfly", "drum", "dumbbell", "electric fan", "elephant", "face powder", "fig", "filing cabinet", "flower pot", "flute", "fox", "french horn", "frog", "frying pan", "giant panda", "goldfish", "golf ball", "golfcart", "guacamole", "guitar", "hair dryer", "hair spray", "hamburger", "hammer", "hamster", "harmonica", "harp", "hat with a wide brim", "head cabbage", "helmet", "hippopotamus", "horizontal bar", "horse", "hotdog", "iPod", "isopod", "jellyfish", "koala bear", "ladle", "ladybug", "lamp", "laptop", "lemon", "lion", "lipstick", "lizard", "lobster", "maillot", "maraca", "microphone", "microwave", "milk can", "miniskirt", "monkey", "motorcycle", "mushroom", "nail", "neck brace", "oboe", "orange", "otter", "pencil box", "pencil sharpener", "perfume", "person", "piano", "pineapple", "ping-pong ball", "pitcher", "pizza", "plastic bag", "plate rack", "pomegranate", "popsicle", "porcupine", "power drill", "pretzel", "printer", "puck", "punching bag", "purse", "rabbit", "racket", "ray", "red panda", "refrigerator", "remote control", "rubber eraser", "rugby ball", "ruler", "salt or pepper shaker", "saxophone", "scorpion", "screwdriver", "seal", "sheep", "ski", "skunk", "snail", "snake", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sofa", "spatula", "squirrel", "starfish", "stethoscope", "stove", "strainer", "strawberry", "stretcher", "sunglasses", "swimming trunks", "swine", "syringe", "table", "tape player", "tennis ball", "tick", "tie", "tiger", "toaster", "traffic light", "train", "trombone", "trumpet", "turtle", "tv or monitor", "unicycle", "vacuum", "violin", "volleyball", "waffle iron", "washer", "water bottle", "watercraft", "whale", "wine bottle", "zebra"};
#define AMNT 3
void draw_detection(image im, float *box, int side)
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
            if(box[j+class] > .2){
                //int z;
                //for(z = 0; z < classes; ++z) printf("%f %s\n", box[j+z], class_names[z]);
                printf("%f %s\n", box[j+class], class_names[class]);
                float red = get_color(0,class,classes);
                float green = get_color(1,class,classes);
                float blue = get_color(2,class,classes);

                //float maxheight = distance_from_edge(r, side);
                //float maxwidth  = distance_from_edge(c, side);
                j += classes;
                float y = box[j+0];
                float x = box[j+1];
                x = (x+c)/side;
                y = (y+r)/side;
                float h = box[j+2]; //*maxheight;
                float w = box[j+3]; //*maxwidth;
                h = h*h;
                w = w*w;
                //printf("coords %f %f %f %f\n", x, y, w, h);

                int left  = (x-w/2)*im.w;
                int right = (x+w/2)*im.w;
                int top   = (y-h/2)*im.h;
                int bot   = (y+h/2)*im.h;
                draw_box(im, left, top, right, bot, red, green, blue);
            }
        }
    }
    //printf("Done\n");
    show_image(im, "box");
    cvWaitKey(0);
}

void train_detection(char *cfgfile, char *weightfile)
{
    srand(time(0));
    int imgnet = 0;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer *layer = get_network_detection_layer(net);
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = 128;
    int i = net.seen/imgs;
    data train, buffer;

    int classes = layer->classes;
    int background = layer->background;
    int side = sqrt(get_detection_layer_locations(*layer));

    char **paths;
    list *plist;
    if (imgnet){
        plist = get_paths("/home/pjreddie/data/imagenet/det.train.list");
    }else{
        plist = get_paths("/home/pjreddie/data/voc/trainall.txt");
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
           draw_detection(im, train.y.vals[114], 7);
           */

        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
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

void validate_detection(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer *layer = get_network_detection_layer(net);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    list *plist = get_paths("/home/pjreddie/data/voc/val.txt");
    //list *plist = get_paths("/home/pjreddie/data/voc/val.expanded.txt");
    //list *plist = get_paths("/home/pjreddie/data/voc/train.txt");
    char **paths = (char **)list_to_array(plist);

    int classes = layer->classes;
    int nuisance = layer->nuisance;
    int background = (layer->background && !nuisance);
    int num_boxes = sqrt(get_detection_layer_locations(*layer));

    int per_box = 4+classes+background+nuisance;
    int num_output = num_boxes*num_boxes*per_box;

    int m = plist->size;
    int i = 0;
    int splits = 100;
    int num = (i+1)*m/splits - i*m/splits;

    fprintf(stderr, "%d\n", m);
    data val, buffer;
    pthread_t load_thread = load_data_thread(paths, num, 0, 0, num_output, net.w, net.h, &buffer);
    clock_t time;
    for(i = 1; i <= splits; ++i){
        time=clock();
        pthread_join(load_thread, 0);
        val = buffer;

        num = (i+1)*m/splits - i*m/splits;
        char **part = paths+(i*m/splits);
        if(i != splits) load_thread = load_data_thread(part, num, 0, 0, num_output, net.w, net.h, &buffer);

        fprintf(stderr, "%d: Loaded: %lf seconds\n", i, sec(clock()-time));
        matrix pred = network_predict_data(net, val);
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
                    float y = (pred.vals[j][ci + 0] + row)/num_boxes;
                    float x = (pred.vals[j][ci + 1] + col)/num_boxes;
                    float h = pred.vals[j][ci + 2]; //* distance_from_edge(row, num_boxes);
                    h = h*h;
                    float w = pred.vals[j][ci + 3]; //* distance_from_edge(col, num_boxes);
                    w = w*w;
                    float prob = scale*pred.vals[j][k+class+background+nuisance];
                    if(prob < .001) continue;
                    printf("%d %d %f %f %f %f %f\n", (i-1)*m/splits + j, class, prob, y, x, h, w);
                }
            }
        }
        time=clock();
        free_data(val);
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
        image im = load_image_color(filename, im_size, im_size);
        translate_image(im, -128);
        scale_image(im, 1/128.);
        printf("%d %d %d\n", im.h, im.w, im.c);
        float *X = im.data;
        time=clock();
        float *predictions = network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", filename, sec(clock()-time));
        draw_detection(im, predictions, 7);
        free_image(im);
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
