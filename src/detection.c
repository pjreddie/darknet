#include "network.h"
#include "utils.h"
#include "parser.h"


char *class_names[] = {"bg", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
#define AMNT 3
void draw_detection(image im, float *box, int side)
{
    int classes = 21;
    int elems = 4+classes;
    int j;
    int r, c;

    for(r = 0; r < side; ++r){
        for(c = 0; c < side; ++c){
            j = (r*side + c) * elems;
            //printf("%d\n", j);
            //printf("Prob: %f\n", box[j]);
            int class = max_index(box+j, classes);
            if(box[j+class] > .02 || 1){
                //int z;
                //for(z = 0; z < classes; ++z) printf("%f %s\n", box[j+z], class_names[z]);
                printf("%f %s\n", box[j+class], class_names[class]);
                float red = get_color(0,class,classes);
                float green = get_color(1,class,classes);
                float blue = get_color(2,class,classes);

                j += classes;
                int d = im.w/side;
                int y = r*d+box[j]*d;
                int x = c*d+box[j+1]*d;
                int h = box[j+2]*im.h;
                int w = box[j+3]*im.w;
                draw_box(im, x-w/2, y-h/2, x+w/2, y+h/2,red,green,blue);
            }
        }
    }
    //printf("Done\n");
    show_image(im, "box");
    cvWaitKey(0);
}

void train_detection(char *cfgfile, char *weightfile)
{
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    //net.seen = 0;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = 128;
    srand(time(0));
    //srand(23410);
    int i = net.seen/imgs;
    list *plist = get_paths("/home/pjreddie/data/voc/train.txt");
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    data train, buffer;
    int im_dim = 512;
    int jitter = 64;
    int classes = 20;
    int background = 1;
    pthread_t load_thread = load_data_detection_thread(imgs, paths, plist->size, classes, im_dim, im_dim, 7, 7, jitter, background, &buffer);
    clock_t time;
    while(1){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_detection_thread(imgs, paths, plist->size, classes, im_dim, im_dim, 7, 7, jitter, background, &buffer);

/*
           image im = float_to_image(im_dim - jitter, im_dim-jitter, 3, train.X.vals[114]);
           draw_detection(im, train.y.vals[114], 7);
           show_image(im, "truth");
           cvWaitKey(0);
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
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    list *plist = get_paths("/home/pjreddie/data/voc/val.txt");
    //list *plist = get_paths("/home/pjreddie/data/voc/train.txt");
    char **paths = (char **)list_to_array(plist);
    int im_size = 448;
    int classes = 20;
    int background = 0;
    int nuisance = 1;
    int num_output = 7*7*(4+classes+background+nuisance);

    int m = plist->size;
    int i = 0;
    int splits = 100;
    int num = (i+1)*m/splits - i*m/splits;

    fprintf(stderr, "%d\n", m);
    data val, buffer;
    pthread_t load_thread = load_data_thread(paths, num, 0, 0, num_output, im_size, im_size, &buffer);
    clock_t time;
    for(i = 1; i <= splits; ++i){
        time=clock();
        pthread_join(load_thread, 0);
        val = buffer;

        num = (i+1)*m/splits - i*m/splits;
        char **part = paths+(i*m/splits);
        if(i != splits) load_thread = load_data_thread(part, num, 0, 0, num_output, im_size, im_size, &buffer);

        fprintf(stderr, "%d: Loaded: %lf seconds\n", i, sec(clock()-time));
        matrix pred = network_predict_data(net, val);
        int j, k, class;
        for(j = 0; j < pred.rows; ++j){
            for(k = 0; k < pred.cols; k += classes+4+background+nuisance){
                float scale = 1.;
                if(nuisance) scale = 1.-pred.vals[j][k];
                for(class = 0; class < classes; ++class){
                    int index = (k)/(classes+4+background+nuisance); 
                    int r = index/7;
                    int c = index%7;
                    int ci = k+classes+background+nuisance;
                    float y = (r + pred.vals[j][ci + 0])/7.;
                    float x = (c + pred.vals[j][ci + 1])/7.;
                    float h = pred.vals[j][ci + 2];
                    float w = pred.vals[j][ci + 3];
                    printf("%d %d %f %f %f %f %f\n", (i-1)*m/splits + j, class, scale*pred.vals[j][k+class+background+nuisance], y, x, h, w);
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
