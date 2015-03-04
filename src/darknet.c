#include "connected_layer.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "network.h"
#include "image.h"
#include "parser.h"
#include "data.h"
#include "matrix.h"
#include "utils.h"
#include "blas.h"
#include "matrix.h"
#include "server.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#define _GNU_SOURCE
#include <fenv.h>

void test_load()
{
    image dog = load_image("dog.jpg", 300, 400);
    show_image(dog, "Test Load");
    show_image_layers(dog, "Test Load");
}

void test_parser()
{
    network net = parse_network_cfg("cfg/trained_imagenet.cfg");
    save_network(net, "cfg/trained_imagenet_smaller.cfg");
}

char *class_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
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

char *basename(char *cfgfile)
{
    char *c = cfgfile;
    char *next;
    while((next = strchr(c, '/')))
    {
        c = next+1;
    }
    c = copy_string(c);
    next = strchr(c, '_');
    if (next) *next = 0;
    next = strchr(c, '.');
    if (next) *next = 0;
    return c;
}

void train_detection_net(char *cfgfile, char *weightfile)
{
    char *base = basename(cfgfile);
    printf("%s\n", base);
    float avg_loss = 1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
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
    int classes = 21;
    pthread_t load_thread = load_data_detection_thread(imgs, paths, plist->size, classes, im_dim, im_dim, 7, 7, jitter, &buffer);
    clock_t time;
    while(1){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_detection_thread(imgs, paths, plist->size, classes, im_dim, im_dim, 7, 7, jitter, &buffer);

        /*
           image im = float_to_image(im_dim - jitter, im_dim-jitter, 3, train.X.vals[0]);
           draw_detection(im, train.y.vals[0], 7);
           show_image(im, "truth");
           cvWaitKey(0);
         */

        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
        net.seen += imgs;
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

void validate_detection_net(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    list *plist = get_paths("/home/pjreddie/data/voc/val.txt");
    char **paths = (char **)list_to_array(plist);
    int num_output = 1225;
    int im_size = 448;
    int classes = 21;

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
            for(k = 0; k < pred.cols; k += classes+4){

                /*
                   int z;
                   for(z = 0; z < 25; ++z) printf("%f, ", pred.vals[j][k+z]);
                   printf("\n");
                 */

                //if (pred.vals[j][k] > .001){
                for(class = 0; class < classes-1; ++class){
                    int index = (k)/(classes+4); 
                    int r = index/7;
                    int c = index%7;
                    float y = (r + pred.vals[j][k+0+classes])/7.;
                    float x = (c + pred.vals[j][k+1+classes])/7.;
                    float h = pred.vals[j][k+2+classes];
                    float w = pred.vals[j][k+3+classes];
                    printf("%d %d %f %f %f %f %f\n", (i-1)*m/splits + j, class, pred.vals[j][k+class], y, x, h, w);
                }
                //}
            }
        }

        time=clock();
        free_data(val);
    }
}
/*

   void train_imagenet_distributed(char *address)
   {
   float avg_loss = 1;
   srand(time(0));
   network net = parse_network_cfg("cfg/net.cfg");
   set_learning_network(&net, 0, 1, 0);
   printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
   int imgs = net.batch;
   int i = 0;
   char **labels = get_labels("/home/pjreddie/data/imagenet/cls.labels.list");
   list *plist = get_paths("/data/imagenet/cls.train.list");
   char **paths = (char **)list_to_array(plist);
   printf("%d\n", plist->size);
   clock_t time;
   data train, buffer;
   pthread_t load_thread = load_data_thread(paths, imgs, plist->size, labels, 1000, 224, 224, &buffer);
   while(1){
   i += 1;

   time=clock();
   client_update(net, address);
   printf("Updated: %lf seconds\n", sec(clock()-time));

   time=clock();
   pthread_join(load_thread, 0);
   train = buffer;
   normalize_data_rows(train);
   load_thread = load_data_thread(paths, imgs, plist->size, labels, 1000, 224, 224, &buffer);
   printf("Loaded: %lf seconds\n", sec(clock()-time));
   time=clock();

   float loss = train_network(net, train);
   avg_loss = avg_loss*.9 + loss*.1;
   printf("%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock()-time), i*imgs);
   free_data(train);
   }
   }
 */

void convert(char *cfgfile, char *outfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    save_network(net, outfile);
}

void train_captcha(char *cfgfile, char *weightfile)
{
    float avg_loss = -1;
    srand(time(0));
    char *base = basename(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = 1024;
    int i = net.seen/imgs;
    list *plist = get_paths("/data/captcha/train.list");
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    clock_t time;
    while(1){
        ++i;
        time=clock();
        data train = load_data_captcha(paths, imgs, plist->size, 10, 60, 200);
        translate_data_rows(train, -128);
        scale_data_rows(train, 1./128);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
        net.seen += imgs;
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock()-time), net.seen);
        free_data(train);
        if(i%100==0){
            char buff[256];
            sprintf(buff, "/home/pjreddie/imagenet_backup/%s_%d.weights",base, i);
            save_weights(net, buff);
        }
    }
}


void validate_captcha(char *cfgfile, char *weightfile)
{
    srand(time(0));
    char *base = basename(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    int imgs = 1000;
    int numchars = 37;
    list *plist = get_paths("/data/captcha/valid.list");
    char **paths = (char **)list_to_array(plist);
    data valid = load_data_captcha(paths, imgs, 0, 10, 60, 200);
    translate_data_rows(valid, -128);
    scale_data_rows(valid, 1./128);
    matrix pred = network_predict_data(net, valid);
    int i, k;
    int correct = 0;
    int total = 0;
    int accuracy = 0;
    for(i = 0; i < imgs; ++i){
        int allcorrect = 1;
        for(k = 0; k < 10; ++k){
            char truth = int_to_alphanum(max_index(valid.y.vals[i]+k*numchars, numchars));
            char prediction = int_to_alphanum(max_index(pred.vals[i]+k*numchars, numchars));
            if (truth != prediction) allcorrect=0;
            if (truth != '.' && truth == prediction) ++correct;
            if (truth != '.' || truth != prediction) ++total;
        }
        accuracy += allcorrect;
    }
    printf("Word Accuracy: %f, Char Accuracy %f\n", (float)accuracy/imgs, (float)correct/total);
    free_data(valid);
}

void test_captcha(char *cfgfile, char *weightfile)
{
    srand(time(0));
    char *base = basename(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    set_batch_network(&net, 1);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    clock_t time;
    char filename[256];
    while(1){
        printf("Enter filename: ");
        fgets(filename, 256, stdin);
        strtok(filename, "\n");
        time = clock();
        image im = load_image_color(filename, 60, 200);
        translate_image(im, -128);
        scale_image(im, 1/128.);
        float *X = im.data;
        time=clock();
        float *predictions = network_predict(net, X);
        printf("Predicted in %f\n", sec(clock() - time));
        print_letters(predictions, 10);
        free_image(im);
    }
}

void train_imagenet(char *cfgfile, char *weightfile)
{
    float avg_loss = -1;
    srand(time(0));
    char *base = basename(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = 1024;
    int i = net.seen/imgs;
    char **labels = get_labels("/home/pjreddie/data/imagenet/cls.labels.list");
    list *plist = get_paths("/data/imagenet/cls.train.list");
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    clock_t time;
    pthread_t load_thread;
    data train;
    data buffer;
    load_thread = load_data_thread(paths, imgs, plist->size, labels, 1000, 256, 256, &buffer);
    while(1){
        ++i;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_thread(paths, imgs, plist->size, labels, 1000, 256, 256, &buffer);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
        net.seen += imgs;
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock()-time), net.seen);
        free_data(train);
        if(i%100==0){
            char buff[256];
            sprintf(buff, "/home/pjreddie/imagenet_backup/%s_%d.weights",base, i);
            save_weights(net, buff);
        }
    }
}

void validate_imagenet(char *filename, char *weightfile)
{
    int i = 0;
    network net = parse_network_cfg(filename);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    srand(time(0));

    char **labels = get_labels("/home/pjreddie/data/imagenet/cls.val.labels.list");

    list *plist = get_paths("/data/imagenet/cls.val.list");
    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    clock_t time;
    float avg_acc = 0;
    float avg_top5 = 0;
    int splits = 50;
    int num = (i+1)*m/splits - i*m/splits;

    data val, buffer;
    pthread_t load_thread = load_data_thread(paths, num, 0, labels, 1000, 256, 256, &buffer);
    for(i = 1; i <= splits; ++i){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        num = (i+1)*m/splits - i*m/splits;
        char **part = paths+(i*m/splits);
        if(i != splits) load_thread = load_data_thread(part, num, 0, labels, 1000, 256, 256, &buffer);
        printf("Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
        float *acc = network_accuracies(net, val);
        avg_acc += acc[0];
        avg_top5 += acc[1];
        printf("%d: top1: %f, top5: %f, %lf seconds, %d images\n", i, avg_acc/i, avg_top5/i, sec(clock()-time), val.X.rows);
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

void test_init(char *cfgfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    set_batch_network(&net, 1);
    srand(2222222);
    int i = 0;
    char *filename = "data/test.jpg";

    image im = load_image_color(filename, 256, 256);
    //z_normalize_image(im);
    translate_image(im, -128);
    scale_image(im, 1/128.);
    float *X = im.data;
    forward_network(net, X, 0, 1);
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            image output = get_convolutional_image(layer);
            int size = output.h*output.w*output.c;
            float v = variance_array(layer.output, size);
            float m = mean_array(layer.output, size);
            printf("%d: Convolutional, mean: %f, variance %f\n", i, m, v);
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            int size = layer.outputs;
            float v = variance_array(layer.output, size);
            float m = mean_array(layer.output, size);
            printf("%d: Connected, mean: %f, variance %f\n", i, m, v);
        }
    }
    free_image(im);
}
void test_dog(char *cfgfile)
{
    image im = load_image_color("data/dog.jpg", 256, 256);
    translate_image(im, -128);
    print_image(im);
    float *X = im.data;
    network net = parse_network_cfg(cfgfile);
    set_batch_network(&net, 1);
    network_predict(net, X);
    image crop = get_network_image_layer(net, 0);
    show_image(crop, "cropped");
    print_image(crop);
    show_image(im, "orig");
    float * inter = get_network_output(net);
    pm(1000, 1, inter);
    cvWaitKey(0);
}

void test_voc_segment(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    while(1){
        char filename[256];
        fgets(filename, 256, stdin);
        strtok(filename, "\n");
        image im = load_image_color(filename, 500, 500);
        //resize_network(net, im.h, im.w, im.c);
        translate_image(im, -128);
        scale_image(im, 1/128.);
        //float *predictions = network_predict(net, im.data);
        network_predict(net, im.data);
        free_image(im);
        image output = get_network_image_layer(net, net.n-2);
        show_image(output, "Segment Output");
        cvWaitKey(0);
    }
}

void test_imagenet(char *cfgfile)
{
    network net = parse_network_cfg(cfgfile);
    set_batch_network(&net, 1);
    //imgs=1;
    srand(2222222);
    int i = 0;
    char **names = get_labels("cfg/shortnames.txt");
    clock_t time;
    char filename[256];
    int indexes[10];
    while(1){
        fgets(filename, 256, stdin);
        strtok(filename, "\n");
        image im = load_image_color(filename, 256, 256);
        translate_image(im, -128);
        scale_image(im, 1/128.);
        printf("%d %d %d\n", im.h, im.w, im.c);
        float *X = im.data;
        time=clock();
        float *predictions = network_predict(net, X);
        top_predictions(net, 10, indexes);
        printf("%s: Predicted in %f seconds.\n", filename, sec(clock()-time));
        for(i = 0; i < 10; ++i){
            int index = indexes[i];
            printf("%s: %f\n", names[index], predictions[index]);
        }
        free_image(im);
    }
}

void test_visualize(char *filename)
{
    network net = parse_network_cfg(filename);
    visualize_network(net);
    cvWaitKey(0);
}

void test_cifar10(char *cfgfile)
{
    network net = parse_network_cfg(cfgfile);
    data test = load_cifar10_data("data/cifar10/test_batch.bin");
    clock_t start = clock(), end;
    float test_acc = network_accuracy_multi(net, test, 10);
    end = clock();
    printf("%f in %f Sec\n", test_acc, sec(end-start));
    //visualize_network(net);
    //cvWaitKey(0);
}

void train_cifar10(char *cfgfile)
{
    srand(555555);
    srand(time(0));
    network net = parse_network_cfg(cfgfile);
    data test = load_cifar10_data("data/cifar10/test_batch.bin");
    int count = 0;
    int iters = 50000/net.batch;
    data train = load_all_cifar10();
    while(++count <= 10000){
        clock_t time = clock();
        float loss = train_network_sgd(net, train, iters);

        if(count%10 == 0){
            float test_acc = network_accuracy(net, test);
            printf("%d: Loss: %f, Test Acc: %f, Time: %lf seconds\n", count, loss, test_acc,sec(clock()-time));
            char buff[256];
            sprintf(buff, "/home/pjreddie/imagenet_backup/cifar10_%d.cfg", count);
            save_network(net, buff);
        }else{
            printf("%d: Loss: %f, Time: %lf seconds\n", count, loss, sec(clock()-time));
        }

    }
    free_data(train);
}

void compare_nist(char *p1,char *p2)
{
    srand(222222);
    network n1 = parse_network_cfg(p1);
    network n2 = parse_network_cfg(p2);
    data test = load_categorical_data_csv("data/mnist/mnist_test.csv",0,10);
    normalize_data_rows(test);
    compare_networks(n1, n2, test);
}

void test_nist(char *path)
{
    srand(222222);
    network net = parse_network_cfg(path);
    data test = load_categorical_data_csv("data/mnist/mnist_test.csv",0,10);
    normalize_data_rows(test);
    clock_t start = clock(), end;
    float test_acc = network_accuracy(net, test);
    end = clock();
    printf("Accuracy: %f, Time: %lf seconds\n", test_acc,(float)(end-start)/CLOCKS_PER_SEC);
}

void train_nist(char *cfgfile)
{
    srand(222222);
    // srand(time(0));
    data train = load_categorical_data_csv("data/mnist/mnist_train.csv", 0, 10);
    data test = load_categorical_data_csv("data/mnist/mnist_test.csv",0,10);
    network net = parse_network_cfg(cfgfile);
    int count = 0;
    int iters = 6000/net.batch + 1;
    while(++count <= 100){
        clock_t start = clock(), end;
        normalize_data_rows(train);
        normalize_data_rows(test);
        float loss = train_network_sgd(net, train, iters);
        float test_acc = 0;
        if(count%1 == 0) test_acc = network_accuracy(net, test);
        end = clock();
        printf("%d: Loss: %f, Test Acc: %f, Time: %lf seconds\n", count, loss, test_acc,(float)(end-start)/CLOCKS_PER_SEC);
    }
    free_data(train);
    free_data(test);
    char buff[256];
    sprintf(buff, "%s.trained", cfgfile);
    save_network(net, buff);
}

/*
   void train_nist_distributed(char *address)
   {
   srand(time(0));
   network net = parse_network_cfg("cfg/nist.client");
   data train = load_categorical_data_csv("data/mnist/mnist_train.csv", 0, 10);
//data test = load_categorical_data_csv("data/mnist/mnist_test.csv",0,10);
normalize_data_rows(train);
//normalize_data_rows(test);
int count = 0;
int iters = 50000/net.batch;
iters = 1000/net.batch + 1;
while(++count <= 2000){
clock_t start = clock(), end;
float loss = train_network_sgd(net, train, iters);
client_update(net, address);
end = clock();
//float test_acc = network_accuracy_gpu(net, test);
//float test_acc = 0;
printf("%d: Loss: %f, Time: %lf seconds\n", count, loss, (float)(end-start)/CLOCKS_PER_SEC);
}
}
 */

void test_ensemble()
{
    int i;
    srand(888888);
    data d = load_categorical_data_csv("mnist/mnist_train.csv", 0, 10);
    normalize_data_rows(d);
    data test = load_categorical_data_csv("mnist/mnist_test.csv", 0,10);
    normalize_data_rows(test);
    data train = d;
    //   data *split = split_data(d, 1, 10);
    //   data train = split[0];
    //   data test = split[1];
    matrix prediction = make_matrix(test.y.rows, test.y.cols);
    int n = 30;
    for(i = 0; i < n; ++i){
        int count = 0;
        float lr = .0005;
        float momentum = .9;
        float decay = .01;
        network net = parse_network_cfg("nist.cfg");
        while(++count <= 15){
            float acc = train_network_sgd(net, train, train.X.rows);
            printf("Training Accuracy: %lf Learning Rate: %f Momentum: %f Decay: %f\n", acc, lr, momentum, decay );
            lr /= 2; 
        }
        matrix partial = network_predict_data(net, test);
        float acc = matrix_topk_accuracy(test.y, partial,1);
        printf("Model Accuracy: %lf\n", acc);
        matrix_add_matrix(partial, prediction);
        acc = matrix_topk_accuracy(test.y, prediction,1);
        printf("Current Ensemble Accuracy: %lf\n", acc);
        free_matrix(partial);
    }
    float acc = matrix_topk_accuracy(test.y, prediction,1);
    printf("Full Ensemble Accuracy: %lf\n", acc);
}

void visualize_cat()
{
    network net = parse_network_cfg("cfg/voc_imagenet.cfg");
    image im = load_image_color("data/cat.png", 0, 0);
    printf("Processing %dx%d image\n", im.h, im.w);
    resize_network(net, im.h, im.w, im.c);
    forward_network(net, im.data, 0, 0);

    visualize_network(net);
    cvWaitKey(0);
}

void test_correct_nist()
{
    network net = parse_network_cfg("cfg/nist_conv.cfg");
    srand(222222);
    net = parse_network_cfg("cfg/nist_conv.cfg");
    data train = load_categorical_data_csv("data/mnist/mnist_train.csv", 0, 10);
    data test = load_categorical_data_csv("data/mnist/mnist_test.csv",0,10);
    normalize_data_rows(train);
    normalize_data_rows(test);
    int count = 0;
    int iters = 1000/net.batch;

    while(++count <= 5){
        clock_t start = clock(), end;
        float loss = train_network_sgd(net, train, iters);
        end = clock();
        float test_acc = network_accuracy(net, test);
        printf("%d: Loss: %f, Test Acc: %f, Time: %lf seconds, LR: %f, Momentum: %f, Decay: %f\n", count, loss, test_acc,(float)(end-start)/CLOCKS_PER_SEC, net.learning_rate, net.momentum, net.decay);
    }
    save_network(net, "cfg/nist_gpu.cfg");

    gpu_index = -1;
    count = 0;
    srand(222222);
    net = parse_network_cfg("cfg/nist_conv.cfg");
    while(++count <= 5){
        clock_t start = clock(), end;
        float loss = train_network_sgd(net, train, iters);
        end = clock();
        float test_acc = network_accuracy(net, test);
        printf("%d: Loss: %f, Test Acc: %f, Time: %lf seconds, LR: %f, Momentum: %f, Decay: %f\n", count, loss, test_acc,(float)(end-start)/CLOCKS_PER_SEC, net.learning_rate, net.momentum, net.decay);
    }
    save_network(net, "cfg/nist_cpu.cfg");
}

void test_correct_alexnet()
{
    char **labels = get_labels("/home/pjreddie/data/imagenet/cls.labels.list");
    list *plist = get_paths("/data/imagenet/cls.train.list");
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    clock_t time;
    int count = 0;
    network net;

    srand(222222);
    net = parse_network_cfg("cfg/net.cfg");
    int imgs = net.batch;

    count = 0;
    while(++count <= 5){
        time=clock();
        data train = load_data(paths, imgs, plist->size, labels, 1000, 256, 256);
        normalize_data_rows(train);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
        printf("%d: %f, %lf seconds, %d images\n", count, loss, sec(clock()-time), imgs*net.batch);
        free_data(train);
    }

    gpu_index = -1;
    count = 0;
    srand(222222);
    net = parse_network_cfg("cfg/net.cfg");
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    while(++count <= 5){
        time=clock();
        data train = load_data(paths, imgs, plist->size, labels, 1000, 256,256);
        normalize_data_rows(train);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        float loss = train_network(net, train);
        printf("%d: %f, %lf seconds, %d images\n", count, loss, sec(clock()-time), imgs*net.batch);
        free_data(train);
    }
}

/*
   void run_server()
   {
   srand(time(0));
   network net = parse_network_cfg("cfg/net.cfg");
   set_batch_network(&net, 1);
   server_update(net);
   }

   void test_client()
   {
   network net = parse_network_cfg("cfg/alexnet.client");
   clock_t time=clock();
   client_update(net, "localhost");
   printf("1\n");
   client_update(net, "localhost");
   printf("2\n");
   client_update(net, "localhost");
   printf("3\n");
   printf("Transfered: %lf seconds\n", sec(clock()-time));
   }
 */

void del_arg(int argc, char **argv, int index)
{
    int i;
    for(i = index; i < argc-1; ++i) argv[i] = argv[i+1];
    argv[i] = 0;
}

int find_arg(int argc, char* argv[], char *arg)
{
    int i;
    for(i = 0; i < argc; ++i) {
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)) {
            del_arg(argc, argv, i);
            return 1;
        }
    }
    return 0;
}

int find_int_arg(int argc, char **argv, char *arg, int def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = atoi(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

void scale_rate(char *filename, float scale)
{
    // Ready for some weird shit??
    FILE *fp = fopen(filename, "r+b");
    if(!fp) file_error(filename);
    float rate = 0;
    fread(&rate, sizeof(float), 1, fp);
    printf("Scaling learning rate from %f to %f\n", rate, rate*scale);
    rate = rate*scale;
    fseek(fp, 0, SEEK_SET);
    fwrite(&rate, sizeof(float), 1, fp);
    fclose(fp);
}

int main(int argc, char **argv)
{
    //test_convolutional_layer();
    if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    gpu_index = find_int_arg(argc, argv, "-i", 0);
    if(find_arg(argc, argv, "-nogpu")) gpu_index = -1;

#ifndef GPU
    gpu_index = -1;
#else
    if(gpu_index >= 0){
        cudaSetDevice(gpu_index);
    }
#endif

    if(0==strcmp(argv[1], "test_correct")) test_correct_alexnet();
    else if(0==strcmp(argv[1], "test_correct_nist")) test_correct_nist();
    //else if(0==strcmp(argv[1], "server")) run_server();

#ifdef GPU
    else if(0==strcmp(argv[1], "test_gpu")) test_gpu_blas();
#endif

    else if(argc < 3){
        fprintf(stderr, "usage: %s <function> <filename>\n", argv[0]);
        return 0;
    }
    else if(0==strcmp(argv[1], "detection")) train_detection_net(argv[2], (argc > 3)? argv[3] : 0);
    else if(0==strcmp(argv[1], "test")) test_imagenet(argv[2]);
    else if(0==strcmp(argv[1], "dog")) test_dog(argv[2]);
    else if(0==strcmp(argv[1], "ctrain")) train_cifar10(argv[2]);
    else if(0==strcmp(argv[1], "nist")) train_nist(argv[2]);
    else if(0==strcmp(argv[1], "ctest")) test_cifar10(argv[2]);
    else if(0==strcmp(argv[1], "train")) train_imagenet(argv[2], (argc > 3)? argv[3] : 0);
    else if(0==strcmp(argv[1], "captcha")) train_captcha(argv[2], (argc > 3)? argv[3] : 0);
    else if(0==strcmp(argv[1], "tcaptcha")) test_captcha(argv[2], (argc > 3)? argv[3] : 0);
    else if(0==strcmp(argv[1], "vcaptcha")) validate_captcha(argv[2], (argc > 3)? argv[3] : 0);
    else if(0==strcmp(argv[1], "testseg")) test_voc_segment(argv[2], (argc > 3)? argv[3] : 0);
    //else if(0==strcmp(argv[1], "client")) train_imagenet_distributed(argv[2]);
    else if(0==strcmp(argv[1], "detect")) test_detection(argv[2], (argc > 3)? argv[3] : 0);
    else if(0==strcmp(argv[1], "init")) test_init(argv[2]);
    else if(0==strcmp(argv[1], "visualize")) test_visualize(argv[2]);
    else if(0==strcmp(argv[1], "valid")) validate_imagenet(argv[2], (argc > 3)? argv[3] : 0);
    else if(0==strcmp(argv[1], "testnist")) test_nist(argv[2]);
    else if(0==strcmp(argv[1], "validetect")) validate_detection_net(argv[2], (argc > 3)? argv[3] : 0);
    else if(argc < 4){
        fprintf(stderr, "usage: %s <function> <filename> <filename>\n", argv[0]);
        return 0;
    }
    else if(0==strcmp(argv[1], "compare")) compare_nist(argv[2], argv[3]);
    else if(0==strcmp(argv[1], "convert")) convert(argv[2], argv[3], (argc > 4)? argv[4] : 0);
    else if(0==strcmp(argv[1], "scale")) scale_rate(argv[2], atof(argv[3]));
    fprintf(stderr, "Success!\n");
    return 0;
}

