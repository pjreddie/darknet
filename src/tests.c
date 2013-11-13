#include "connected_layer.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "network.h"
#include "image.h"
#include "parser.h"
#include "data.h"
#include "matrix.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

void test_convolve()
{
    image dog = load_image("dog.jpg");
    //show_image_layers(dog, "Dog");
    printf("dog channels %d\n", dog.c);
    image kernel = make_random_image(3,3,dog.c);
    image edge = make_image(dog.h, dog.w, 1);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i < 1000; ++i){
        convolve(dog, kernel, 1, 0, edge);
    }
    end = clock();
    printf("Convolutions: %lf seconds\n", (double)(end-start)/CLOCKS_PER_SEC);
    show_image_layers(edge, "Test Convolve");
}

void test_color()
{
    image dog = load_image("test_color.png");
    show_image_layers(dog, "Test Color");
}

void test_convolutional_layer()
{
    srand(0);
    image dog = load_image("dog.jpg");
    int i;
    int n = 3;
    int stride = 1;
    int size = 3;
    convolutional_layer layer = *make_convolutional_layer(dog.h, dog.w, dog.c, n, size, stride, RELU);
    char buff[256];
    for(i = 0; i < n; ++i) {
        sprintf(buff, "Kernel %d", i);
        show_image(layer.kernels[i], buff);
    }
    forward_convolutional_layer(layer, dog.data);
    
    image output = get_convolutional_image(layer);
    maxpool_layer mlayer = *make_maxpool_layer(output.h, output.w, output.c, 2);
    forward_maxpool_layer(mlayer, layer.output);

    show_image_layers(get_maxpool_image(mlayer), "Test Maxpool Layer");
}

void test_load()
{
    image dog = load_image("dog.jpg");
    show_image(dog, "Test Load");
    show_image_layers(dog, "Test Load");
}
void test_upsample()
{
    image dog = load_image("dog.jpg");
    int n = 3;
    image up = make_image(n*dog.h, n*dog.w, dog.c);
    upsample_image(dog, n, up);
    show_image(up, "Test Upsample");
    show_image_layers(up, "Test Upsample");
}

void test_rotate()
{
    int i;
    image dog = load_image("dog.jpg");
    clock_t start = clock(), end;
    for(i = 0; i < 1001; ++i){
        rotate_image(dog);
    }
    end = clock();
    printf("Rotations: %lf seconds\n", (double)(end-start)/CLOCKS_PER_SEC);
    show_image(dog, "Test Rotate");

    image random = make_random_image(3,3,3);
    show_image(random, "Test Rotate Random");
    rotate_image(random);
    show_image(random, "Test Rotate Random");
    rotate_image(random);
    show_image(random, "Test Rotate Random");
}

void test_parser()
{
    network net = parse_network_cfg("test_parser.cfg");
    double input[1];
    int count = 0;
        
    double avgerr = 0;
    while(1){
        double v = ((double)rand()/RAND_MAX);
        double truth = v*v;
        input[0] = v;
        forward_network(net, input);
        double *out = get_network_output(net);
        double *delta = get_network_delta(net);
        double err = pow((out[0]-truth),2.);
        avgerr = .99 * avgerr + .01 * err;
        //if(++count % 100000 == 0) printf("%f\n", avgerr);
        if(++count % 1000000 == 0) printf("%f %f :%f AVG %f \n", truth, out[0], err, avgerr);
        delta[0] = truth - out[0];
        learn_network(net, input);
        update_network(net, .001);
    }
}

void test_data()
{
    batch train = random_batch("train_paths.txt", 101);
    show_image(train.images[0], "Test Data Loading");
    show_image(train.images[100], "Test Data Loading");
    show_image(train.images[10], "Test Data Loading");
    free_batch(train);
}

void test_train()
{
    network net = parse_network_cfg("test.cfg");
    srand(0);
    //visualize_network(net);
    int i = 1000;
    //while(1){
    while(i > 0){
        batch train = random_batch("train_paths.txt", 100);
        train_network_batch(net, train);
        //show_image_layers(get_network_image(net), "hey");
        //visualize_network(net);
        //cvWaitKey(0);
        free_batch(train);
        --i;
        }
    //}
}

double error_network(network net, matrix m, double *truth)
{
    int i;
    int correct = 0;
    for(i = 0; i < m.rows; ++i){
        forward_network(net, m.vals[i]);
        double *out = get_network_output(net);
        double err = truth[i] - out[0];
        if(fabs(err) < .5) ++correct;
    }
    return (double)correct/m.rows;
}

void classify_random_filters()
{
    network net = parse_network_cfg("random_filter_finish.cfg");
    matrix m = csv_to_matrix("train.csv");
    matrix ho = hold_out_matrix(&m, 2500);
    double *truth = pop_column(&m, 0);
    double *ho_truth = pop_column(&ho, 0);
    int i;
    clock_t start = clock(), end;
    int count = 0;
    while(++count <= 300){
        for(i = 0; i < m.rows; ++i){
            int index = rand()%m.rows;
            //image p = double_to_image(1690,1,1,m.vals[index]);
            //normalize_image(p);
            forward_network(net, m.vals[index]);
            double *out = get_network_output(net);
            double *delta = get_network_delta(net);
            //printf("%f\n", out[0]);
            delta[0] = truth[index] - out[0];
           // printf("%f\n", delta[0]);
            //printf("%f %f\n", truth[index], out[0]);
            learn_network(net, m.vals[index]);
            update_network(net, .000005);
        }
        double test_acc = error_network(net, m, truth);
        double valid_acc = error_network(net, ho, ho_truth);
        printf("%f, %f\n", test_acc, valid_acc);
        fprintf(stderr, "%5d: %f Valid: %f\n",count, test_acc, valid_acc);
        //if(valid_acc > .70) break;
    }
    end = clock();
    FILE *fp = fopen("submission/out.txt", "w");
    matrix test = csv_to_matrix("test.csv");
    truth = pop_column(&test, 0);
    for(i = 0; i < test.rows; ++i){
        forward_network(net, test.vals[i]);
        double *out = get_network_output(net);
        if(fabs(out[0]) < .5) fprintf(fp, "0\n");
        else fprintf(fp, "1\n");
    }
    fclose(fp);
    printf("Neural Net Learning: %lf seconds\n", (double)(end-start)/CLOCKS_PER_SEC);
}

void test_random_filters()
{
    FILE *file = fopen("test.csv", "w");
    int i,j,k;
    srand(0);
    network net = parse_network_cfg("test_random_filter.cfg");
    for(i = 0; i < 100; ++i){
        printf("%d\n", i);
        batch part = get_batch("test_paths.txt", i, 100);
        for(j = 0; j < part.n; ++j){
            forward_network(net, part.images[j].data);
            double *out = get_network_output(net);
            fprintf(file, "%f", part.truth[j][0]);
            for(k = 0; k < get_network_output_size(net); ++k){
                fprintf(file, ",%f", out[k]);
            }
            fprintf(file, "\n");
        }
        free_batch(part);
    }
}

int main()
{
    //classify_random_filters();
    //test_random_filters();
    test_train();
    //test_parser();
    //test_backpropagate();
    //test_ann();
    //test_convolve();
    //test_upsample();
    //test_rotate();
    //test_load();
    //test_network();
    //test_convolutional_layer();
    //test_color();
    cvWaitKey(0);
    return 0;
}
