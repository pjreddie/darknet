#include "connected_layer.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "network.h"
#include "image.h"

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
    convolutional_layer layer = make_convolutional_layer(dog.h, dog.w, dog.c, n, size, stride);
    char buff[256];
    for(i = 0; i < n; ++i) {
        sprintf(buff, "Kernel %d", i);
        show_image(layer.kernels[i], buff);
    }
    run_convolutional_layer(dog, layer);
    
    maxpool_layer mlayer = make_maxpool_layer(layer.output.h, layer.output.w, layer.output.c, 2);
    run_maxpool_layer(layer.output,mlayer);

    show_image_layers(mlayer.output, "Test Maxpool Layer");
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

void test_network()
{
    network net;
    net.n = 11;
    net.layers = calloc(net.n, sizeof(void *));
    net.types = calloc(net.n, sizeof(LAYER_TYPE));
    net.types[0] = CONVOLUTIONAL;
    net.types[1] = MAXPOOL;
    net.types[2] = CONVOLUTIONAL;
    net.types[3] = MAXPOOL;
    net.types[4] = CONVOLUTIONAL;
    net.types[5] = CONVOLUTIONAL;
    net.types[6] = CONVOLUTIONAL;
    net.types[7] = MAXPOOL;
    net.types[8] = CONNECTED;
    net.types[9] = CONNECTED;
    net.types[10] = CONNECTED;

    image dog = load_image("test_hinton.jpg");

    int n = 48;
    int stride = 4;
    int size = 11;
    convolutional_layer cl = make_convolutional_layer(dog.h, dog.w, dog.c, n, size, stride);
    maxpool_layer ml = make_maxpool_layer(cl.output.h, cl.output.w, cl.output.c, 2);

    n = 128;
    size = 5;
    stride = 1;
    convolutional_layer cl2 = make_convolutional_layer(ml.output.h, ml.output.w, ml.output.c, n, size, stride);
    maxpool_layer ml2 = make_maxpool_layer(cl2.output.h, cl2.output.w, cl2.output.c, 2);

    n = 192;
    size = 3;
    convolutional_layer cl3 = make_convolutional_layer(ml2.output.h, ml2.output.w, ml2.output.c, n, size, stride);
    convolutional_layer cl4 = make_convolutional_layer(cl3.output.h, cl3.output.w, cl3.output.c, n, size, stride);
    n = 128;
    convolutional_layer cl5 = make_convolutional_layer(cl4.output.h, cl4.output.w, cl4.output.c, n, size, stride);
    maxpool_layer ml3 = make_maxpool_layer(cl5.output.h, cl5.output.w, cl5.output.c, 4);
    connected_layer nl = make_connected_layer(ml3.output.h*ml3.output.w*ml3.output.c, 4096, RELU);
    connected_layer nl2 = make_connected_layer(4096, 4096, RELU);
    connected_layer nl3 = make_connected_layer(4096, 1000, RELU);

    net.layers[0] = &cl;
    net.layers[1] = &ml;
    net.layers[2] = &cl2;
    net.layers[3] = &ml2;
    net.layers[4] = &cl3;
    net.layers[5] = &cl4;
    net.layers[6] = &cl5;
    net.layers[7] = &ml3;
    net.layers[8] = &nl;
    net.layers[9] = &nl2;
    net.layers[10] = &nl3;

    int i;
    clock_t start = clock(), end;
    for(i = 0; i < 10; ++i){
        run_network(dog, net);
        rotate_image(dog);
    }
    end = clock();
    printf("Ran %lf second per iteration\n", (double)(end-start)/CLOCKS_PER_SEC/10);

    show_image_layers(get_network_image(net), "Test Network Layer");
}

void test_backpropagate()
{
    int n = 3;
    int size = 4;
    int stride = 10;
    image dog = load_image("dog.jpg");
    show_image(dog, "Test Backpropagate Input");
    image dog_copy = copy_image(dog);
    convolutional_layer cl = make_convolutional_layer(dog.h, dog.w, dog.c, n, size, stride);
    run_convolutional_layer(dog, cl);
    show_image(cl.output, "Test Backpropagate Output");
    int i;
    clock_t start = clock(), end;
    for(i = 0; i < 100; ++i){
        backpropagate_convolutional_layer(dog_copy, cl);
    }
    end = clock();
    printf("Backpropagate: %lf seconds\n", (double)(end-start)/CLOCKS_PER_SEC);
    start = clock();
    for(i = 0; i < 100; ++i){
        backpropagate_convolutional_layer_convolve(dog, cl);
    }
    end = clock();
    printf("Backpropagate Using Convolutions: %lf seconds\n", (double)(end-start)/CLOCKS_PER_SEC);
    show_image(dog_copy, "Test Backpropagate 1");
    show_image(dog, "Test Backpropagate 2");
    subtract_image(dog, dog_copy);
    show_image(dog, "Test Backpropagate Difference");
}

void test_ann()
{
    network net;
    net.n = 3;
    net.layers = calloc(net.n, sizeof(void *));
    net.types = calloc(net.n, sizeof(LAYER_TYPE));
    net.types[0] = CONNECTED;
    net.types[1] = CONNECTED;
    net.types[2] = CONNECTED;

    connected_layer nl = make_connected_layer(1, 20, RELU);
    connected_layer nl2 = make_connected_layer(20, 20, RELU);
    connected_layer nl3 = make_connected_layer(20, 1, RELU);

    net.layers[0] = &nl;
    net.layers[1] = &nl2;
    net.layers[2] = &nl3;

    image t = make_image(1,1,1);
    int count = 0;
        
    double avgerr = 0;
    while(1){
        double v = ((double)rand()/RAND_MAX);
        double truth = v*v;
        set_pixel(t,0,0,0,v);
        run_network(t, net);
        double *out = get_network_output(net);
        double err = pow((out[0]-truth),2.);
        avgerr = .99 * avgerr + .01 * err;
        //if(++count % 100000 == 0) printf("%f\n", avgerr);
        if(++count % 100000 == 0) printf("%f %f :%f AVG %f \n", truth, out[0], err, avgerr);
        out[0] = truth - out[0];
        learn_network(t, net);
        update_network(net, .001);
    }

}

int main()
{
    //test_backpropagate();
    test_ann();
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
