#include "connected_layer.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "network.h"
#include "image.h"
#include "parser.h"
#include "data.h"
#include "matrix.h"
#include "utils.h"
#include "mini_blas.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#define _GNU_SOURCE
#include <fenv.h>

void test_convolve()
{
    image dog = load_image("dog.jpg",300,400);
    printf("dog channels %d\n", dog.c);
    image kernel = make_random_image(3,3,dog.c);
    image edge = make_image(dog.h, dog.w, 1);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i < 1000; ++i){
        convolve(dog, kernel, 1, 0, edge, 1);
    }
    end = clock();
    printf("Convolutions: %lf seconds\n", (float)(end-start)/CLOCKS_PER_SEC);
    show_image_layers(edge, "Test Convolve");
}

#ifdef GPU

void test_convolutional_layer()
{
    int i;
    image dog = load_image("data/dog.jpg",224,224);
    network net = parse_network_cfg("cfg/convolutional.cfg");
    //    data test = load_cifar10_data("data/cifar10/test_batch.bin");
    //    float *X = calloc(net.batch*test.X.cols, sizeof(float));
    //    float *y = calloc(net.batch*test.y.cols, sizeof(float));
    int in_size = get_network_input_size(net)*net.batch;
    int del_size = get_network_output_size_layer(net, 0)*net.batch;
    int size = get_network_output_size(net)*net.batch;
    float *X = calloc(in_size, sizeof(float));
    float *y = calloc(size, sizeof(float));
    for(i = 0; i < in_size; ++i){
        X[i] = dog.data[i%get_network_input_size(net)];
    }
    //    get_batch(test, net.batch, X, y);
    clock_t start, end;
    cl_mem input_cl = cl_make_array(X, in_size);
    cl_mem truth_cl = cl_make_array(y, size);

    forward_network_gpu(net, input_cl, truth_cl, 1);
    start = clock();
    forward_network_gpu(net, input_cl, truth_cl, 1);
    end = clock();
    float gpu_sec = (float)(end-start)/CLOCKS_PER_SEC;
    printf("forward gpu: %f sec\n", gpu_sec);
    start = clock();
    backward_network_gpu(net, input_cl);
    end = clock();
    gpu_sec = (float)(end-start)/CLOCKS_PER_SEC;
    printf("backward gpu: %f sec\n", gpu_sec);
    //float gpu_cost = get_network_cost(net);
    float *gpu_out = calloc(size, sizeof(float));
    memcpy(gpu_out, get_network_output(net), size*sizeof(float));

    float *gpu_del = calloc(del_size, sizeof(float));
    memcpy(gpu_del, get_network_delta_layer(net, 0), del_size*sizeof(float));

    /*
       start = clock();
       forward_network(net, X, y, 1);
       backward_network(net, X);
       float cpu_cost = get_network_cost(net);
       end = clock();
       float cpu_sec = (float)(end-start)/CLOCKS_PER_SEC;
       float *cpu_out = calloc(size, sizeof(float));
       memcpy(cpu_out, get_network_output(net), size*sizeof(float));
       float *cpu_del = calloc(del_size, sizeof(float));
       memcpy(cpu_del, get_network_delta_layer(net, 0), del_size*sizeof(float));

       float sum = 0;
       float del_sum = 0;
       for(i = 0; i < size; ++i) sum += pow(gpu_out[i] - cpu_out[i], 2);
       for(i = 0; i < del_size; ++i) {
    //printf("%f %f\n", cpu_del[i], gpu_del[i]);
    del_sum += pow(cpu_del[i] - gpu_del[i], 2);
    }
    printf("GPU cost: %f, CPU cost: %f\n", gpu_cost, cpu_cost);
    printf("gpu: %f sec, cpu: %f sec, diff: %f, delta diff: %f, size: %d\n", gpu_sec, cpu_sec, sum, del_sum, size);
     */
}

void test_col2im()
{
    float col[] =  {1,2,1,2,
        1,2,1,2,
        1,2,1,2,
        1,2,1,2,
        1,2,1,2,
        1,2,1,2,
        1,2,1,2,
        1,2,1,2,
        1,2,1,2};
    float im[16] = {0};
    int batch = 1;
    int channels = 1;
    int height=4;
    int width=4;
    int ksize = 3;
    int stride = 1;
    int pad = 0;
    col2im_gpu(col, batch,
            channels,  height,  width,
            ksize,  stride, pad, im);
    int i;
    for(i = 0; i < 16; ++i)printf("%f,", im[i]);
    printf("\n");
    /*
       float data_im[] = {
       1,2,3,4,
       5,6,7,8,
       9,10,11,12
       };
       float data_col[18] = {0};
       im2col_cpu(data_im,  batch,
       channels,   height,  width,
       ksize,   stride,  pad, data_col) ;
       for(i = 0; i < 18; ++i)printf("%f,", data_col[i]);
       printf("\n");
     */
}

#endif

void test_convolve_matrix()
{
    image dog = load_image("dog.jpg",300,400);
    printf("dog channels %d\n", dog.c);

    int size = 11;
    int stride = 4;
    int n = 40;
    float *filters = make_random_image(size, size, dog.c*n).data;

    int mw = ((dog.h-size)/stride+1)*((dog.w-size)/stride+1);
    int mh = (size*size*dog.c);
    float *matrix = calloc(mh*mw, sizeof(float));

    image edge = make_image((dog.h-size)/stride+1, (dog.w-size)/stride+1, n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i < 1000; ++i){
        im2col_cpu(dog.data,1, dog.c,  dog.h,  dog.w,  size,  stride, 0, matrix);
        gemm(0,0,n,mw,mh,1,filters,mh,matrix,mw,1,edge.data,mw);
    }
    end = clock();
    printf("Convolutions: %lf seconds\n", (float)(end-start)/CLOCKS_PER_SEC);
    show_image_layers(edge, "Test Convolve");
    cvWaitKey(0);
}

void test_color()
{
    image dog = load_image("test_color.png", 300, 400);
    show_image_layers(dog, "Test Color");
}

void verify_convolutional_layer()
{
    srand(0);
    int i;
    int n = 1;
    int stride = 1;
    int size = 3;
    float eps = .00000001;
    image test = make_random_image(5,5, 1);
    convolutional_layer layer = *make_convolutional_layer(1,test.h,test.w,test.c, n, size, stride, 0, RELU,0,0,0);
    image out = get_convolutional_image(layer);
    float **jacobian = calloc(test.h*test.w*test.c, sizeof(float));

    forward_convolutional_layer(layer, test.data);
    image base = copy_image(out);

    for(i = 0; i < test.h*test.w*test.c; ++i){
        test.data[i] += eps;
        forward_convolutional_layer(layer, test.data);
        image partial = copy_image(out);
        subtract_image(partial, base);
        scale_image(partial, 1/eps);
        jacobian[i] = partial.data;
        test.data[i] -= eps;
    }
    float **jacobian2 = calloc(out.h*out.w*out.c, sizeof(float));
    image in_delta = make_image(test.h, test.w, test.c);
    image out_delta = get_convolutional_delta(layer);
    for(i = 0; i < out.h*out.w*out.c; ++i){
        out_delta.data[i] = 1;
        backward_convolutional_layer(layer, in_delta.data);
        image partial = copy_image(in_delta);
        jacobian2[i] = partial.data;
        out_delta.data[i] = 0;
    }
    int j;
    float *j1 = calloc(test.h*test.w*test.c*out.h*out.w*out.c, sizeof(float));
    float *j2 = calloc(test.h*test.w*test.c*out.h*out.w*out.c, sizeof(float));
    for(i = 0; i < test.h*test.w*test.c; ++i){
        for(j =0 ; j < out.h*out.w*out.c; ++j){
            j1[i*out.h*out.w*out.c + j] = jacobian[i][j];
            j2[i*out.h*out.w*out.c + j] = jacobian2[j][i];
            printf("%f %f\n", jacobian[i][j], jacobian2[j][i]);
        }
    }


    image mj1 = float_to_image(test.w*test.h*test.c, out.w*out.h*out.c, 1, j1);
    image mj2 = float_to_image(test.w*test.h*test.c, out.w*out.h*out.c, 1, j2);
    printf("%f %f\n", avg_image_layer(mj1,0), avg_image_layer(mj2,0));
    show_image(mj1, "forward jacobian");
    show_image(mj2, "backward jacobian");
}

void test_load()
{
    image dog = load_image("dog.jpg", 300, 400);
    show_image(dog, "Test Load");
    show_image_layers(dog, "Test Load");
}
void test_upsample()
{
    image dog = load_image("dog.jpg", 300, 400);
    int n = 3;
    image up = make_image(n*dog.h, n*dog.w, dog.c);
    upsample_image(dog, n, up);
    show_image(up, "Test Upsample");
    show_image_layers(up, "Test Upsample");
}

void test_rotate()
{
    int i;
    image dog = load_image("dog.jpg",300,400);
    clock_t start = clock(), end;
    for(i = 0; i < 1001; ++i){
        rotate_image(dog);
    }
    end = clock();
    printf("Rotations: %lf seconds\n", (float)(end-start)/CLOCKS_PER_SEC);
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
    network net = parse_network_cfg("cfg/trained_imagenet.cfg");
    save_network(net, "cfg/trained_imagenet_smaller.cfg");
}

void train_asirra()
{
    network net = parse_network_cfg("cfg/imagenet.cfg");
    int imgs = 1000/net.batch+1;
    //imgs = 1;
    srand(2222222);
    int i = 0;
    char *labels[] = {"cat","dog"};

    list *plist = get_paths("data/assira/train.list");
    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    clock_t time;

    while(1){
        i += 1;
        time=clock();
        data train = load_data_random(imgs*net.batch, paths, m, labels, 2, 256, 256);
        normalize_data_rows(train);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
        //float loss = train_network_data(net, train, imgs);
        float loss = 0;
        printf("%d: %f, Time: %lf seconds\n", i*net.batch*imgs, loss, sec(clock()-time));
        free_data(train);
        if(i%10==0){
            char buff[256];
            sprintf(buff, "cfg/asirra_backup_%d.cfg", i);
            save_network(net, buff);
        }
        //lr *= .99;
    }
}

void train_detection_net()
{
    float avg_loss = 1;
    //network net = parse_network_cfg("/home/pjreddie/imagenet_backup/alexnet_1270.cfg");
    network net = parse_network_cfg("cfg/detnet.cfg");
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = 1000/net.batch+1;
    srand(time(0));
    int i = 0;
    list *plist = get_paths("/home/pjreddie/data/imagenet/horse.txt");
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    clock_t time;
    while(1){
        i += 1;
        time=clock();
        data train = load_data_detection_random(imgs*net.batch, paths, plist->size, 256, 256, 8, 8, 256);
        //translate_data_rows(train, -144);
        normalize_data_rows(train);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
#ifdef GPU
        float loss = train_network_data_gpu(net, train, imgs);
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock()-time), i*imgs*net.batch);
#endif
        free_data(train);
        if(i%10==0){
            char buff[256];
            sprintf(buff, "/home/pjreddie/imagenet_backup/imagenet_%d.cfg", i);
            save_network(net, buff);
        }
    }
}


void train_imagenet()
{
    float avg_loss = 1;
    //network net = parse_network_cfg("/home/pjreddie/imagenet_backup/alexnet_1270.cfg");
    network net = parse_network_cfg("cfg/trained_alexnet.cfg");
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = 1000/net.batch+1;
    srand(time(0));
    int i = 0;
    char **labels = get_labels("/home/pjreddie/data/imagenet/cls.labels.list");
    list *plist = get_paths("/data/imagenet/cls.train.list");
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    clock_t time;
    while(1){
        i += 1;
        time=clock();
        data train = load_data_random(imgs*net.batch, paths, plist->size, labels, 1000, 256, 256);
        //translate_data_rows(train, -144);
        normalize_data_rows(train);
        printf("Loaded: %lf seconds\n", sec(clock()-time));
        time=clock();
#ifdef GPU
        float loss = train_network_data_gpu(net, train, imgs);
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%d: %f, %f avg, %lf seconds, %d images\n", i, loss, avg_loss, sec(clock()-time), i*imgs*net.batch);
#endif
        free_data(train);
        if(i%10==0){
            char buff[256];
            sprintf(buff, "/home/pjreddie/imagenet_backup/imagenet_%d.cfg", i);
            save_network(net, buff);
        }
    }
}

void validate_imagenet(char *filename)
{
    int i;
    network net = parse_network_cfg(filename);
    srand(time(0));

    char **labels = get_labels("/home/pjreddie/data/imagenet/cls.val.labels.list");

    list *plist = get_paths("/home/pjreddie/data/imagenet/cls.val.list");
    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    clock_t time;
    float avg_acc = 0;
    int splits = 50;

    for(i = 0; i < splits; ++i){
        time=clock();
        char **part = paths+(i*m/splits);
        int num = (i+1)*m/splits - i*m/splits;
        data val = load_data(part, num, labels, 1000, 256, 256);
        normalize_data_rows(val);
        printf("Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));
        time=clock();
#ifdef GPU
        float acc = network_accuracy_gpu(net, val);
        avg_acc += acc;
        printf("%d: %f, %f avg, %lf seconds, %d images\n", i, acc, avg_acc/(i+1), sec(clock()-time), val.X.rows);
#endif
        free_data(val);
    }
}

void test_detection()
{
    network net = parse_network_cfg("cfg/detnet_test.cfg");
    //imgs=1;
    srand(2222222);
    int i = 0;
    clock_t time;
    char filename[256];
    int indexes[10];
    while(1){
        fgets(filename, 256, stdin);
        image im = load_image_color(filename, 256, 256);
        z_normalize_image(im);
        printf("%d %d %d\n", im.h, im.w, im.c);
        float *X = im.data;
        time=clock();
        float *predictions = network_predict(net, X);
        top_predictions(net, 10, indexes);
        printf("%s: Predicted in %f seconds.\n", filename, sec(clock()-time));
        free_image(im);
    }
}

void test_imagenet()
{
    network net = parse_network_cfg("cfg/imagenet_test.cfg");
    //imgs=1;
    srand(2222222);
    int i = 0;
    char **names = get_labels("cfg/shortnames.txt");
    clock_t time;
    char filename[256];
    int indexes[10];
    while(1){
        fgets(filename, 256, stdin);
        image im = load_image_color(filename, 256, 256);
        z_normalize_image(im);
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

void test_cifar10()
{
    network net = parse_network_cfg("cfg/cifar10_part5.cfg");
    data test = load_cifar10_data("data/cifar10/test_batch.bin");
    clock_t start = clock(), end;
    float test_acc = network_accuracy(net, test);
    end = clock();
    printf("%f in %f Sec\n", test_acc, (float)(end-start)/CLOCKS_PER_SEC);
    visualize_network(net);
    cvWaitKey(0);
}

void train_cifar10()
{
    srand(555555);
    network net = parse_network_cfg("cfg/cifar10.cfg");
    data test = load_cifar10_data("data/cifar10/test_batch.bin");
    int count = 0;
    int iters = 10000/net.batch;
    data train = load_all_cifar10();
    while(++count <= 10000){
        clock_t start = clock(), end;
        float loss = train_network_sgd(net, train, iters);
        end = clock();
        //visualize_network(net);
        //cvWaitKey(5000);

        //float test_acc = network_accuracy(net, test);
        //printf("%d: Loss: %f, Test Acc: %f, Time: %lf seconds, LR: %f, Momentum: %f, Decay: %f\n", count, loss, test_acc,(float)(end-start)/CLOCKS_PER_SEC, net.learning_rate, net.momentum, net.decay);
        if(count%10 == 0){
            float test_acc = network_accuracy(net, test);
            printf("%d: Loss: %f, Test Acc: %f, Time: %lf seconds, LR: %f, Momentum: %f, Decay: %f\n", count, loss, test_acc,(float)(end-start)/CLOCKS_PER_SEC, net.learning_rate, net.momentum, net.decay);
            char buff[256];
            sprintf(buff, "/home/pjreddie/cifar/cifar10_2_%d.cfg", count);
            save_network(net, buff);
        }else{
            printf("%d: Loss: %f, Time: %lf seconds, LR: %f, Momentum: %f, Decay: %f\n", count, loss, (float)(end-start)/CLOCKS_PER_SEC, net.learning_rate, net.momentum, net.decay);
        }
    }
    free_data(train);
}

void test_vince()
{
    network net = parse_network_cfg("cfg/vince.cfg");
    data train = load_categorical_data_csv("images/vince.txt", 144, 2);
    normalize_data_rows(train);

    int count = 0;
    //float lr = .00005;
    //float momentum = .9;
    //float decay = 0.0001;
    //decay = 0;
    int batch = 10000;
    while(++count <= 10000){
        float loss = train_network_sgd(net, train, batch);
        printf("%5f %5f\n",(double)count*batch/train.X.rows, loss);
    }
}

void test_nist_single()
{
    srand(222222);
    network net = parse_network_cfg("cfg/nist_single.cfg");
    data train = load_categorical_data_csv("data/mnist/mnist_tiny.csv", 0, 10);
    normalize_data_rows(train);
    float loss = train_network_sgd(net, train, 1);
    printf("Loss: %f, LR: %f, Momentum: %f, Decay: %f\n", loss, net.learning_rate, net.momentum, net.decay);

}

void test_nist()
{
    srand(222222);
    network net = parse_network_cfg("cfg/nist_final.cfg");
    data test = load_categorical_data_csv("data/mnist/mnist_test.csv",0,10);
    translate_data_rows(test, -144);
    clock_t start = clock(), end;
    float test_acc = network_accuracy_multi(net, test,16);
    end = clock();
    printf("Accuracy: %f, Time: %lf seconds\n", test_acc,(float)(end-start)/CLOCKS_PER_SEC);
}

void train_nist()
{
    srand(222222);
    network net = parse_network_cfg("cfg/nist.cfg");
    data train = load_categorical_data_csv("data/mnist/mnist_train.csv", 0, 10);
    data test = load_categorical_data_csv("data/mnist/mnist_test.csv",0,10);
    translate_data_rows(train, -144);
    translate_data_rows(test, -144);
    int count = 0;
    int iters = 50000/net.batch;
    while(++count <= 2000){
        clock_t start = clock(), end;
        float loss = train_network_sgd(net, train, iters);
        end = clock();
        float test_acc = network_accuracy(net, test);
        printf("%d: Loss: %f, Test Acc: %f, Time: %lf seconds\n", count, loss, test_acc,(float)(end-start)/CLOCKS_PER_SEC);
    }
}

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
        float acc = matrix_accuracy(test.y, partial);
        printf("Model Accuracy: %lf\n", acc);
        matrix_add_matrix(partial, prediction);
        acc = matrix_accuracy(test.y, prediction);
        printf("Current Ensemble Accuracy: %lf\n", acc);
        free_matrix(partial);
    }
    float acc = matrix_accuracy(test.y, prediction);
    printf("Full Ensemble Accuracy: %lf\n", acc);
}

void test_random_classify()
{
    network net = parse_network_cfg("connected.cfg");
    matrix m = csv_to_matrix("train.csv");
    //matrix ho = hold_out_matrix(&m, 2500);
    float *truth = pop_column(&m, 0);
    //float *ho_truth = pop_column(&ho, 0);
    int i;
    clock_t start = clock(), end;
    int count = 0;
    while(++count <= 300){
        for(i = 0; i < m.rows; ++i){
            int index = rand()%m.rows;
            //image p = float_to_image(1690,1,1,m.vals[index]);
            //normalize_image(p);
            forward_network(net, m.vals[index], 0, 1);
            float *out = get_network_output(net);
            float *delta = get_network_delta(net);
            //printf("%f\n", out[0]);
            delta[0] = truth[index] - out[0];
            // printf("%f\n", delta[0]);
            //printf("%f %f\n", truth[index], out[0]);
            //backward_network(net, m.vals[index], );
            update_network(net);
        }
        //float test_acc = error_network(net, m, truth);
        //float valid_acc = error_network(net, ho, ho_truth);
        //printf("%f, %f\n", test_acc, valid_acc);
        //fprintf(stderr, "%5d: %f Valid: %f\n",count, test_acc, valid_acc);
        //if(valid_acc > .70) break;
    }
    end = clock();
    FILE *fp = fopen("submission/out.txt", "w");
    matrix test = csv_to_matrix("test.csv");
    truth = pop_column(&test, 0);
    for(i = 0; i < test.rows; ++i){
        forward_network(net, test.vals[i],0, 0);
        float *out = get_network_output(net);
        if(fabs(out[0]) < .5) fprintf(fp, "0\n");
        else fprintf(fp, "1\n");
    }
    fclose(fp);
    printf("Neural Net Learning: %lf seconds\n", (float)(end-start)/CLOCKS_PER_SEC);
}

void test_split()
{
    data train = load_categorical_data_csv("mnist/mnist_train.csv", 0, 10);
    data *split = split_data(train, 0, 13);
    printf("%d, %d, %d\n", train.X.rows, split[0].X.rows, split[1].X.rows);
}

void test_im2row()
{
    int h = 20;
    int w = 20;
    int c = 3;
    int stride = 1;
    int size = 11;
    image test = make_random_image(h,w,c);
    int mc = 1;
    int mw = ((h-size)/stride+1)*((w-size)/stride+1);
    int mh = (size*size*c);
    int msize = mc*mw*mh;
    float *matrix = calloc(msize, sizeof(float));
    int i;
    for(i = 0; i < 1000; ++i){
        im2col_cpu(test.data,1,  c,  h,  w,  size,  stride, 0, matrix);
        //image render = float_to_image(mh, mw, mc, matrix);
    }
}

void flip_network()
{
    network net = parse_network_cfg("cfg/voc_imagenet_orig.cfg");
    save_network(net, "cfg/voc_imagenet_rev.cfg");
}


void visualize_cat()
{
    network net = parse_network_cfg("cfg/voc_imagenet.cfg");
    image im = load_image("data/cat.png", 0, 0);
    printf("Processing %dx%d image\n", im.h, im.w);
    resize_network(net, im.h, im.w, im.c);
    forward_network(net, im.data, 0, 0);

    visualize_network(net);
    cvWaitKey(0);
}


void test_gpu_net()
{
    srand(222222);
    network net = parse_network_cfg("cfg/nist.cfg");
    data train = load_categorical_data_csv("data/mnist/mnist_train.csv", 0, 10);
    data test = load_categorical_data_csv("data/mnist/mnist_test.csv",0,10);
    translate_data_rows(train, -144);
    translate_data_rows(test, -144);
    int count = 0;
    int iters = 1000/net.batch;
    while(++count <= 5){
        clock_t start = clock(), end;
        float loss = train_network_sgd(net, train, iters);
        end = clock();
        float test_acc = network_accuracy(net, test);
        printf("%d: Loss: %f, Test Acc: %f, Time: %lf seconds, LR: %f, Momentum: %f, Decay: %f\n", count, loss, test_acc,(float)(end-start)/CLOCKS_PER_SEC, net.learning_rate, net.momentum, net.decay);
    }
#ifdef GPU
    count = 0;
    srand(222222);
    net = parse_network_cfg("cfg/nist.cfg");
    while(++count <= 5){
        clock_t start = clock(), end;
        float loss = train_network_sgd_gpu(net, train, iters);
        end = clock();
        float test_acc = network_accuracy(net, test);
        printf("%d: Loss: %f, Test Acc: %f, Time: %lf seconds, LR: %f, Momentum: %f, Decay: %f\n", count, loss, test_acc,(float)(end-start)/CLOCKS_PER_SEC, net.learning_rate, net.momentum, net.decay);
    }
#endif
}


int main(int argc, char *argv[])
{
    if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    if(0==strcmp(argv[1], "train")) train_imagenet();
    else if(0==strcmp(argv[1], "detection")) train_detection_net();
    else if(0==strcmp(argv[1], "asirra")) train_asirra();
    else if(0==strcmp(argv[1], "nist")) train_nist();
    else if(0==strcmp(argv[1], "test_correct")) test_gpu_net();
    else if(0==strcmp(argv[1], "test")) test_imagenet();
    else if(0==strcmp(argv[1], "visualize")) test_visualize(argv[2]);
    else if(0==strcmp(argv[1], "valid")) validate_imagenet(argv[2]);
#ifdef GPU
    else if(0==strcmp(argv[1], "test_gpu")) test_gpu_blas();
#endif
    fprintf(stderr, "Success!\n");
    return 0;
}

/*
void visualize_imagenet_topk(char *filename)
{
    int i,j,k,l;
    int topk = 10;
    network net = parse_network_cfg("cfg/voc_imagenet.cfg");
    list *plist = get_paths(filename);
    node *n = plist->front;
    int h = voc_size(1), w = voc_size(1);
    int num = get_network_image(net).c;
    image **vizs = calloc(num, sizeof(image*));
    float **score = calloc(num, sizeof(float *));
    for(i = 0; i < num; ++i){
        vizs[i] = calloc(topk, sizeof(image));
        for(j = 0; j < topk; ++j) vizs[i][j] = make_image(h,w,3);
        score[i] = calloc(topk, sizeof(float));
    }

    int count = 0;
    while(n){
        ++count;
        char *image_path = (char *)n->val;
        image im = load_image(image_path, 0, 0);
        n = n->next;
        if(im.h < 200 || im.w < 200) continue;
        printf("Processing %dx%d image\n", im.h, im.w);
        resize_network(net, im.h, im.w, im.c);
        //scale_image(im, 1./255);
        translate_image(im, -144);
        forward_network(net, im.data, 0, 0);
        image out = get_network_image(net);

        int dh = (im.h - h)/(out.h-1);
        int dw = (im.w - w)/(out.w-1);
        //printf("%d %d\n", dh, dw);
        for(k = 0; k < out.c; ++k){
            float topv = 0;
            int topi = -1;
            int topj = -1;
            for(i = 0; i < out.h; ++i){
                for(j = 0; j < out.w; ++j){
                    float val = get_pixel(out, i, j, k);
                    if(val > topv){
                        topv = val;
                        topi = i;
                        topj = j;
                    }
                }
            }
            if(topv){
                image sub = get_sub_image(im, dh*topi, dw*topj, h, w);
                for(l = 0; l < topk; ++l){
                    if(topv > score[k][l]){
                        float swap = score[k][l];
                        score[k][l] = topv;
                        topv = swap;

                        image swapi = vizs[k][l];
                        vizs[k][l] = sub;
                        sub = swapi;
                    }
                }
                free_image(sub);
            }
        }
        free_image(im);
        if(count%50 == 0){
            image grid = grid_images(vizs, num, topk);
            //show_image(grid, "IMAGENET Visualization");
            save_image(grid, "IMAGENET Grid Single Nonorm");
            free_image(grid);
        }
    }
    //cvWaitKey(0);
}

void visualize_imagenet_features(char *filename)
{
    int i,j,k;
    network net = parse_network_cfg("cfg/voc_imagenet.cfg");
    list *plist = get_paths(filename);
    node *n = plist->front;
    int h = voc_size(1), w = voc_size(1);
    int num = get_network_image(net).c;
    image *vizs = calloc(num, sizeof(image));
    for(i = 0; i < num; ++i) vizs[i] = make_image(h, w, 3);
    while(n){
        char *image_path = (char *)n->val;
        image im = load_image(image_path, 0, 0);
        printf("Processing %dx%d image\n", im.h, im.w);
        resize_network(net, im.h, im.w, im.c);
        forward_network(net, im.data, 0, 0);
        image out = get_network_image(net);

        int dh = (im.h - h)/h;
        int dw = (im.w - w)/w;
        for(i = 0; i < out.h; ++i){
            for(j = 0; j < out.w; ++j){
                image sub = get_sub_image(im, dh*i, dw*j, h, w);
                for(k = 0; k < out.c; ++k){
                    float val = get_pixel(out, i, j, k);
                    //printf("%f, ", val);
                    image sub_c = copy_image(sub);
                    scale_image(sub_c, val);
                    add_into_image(sub_c, vizs[k], 0, 0);
                    free_image(sub_c);
                }
                free_image(sub);
            }
        }
        //printf("\n");
        show_images(vizs, 10, "IMAGENET Visualization");
        cvWaitKey(1000);
        n = n->next;
    }
    cvWaitKey(0);
}
void features_VOC_image(char *image_file, char *image_dir, char *out_dir, int flip, int interval)
{
    int i,j;
    network net = parse_network_cfg("cfg/voc_imagenet.cfg");
    char image_path[1024];
    sprintf(image_path, "%s/%s",image_dir, image_file);
    char out_path[1024];
    if (flip)sprintf(out_path, "%s%d/%s_r.txt",out_dir, interval, image_file);
    else sprintf(out_path, "%s%d/%s.txt",out_dir, interval, image_file);
    printf("%s\n", image_file);

    IplImage* src = 0;
    if( (src = cvLoadImage(image_path,-1)) == 0 ) file_error(image_path);
    if(flip)cvFlip(src, 0, 1);
    int w = src->width;
    int h = src->height;
    int sbin = 8;
    double scale = pow(2., 1./interval);
    int m = (w<h)?w:h;
    int max_scale = 1+floor((double)log((double)m/(5.*sbin))/log(scale));
    if(max_scale < interval) error("max_scale must be >= interval");
    image *ims = calloc(max_scale+interval, sizeof(image));

    for(i = 0; i < interval; ++i){
        double factor = 1./pow(scale, i);
        double ih =  round(h*factor);
        double iw =  round(w*factor);
        int ex_h = round(ih/4.) - 2;
        int ex_w = round(iw/4.) - 2;
        ims[i] = features_output_size(net, src, ex_h, ex_w);

        ih =  round(h*factor);
        iw =  round(w*factor);
        ex_h = round(ih/8.) - 2;
        ex_w = round(iw/8.) - 2;
        ims[i+interval] = features_output_size(net, src, ex_h, ex_w);
        for(j = i+interval; j < max_scale; j += interval){
            factor /= 2.;
            ih =  round(h*factor);
            iw =  round(w*factor);
            ex_h = round(ih/8.) - 2;
            ex_w = round(iw/8.) - 2;
            ims[j+interval] = features_output_size(net, src, ex_h, ex_w);
        }
    }
    FILE *fp = fopen(out_path, "w");
    if(fp == 0) file_error(out_path);
    for(i = 0; i < max_scale+interval; ++i){
        image out = ims[i];
        fprintf(fp, "%d, %d, %d\n",out.c, out.h, out.w);
        for(j = 0; j < out.c*out.h*out.w; ++j){
            if(j != 0)fprintf(fp, ",");
            float o = out.data[j];
            if(o < 0) o = 0;
            fprintf(fp, "%g", o);
        }
        fprintf(fp, "\n");
        free_image(out);
    }
    free(ims);
    fclose(fp);
    cvReleaseImage(&src);
}

void test_distribution()
{
    IplImage* img = 0;
    if( (img = cvLoadImage("im_small.jpg",-1)) == 0 ) file_error("im_small.jpg");
    network net = parse_network_cfg("cfg/voc_features.cfg");
    int h = img->height/8-2;
    int w = img->width/8-2;
    image out = features_output_size(net, img, h, w);
    int c = out.c;
    out.c = 1;
    show_image(out, "output");
    out.c = c;
    image input = ipl_to_image(img);
    show_image(input, "input");
    CvScalar s;
    int i,j;
    image affects = make_image(input.h, input.w, 1);
    int count = 0;
    for(i = 0; i<img->height; i += 1){
        for(j = 0; j < img->width; j += 1){
            IplImage *copy = cvCloneImage(img);
            s=cvGet2D(copy,i,j); // get the (i,j) pixel value
            printf("%d/%d\n", count++, img->height*img->width);
            s.val[0]=0;
            s.val[1]=0;
            s.val[2]=0;
            cvSet2D(copy,i,j,s); // set the (i,j) pixel value
            image mod = features_output_size(net, copy, h, w);
            image dist = image_distance(out, mod);
            show_image(affects, "affects");
            cvWaitKey(1);
            cvReleaseImage(&copy);
            //affects.data[i*affects.w + j] += dist.data[3*dist.w+5];
            affects.data[i*affects.w + j] += dist.data[1*dist.w+1];
            free_image(mod);
            free_image(dist);
        }
    }
    show_image(affects, "Origins");
    cvWaitKey(0);
    cvWaitKey(0);
}
*/
