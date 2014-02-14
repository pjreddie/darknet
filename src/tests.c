#include "connected_layer.h"
//#include "old_conv.h"
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
        im2col_cpu(dog.data,  dog.c,  dog.h,  dog.w,  size,  stride, matrix);
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
    convolutional_layer layer = *make_convolutional_layer(test.h,test.w,test.c, n, size, stride, RELU);
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
    network net = parse_network_cfg("test_parser.cfg");
    float input[1];
    int count = 0;
        
    float avgerr = 0;
    while(++count < 100000000){
        float v = ((float)rand()/RAND_MAX);
        float truth = v*v;
        input[0] = v;
        forward_network(net, input);
        float *out = get_network_output(net);
        float *delta = get_network_delta(net);
        float err = pow((out[0]-truth),2.);
        avgerr = .99 * avgerr + .01 * err;
        if(count % 1000000 == 0) printf("%f %f :%f AVG %f \n", truth, out[0], err, avgerr);
        delta[0] = truth - out[0];
        backward_network(net, input, &truth);
        update_network(net, .001,0,0);
    }
}

void test_data()
{
    char *labels[] = {"cat","dog"};
    data train = load_data_image_pathfile_random("train_paths.txt", 101,labels, 2, 300, 400);
    free_data(train);
}

void test_full()
{
    network net = parse_network_cfg("full.cfg");
    srand(2222222);
    int i = 800;
    char *labels[] = {"cat","dog"};
    float lr = .00001;
    float momentum = .9;
    float decay = 0.01;
    while(i++ < 1000 || 1){
        visualize_network(net);
        cvWaitKey(100);
        data train = load_data_image_pathfile_random("train_paths.txt", 1000, labels, 2, 256, 256);
        image im = float_to_image(256, 256, 3,train.X.vals[0]);
        show_image(im, "input");
        cvWaitKey(100);
        //scale_data_rows(train, 1./255.);
        normalize_data_rows(train);
        clock_t start = clock(), end;
        float loss = train_network_sgd(net, train, 100, lr, momentum, decay);
        end = clock();
        printf("%d: %f, Time: %lf seconds, LR: %f, Momentum: %f, Decay: %f\n", i, loss, (float)(end-start)/CLOCKS_PER_SEC, lr, momentum, decay);
        free_data(train);
        if(i%100==0){
            char buff[256];
            sprintf(buff, "backup_%d.cfg", i);
            //save_network(net, buff);
        }
        //lr *= .99;
    }
}

void test_nist()
{
    srand(444444);
    srand(888888);
    network net = parse_network_cfg("nist.cfg");
    data train = load_categorical_data_csv("mnist/mnist_train.csv", 0, 10);
    data test = load_categorical_data_csv("mnist/mnist_test.csv",0,10);
    normalize_data_rows(train);
    normalize_data_rows(test);
    //randomize_data(train);
    int count = 0;
    float lr = .0005;
    float momentum = .9;
    float decay = 0.001;
    clock_t start = clock(), end;
    while(++count <= 100){
        //visualize_network(net);
        float loss = train_network_sgd(net, train, 1000, lr, momentum, decay);
        printf("%5d Training Loss: %lf, Params: %f %f %f, ",count*100, loss, lr, momentum, decay);
        end = clock();
        printf("Time: %lf seconds\n", (float)(end-start)/CLOCKS_PER_SEC);
        start=end;
        //cvWaitKey(100);
        //lr /= 2; 
        if(count%5 == 0){
            float train_acc = network_accuracy(net, train);
            fprintf(stderr, "\nTRAIN: %f\n", train_acc);
            float test_acc = network_accuracy(net, test);
            fprintf(stderr, "TEST: %f\n\n", test_acc);
            printf("%d, %f, %f\n", count, train_acc, test_acc);
            //lr *= .5;
        }
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
            float acc = train_network_sgd(net, train, train.X.rows, lr, momentum, decay);
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
            forward_network(net, m.vals[index]);
            float *out = get_network_output(net);
            float *delta = get_network_delta(net);
            //printf("%f\n", out[0]);
            delta[0] = truth[index] - out[0];
            // printf("%f\n", delta[0]);
            //printf("%f %f\n", truth[index], out[0]);
            //backward_network(net, m.vals[index], );
            update_network(net, .00001, 0,0);
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
        forward_network(net, test.vals[i]);
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
        im2col_cpu(test.data,  c,  h,  w,  size,  stride, matrix);
        //image render = float_to_image(mh, mw, mc, matrix);
    }
}

void train_VOC()
{
    network net = parse_network_cfg("cfg/voc_backup_ramp_80.cfg");
    srand(2222222);
    int i = 0;
    char *labels[] = {"aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"};
    float lr = .00001;
    float momentum = .9;
    float decay = 0.01;
    while(i++ < 1000 || 1){
        visualize_network(net);
        cvWaitKey(100);
        data train = load_data_image_pathfile_random("images/VOC2012/train_paths.txt", 1000, labels, 20, 300, 400);
        image im = float_to_image(300, 400, 3,train.X.vals[0]);
        show_image(im, "input");
        cvWaitKey(100);
        normalize_data_rows(train);
        clock_t start = clock(), end;
        float loss = train_network_sgd(net, train, 1000, lr, momentum, decay);
        end = clock();
        printf("%d: %f, Time: %lf seconds, LR: %f, Momentum: %f, Decay: %f\n", i, loss, (float)(end-start)/CLOCKS_PER_SEC, lr, momentum, decay);
        free_data(train);
        if(i%10==0){
            char buff[256];
            sprintf(buff, "cfg/voc_backup_ramp_%d.cfg", i);
            save_network(net, buff);
        }
        //lr *= .99;
    }
}

int main()
{
    //feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

    //test_blas();
    //test_convolve_matrix();
    //    test_im2row();
    //test_split();
    //test_ensemble();
    //test_nist();
    //test_full();
    train_VOC();
    //test_random_preprocess();
    //test_random_classify();
    //test_parser();
    //test_backpropagate();
    //test_ann();
    //test_convolve();
    //test_upsample();
    //test_rotate();
    //test_load();
    //test_network();
    //test_convolutional_layer();
    //verify_convolutional_layer();
    //test_color();
    //cvWaitKey(0);
    return 0;
}
