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

void test_convolve()
{
    image dog = load_image("dog.jpg");
    printf("dog channels %d\n", dog.c);
    image kernel = make_random_image(3,3,dog.c);
    image edge = make_image(dog.h, dog.w, 1);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i < 1000; ++i){
        convolve(dog, kernel, 1, 0, edge, 1);
    }
    end = clock();
    printf("Convolutions: %lf seconds\n", (double)(end-start)/CLOCKS_PER_SEC);
    show_image_layers(edge, "Test Convolve");
}

void test_convolve_matrix()
{
    image dog = load_image("dog.jpg");
    printf("dog channels %d\n", dog.c);
    
    int size = 11;
    int stride = 1;
    int n = 40;
    double *filters = make_random_image(size, size, dog.c*n).data;

    int mw = ((dog.h-size)/stride+1)*((dog.w-size)/stride+1);
    int mh = (size*size*dog.c);
    double *matrix = calloc(mh*mw, sizeof(double));

    image edge = make_image((dog.h-size)/stride+1, (dog.w-size)/stride+1, n);


    int i;
    clock_t start = clock(), end;
    for(i = 0; i < 1000; ++i){
        im2col_cpu(dog.data,  dog.c,  dog.h,  dog.w,  size,  stride, matrix);
        gemm(0,0,n,mw,mh,1,filters,mh,matrix,mw,1,edge.data,mw);
    }
    end = clock();
    printf("Convolutions: %lf seconds\n", (double)(end-start)/CLOCKS_PER_SEC);
    show_image_layers(edge, "Test Convolve");
    cvWaitKey(0);
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

void verify_convolutional_layer()
{
    srand(0);
    int i;
    int n = 1;
    int stride = 1;
    int size = 3;
    double eps = .00000001;
    image test = make_random_image(5,5, 1);
    convolutional_layer layer = *make_convolutional_layer(test.h,test.w,test.c, n, size, stride, RELU);
    image out = get_convolutional_image(layer);
    double **jacobian = calloc(test.h*test.w*test.c, sizeof(double));
    
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
    double **jacobian2 = calloc(out.h*out.w*out.c, sizeof(double));
    image in_delta = make_image(test.h, test.w, test.c);
    image out_delta = get_convolutional_delta(layer);
    for(i = 0; i < out.h*out.w*out.c; ++i){
        out_delta.data[i] = 1;
        backward_convolutional_layer(layer, test.data, in_delta.data);
        image partial = copy_image(in_delta);
        jacobian2[i] = partial.data;
        out_delta.data[i] = 0;
    }
    int j;
    double *j1 = calloc(test.h*test.w*test.c*out.h*out.w*out.c, sizeof(double));
    double *j2 = calloc(test.h*test.w*test.c*out.h*out.w*out.c, sizeof(double));
    for(i = 0; i < test.h*test.w*test.c; ++i){
        for(j =0 ; j < out.h*out.w*out.c; ++j){
            j1[i*out.h*out.w*out.c + j] = jacobian[i][j];
            j2[i*out.h*out.w*out.c + j] = jacobian2[j][i];
            printf("%f %f\n", jacobian[i][j], jacobian2[j][i]);
        }
    }


    image mj1 = double_to_image(test.w*test.h*test.c, out.w*out.h*out.c, 1, j1);
    image mj2 = double_to_image(test.w*test.h*test.c, out.w*out.h*out.c, 1, j2);
    printf("%f %f\n", avg_image_layer(mj1,0), avg_image_layer(mj2,0));
    show_image(mj1, "forward jacobian");
    show_image(mj2, "backward jacobian");
    
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
    while(++count < 100000000){
        double v = ((double)rand()/RAND_MAX);
        double truth = v*v;
        input[0] = v;
        forward_network(net, input);
        double *out = get_network_output(net);
        double *delta = get_network_delta(net);
        double err = pow((out[0]-truth),2.);
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
    data train = load_data_image_pathfile_random("train_paths.txt", 101,labels, 2);
    free_data(train);
}

void test_full()
{
    network net = parse_network_cfg("full.cfg");
    srand(0);
    int i = 0;
    char *labels[] = {"cat","dog"};
    double lr = .00001;
    double momentum = .9;
    double decay = 0.01;
    while(i++ < 1000 || 1){
        data train = load_data_image_pathfile_random("train_paths.txt", 1000, labels, 2);
        train_network(net, train, lr, momentum, decay);
        free_data(train);
        printf("Round %d\n", i);
    }
}

void test_nist()
{
    srand(444444);
    srand(888888);
    network net = parse_network_cfg("nist_basic.cfg");
    data train = load_categorical_data_csv("mnist/mnist_train.csv", 0, 10);
    data test = load_categorical_data_csv("mnist/mnist_test.csv",0,10);
    normalize_data_rows(train);
    normalize_data_rows(test);
    //randomize_data(train);
    int count = 0;
    double lr = .0005;
    double momentum = .9;
    double decay = 0.01;
    clock_t start = clock(), end;
    while(++count <= 1000){
        double acc = train_network_sgd(net, train, 6400, lr, momentum, decay);
        printf("%5d Training Loss: %lf, Params: %f %f %f, ",count*100, 1.-acc, lr, momentum, decay);
        end = clock();
        printf("Time: %lf seconds\n", (double)(end-start)/CLOCKS_PER_SEC);
        start=end;
        //visualize_network(net);
        //cvWaitKey(100);
        //lr /= 2; 
        if(count%5 == 0 && 0){
            double train_acc = network_accuracy(net, train);
            fprintf(stderr, "\nTRAIN: %f\n", train_acc);
            double test_acc = network_accuracy(net, test);
            fprintf(stderr, "TEST: %f\n\n", test_acc);
            printf("%d, %f, %f\n", count, train_acc, test_acc);
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
    /*
       data *split = split_data(d, 1, 10);
       data train = split[0];
       data test = split[1];
     */
    matrix prediction = make_matrix(test.y.rows, test.y.cols);
    int n = 30;
    for(i = 0; i < n; ++i){
        int count = 0;
        double lr = .0005;
        double momentum = .9;
        double decay = .01;
        network net = parse_network_cfg("nist.cfg");
        while(++count <= 15){
            double acc = train_network_sgd(net, train, train.X.rows, lr, momentum, decay);
            printf("Training Accuracy: %lf Learning Rate: %f Momentum: %f Decay: %f\n", acc, lr, momentum, decay );
            lr /= 2; 
        }
        matrix partial = network_predict_data(net, test);
        double acc = matrix_accuracy(test.y, partial);
        printf("Model Accuracy: %lf\n", acc);
        matrix_add_matrix(partial, prediction);
        acc = matrix_accuracy(test.y, prediction);
        printf("Current Ensemble Accuracy: %lf\n", acc);
        free_matrix(partial);
    }
    double acc = matrix_accuracy(test.y, prediction);
    printf("Full Ensemble Accuracy: %lf\n", acc);
}

void test_kernel_update()
{
    srand(0);
    double delta[] = {.1};
    double input[] = {.3, .5, .3, .5, .5, .5, .5, .0, .5};
    double kernel[] = {1,2,3,4,5,6,7,8,9};
    convolutional_layer layer = *make_convolutional_layer(3, 3, 1, 1, 3, 1, LINEAR);
    layer.kernels[0].data = kernel;
    layer.delta = delta;
    learn_convolutional_layer(layer, input);
    print_image(layer.kernels[0]);
    print_image(get_convolutional_delta(layer));
    print_image(layer.kernel_updates[0]);

}

void test_random_classify()
{
    network net = parse_network_cfg("connected.cfg");
    matrix m = csv_to_matrix("train.csv");
    //matrix ho = hold_out_matrix(&m, 2500);
    double *truth = pop_column(&m, 0);
    //double *ho_truth = pop_column(&ho, 0);
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
            //backward_network(net, m.vals[index], );
            update_network(net, .00001, 0,0);
        }
        //double test_acc = error_network(net, m, truth);
        //double valid_acc = error_network(net, ho, ho_truth);
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
        double *out = get_network_output(net);
        if(fabs(out[0]) < .5) fprintf(fp, "0\n");
        else fprintf(fp, "1\n");
    }
    fclose(fp);
    printf("Neural Net Learning: %lf seconds\n", (double)(end-start)/CLOCKS_PER_SEC);
}

void test_split()
{
    data train = load_categorical_data_csv("mnist/mnist_train.csv", 0, 10);
    data *split = split_data(train, 0, 13);
    printf("%d, %d, %d\n", train.X.rows, split[0].X.rows, split[1].X.rows);
}

double *random_matrix(int rows, int cols)
{
    int i, j;
    double *m = calloc(rows*cols, sizeof(double));
    for(i = 0; i < rows; ++i){
        for(j = 0; j < cols; ++j){
            m[i*cols+j] = (double)rand()/RAND_MAX;
        }
    }
    return m;
}

void test_blas()
{
    int m = 6025, n = 20, k = 11*11*3;
    double *a = random_matrix(m,k);
    double *b = random_matrix(k,n);
    double *c = random_matrix(m,n);
    int i;
    for(i = 0; i<1000; ++i){
        gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
    }
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
    double *matrix = calloc(msize, sizeof(double));
    int i;
    for(i = 0; i < 1000; ++i){
    im2col_cpu(test.data,  c,  h,  w,  size,  stride, matrix);
    image render = double_to_image(mh, mw, mc, matrix);
    }
}

int main()
{
    //test_blas();
 test_convolve_matrix();
//    test_im2row();
    //test_kernel_update();
    //test_split();
    //test_ensemble();
    //test_nist();
    //test_full();
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
