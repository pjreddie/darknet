#include "network.h"
#include "utils.h"
#include "parser.h"
#include "data.h"
#include "rbm_layer.h"
#include "image.h"
#include "data.h"
#include "cuda.h"
 
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

data load_data_from_memory(float **p, float **labels, int n, int c, int w, int h)
{
    data d;
    d.shallow = 0;
    matrix X = make_matrix(n, w*h);
    matrix Y = make_matrix(n, c);
    d.X = X;
    d.y = Y;
    long i, j, k;
    for(i = 0; i < n; ++i) {
        for(j=0; j<X.cols; ++j) {
            d.X.vals[i][j] = p[i][j];
        }
        for(k=0; k<c; ++k) {
            d.y.vals[i][k] = labels[i][k];
        }
    }

    return d;
}

void get_next(data d, int n, int offset, float *X)
{
    int j;
    for(j = 0; j < n; ++j){
        int index = offset + j;
        memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
    }
}

void pretrain_autoencoder_datum(network net, float *x)
{
    *net.seen += net.batch;
    network_state state;
    state.index = 0;
    state.net = net;
    state.input = x;
    state.train = 1;
    int i;
    for(i=0; i<(net.n-1)/2; i++) {
        state.index = i;
        preforward_rbm_layer(net.layers[i], state);
        prebackward_rbm_layer(net.layers[i], state);
        preupdate_rbm_layer(net.layers[i], net.batch, net.learning_rate, net.momentum, net.decay);
        forward_rbm_layer(net.layers[i], state);
        state.input = net.layers[i].output;
    }
}


#ifdef GPU
void pretrain_autoencoder_datum_gpu(network net, float *x)
{
    *net.seen += net.batch;
    network_state state;
    state.index = 0;
    state.net = net;
    int x_size = get_network_input_size(net)*net.batch;
    if(!*net.input_gpu){
        *net.input_gpu = cuda_make_array(x, x_size);
    }else{
        cuda_push_array(*net.input_gpu, x, x_size);
    }
    state.input = *net.input_gpu;
    state.train = 1;
    int i;
    for(i=0; i<(net.n-1)/2; i++) {
        state.index = i;
        preforward_rbm_layer_gpu(net.layers[i], state);
        prebackward_rbm_layer_gpu(net.layers[i], state);
        preupdate_rbm_layer_gpu(net.layers[i], net.batch, net.learning_rate, net.momentum, net.decay);
        forward_rbm_layer_gpu(net.layers[i], state);
        state.input = net.layers[i].output_gpu;
    }

    return;
}
#endif

void pretrain_autoencoder(network net, data d)
{
    int batch = net.batch;
    int n = d.X.rows / batch;
    int epcho = 10;
    float *X = (float*)calloc(batch*d.X.cols, sizeof(float));

    int i, j;
    for(j=0; j<epcho; ++j){
        for(i = 0; i < n; ++i){
            get_next(d, batch, i*batch, X);
#ifdef GPU
            pretrain_autoencoder_datum_gpu(net, X);
#else
            pretrain_autoencoder_datum(net, X);
#endif
        }
    }
    free(X);
    X = 0;
}

void mirror_autoencoder(network net) {
    for(int i=0; i<(net.n-1)/2; i++) {
        mirror_rbm_layer(net.layers[i], net.layers[net.n-1-1-i]);
    }
}

void finetune_autoencoder(network net, data d) {
    int i;
    int epcho = 30;
    for(i=0; i<net.n-1; i++) {
        net.layers[i].type = CONNECTED;
    }
    for(i=0; i<epcho; i++) {
        train_network(net, d);
    }
    for(i=0; i<net.n-1; i++) {
        net.layers[i].type = RBM;
    }
}

void predict_autoencoder(network net, network_state state) {
    for(int i=0; i<net.n-1; i++) {
        net.layers[i].type = CONNECTED;
    }
    forward_network(net, state);
    for(int i=0; i<net.n-1; i++) {
        net.layers[i].type = RBM;
    }
}


int run_autoencoder(int argc, char **argv) {
/*pretraining*/
    //load data
    int n = 10; //number of pictures used for training
    int w = 32; // width
    int h = 32; // height
    float **p = (float**)calloc(n, sizeof(float*));
    int i;
    for(i=0; i<n; i++) {
        p[i] = (float*)calloc(w*h, sizeof(float));
    }
    
    printf("Loading data to memory....\n");
    for(i=0; i<n; i++) {
        char name[100]={0};
        strcat(name, "data/face/");
        char temp[20]={0};
        snprintf(temp, 10,"%d",i+1);
        strcat(name, temp);
        strcat(name, ".jpg");
        image src = load_image(name, w, h, 3);
        src = grayscale_image(src);
        int ii, jj;
        for(ii=0; ii<h; ii++) {
	    for(jj=0; jj<w; jj++) {
                p[i][ii*w+jj] = src.data[ii*w+jj];
            }
        }
    }
    printf("Loading data finished.....\n");
    
    //prepair data for pretrain
    printf("Preparing data for pretrain.....\n");
    data d = load_data_from_memory(p, p, n, w*h, w, h);
    printf("Data prparation for pretrain finished....\n");
    for(i=0; i<n; i++) {
        free(p[i]);
        p[i] = 0;
    }
    free(p);
    p = 0;

    //construct network from configure file.
    printf("Constructing network from configure file.....\n");
    char *cfgfile = "cfg/rbms.cfg";
    char *weightfile = 0;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    printf("Network construction finished.....\n");
    for(i=0; i<net.n; ++i) {
        printf("inputs: %d, outputs: %d. \n", net.layers[i].inputs ,net.layers[i].outputs);
    }

    //prepair test data
    printf("Reading test data....\n");
    image test = load_image("data/face/46.jpg", w, h, 3);
    test = grayscale_image(test);
    printf("Reading test data finished....\n");
    show_image(test, "results/test");
    network_state st;
    st.net = net;
    st.index = 0;
    st.input = test.data;
    st.truth = 0;
    st.train = 0;
    st.delta = 0;

    //pretrain network
    printf("Pretraining network.....\n");
    clock_t start, finish;
    double totaltime;
    start = clock();
    pretrain_autoencoder(net, d);
    finish = clock();
    totaltime = (double)(finish-start)/CLOCKS_PER_SEC;
    printf("Pretraining network finished.....\n");
    printf("Pretraining time lasts: %f ....\n", totaltime);

#ifdef GPU
    for(i=0; i<(net.n-1)/2; i++) {
        pull_rbm_layer(net.layers[i]);
    }
#endif

    //Mirror network
    printf("Mirroring network.....\n");
    mirror_autoencoder(net);
    printf("Mirroring network finished.....\n");

    //test prtrained network only use cpu version
    printf("Testing pretrained network.....\n");
    printf("Predicting test data.....\n");
    predict_autoencoder(net, st);
    float *output = net.layers[net.n-2].output;
    printf("predicting test data finished....\n");

    //save pretrain test result image  
    image pretrain_test = make_image(w, h, 1);
    for(int ii=0; ii<h; ii++) {
        for(int jj=0; jj<w; jj++) {
            pretrain_test.data[ii*w+jj] = output[ii*w+jj];
        }
    }
#ifdef GPU
    show_image(pretrain_test, "results/pretrain_test_gpu");
#else
    show_image(pretrain_test, "results/pretrain_test");
#endif

/*fintuning*/
#ifdef GPU
    cuda_free(*net.input_gpu);
    *net.input_gpu = 0;
    for(i=0; i<net.n-1; i++) {
        push_rbm_layer(net.layers[i]);
    }
#endif
    printf("Fintuning network.....\n"); 
    finetune_autoencoder(net, d);
    printf("Fintuning network finished.....\n");

#ifdef GPU
    for(i=0; i<net.n-1; i++) {
        pull_rbm_layer(net.layers[i]);
    }
#endif

    //test finetuned network
    printf("Testing finetuned network.....\n");
    printf("Predicting test data.....\n");
    predict_autoencoder(net, st);
    output = net.layers[net.n-2].output;
    printf("predicting test data finished....\n");

    //save finetune test result image  
    image finetune_test = make_image(w, h, 1);
    for(int ii=0; ii<h; ii++) {
        for(int jj=0; jj<w; jj++) {
            finetune_test.data[ii*w+jj] = output[ii*w+jj];
        }
    }
#ifdef GPU
    show_image(finetune_test, "results/finetune_test_gpu");
#else
    show_image(finetune_test, "results/finetune_test");
#endif
    
    //Error
    printf("Test error....\n");
    float pretrain_error=0;
    float finetune_error=0;
    for(int ii=0; ii<h; ii++) {
        for(int jj=0; jj<w; jj++) {
            pretrain_error += (test.data[ii*w+jj]-pretrain_test.data[ii*w+jj])*(test.data[ii*w+jj]-pretrain_test.data[ii*w+jj]);
            finetune_error += (test.data[ii*w+jj]-finetune_test.data[ii*w+jj])*(test.data[ii*w+jj]-finetune_test.data[ii*w+jj]);
        }
    }
    printf("Pretrain Error: %f, Finetune Error: %f. \n", pretrain_error, finetune_error);
    printf("Test error finished.....\n");

    free_image(test);
    free_image(pretrain_test);
    free_image(finetune_test);
    free_data(d);
    free_network(net);

    return 0;
}
