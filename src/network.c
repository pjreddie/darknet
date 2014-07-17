#include <stdio.h>
#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"

#include "connected_layer.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "softmax_layer.h"

network make_network(int n, int batch)
{
    network net;
    net.n = n;
    net.batch = batch;
    net.layers = calloc(net.n, sizeof(void *));
    net.types = calloc(net.n, sizeof(LAYER_TYPE));
    net.outputs = 0;
    net.output = 0;
    #ifdef GPU
    net.input_cl = 0;
    #endif
    return net;
}

void print_convolutional_cfg(FILE *fp, convolutional_layer *l, int first)
{
    int i;
    fprintf(fp, "[convolutional]\n");
    if(first) fprintf(fp,   "batch=%d\n"
                            "height=%d\n"
                            "width=%d\n"
                            "channels=%d\n",
                            l->batch,l->h, l->w, l->c);
    fprintf(fp, "filters=%d\n"
                "size=%d\n"
                "stride=%d\n"
                "activation=%s\n",
                l->n, l->size, l->stride,
                get_activation_string(l->activation));
    fprintf(fp, "data=");
    for(i = 0; i < l->n; ++i) fprintf(fp, "%g,", l->biases[i]);
    for(i = 0; i < l->n*l->c*l->size*l->size; ++i) fprintf(fp, "%g,", l->filters[i]);
    fprintf(fp, "\n\n");
}
void print_connected_cfg(FILE *fp, connected_layer *l, int first)
{
    int i;
    fprintf(fp, "[connected]\n");
    if(first) fprintf(fp, "batch=%d\ninput=%d\n", l->batch, l->inputs);
    fprintf(fp, "output=%d\n"
            "activation=%s\n",
            l->outputs,
            get_activation_string(l->activation));
    fprintf(fp, "data=");
    for(i = 0; i < l->outputs; ++i) fprintf(fp, "%g,", l->biases[i]);
    for(i = 0; i < l->inputs*l->outputs; ++i) fprintf(fp, "%g,", l->weights[i]);
    fprintf(fp, "\n\n");
}

void print_maxpool_cfg(FILE *fp, maxpool_layer *l, int first)
{
    fprintf(fp, "[maxpool]\n");
    if(first) fprintf(fp,   "batch=%d\n"
            "height=%d\n"
            "width=%d\n"
            "channels=%d\n",
            l->batch,l->h, l->w, l->c);
    fprintf(fp, "stride=%d\n\n", l->stride);
}

void print_normalization_cfg(FILE *fp, normalization_layer *l, int first)
{
    fprintf(fp, "[localresponsenormalization]\n");
    if(first) fprintf(fp,   "batch=%d\n"
            "height=%d\n"
            "width=%d\n"
            "channels=%d\n",
            l->batch,l->h, l->w, l->c);
    fprintf(fp, "size=%d\n"
                "alpha=%g\n"
                "beta=%g\n"
                "kappa=%g\n\n", l->size, l->alpha, l->beta, l->kappa);
}

void print_softmax_cfg(FILE *fp, softmax_layer *l, int first)
{
    fprintf(fp, "[softmax]\n");
    if(first) fprintf(fp, "batch=%d\ninput=%d\n", l->batch, l->inputs);
    fprintf(fp, "\n");
}

void save_network(network net, char *filename)
{
    FILE *fp = fopen(filename, "w");
    if(!fp) file_error(filename);
    int i;
    for(i = 0; i < net.n; ++i)
    {
        if(net.types[i] == CONVOLUTIONAL)
            print_convolutional_cfg(fp, (convolutional_layer *)net.layers[i], i==0);
        else if(net.types[i] == CONNECTED)
            print_connected_cfg(fp, (connected_layer *)net.layers[i], i==0);
        else if(net.types[i] == MAXPOOL)
            print_maxpool_cfg(fp, (maxpool_layer *)net.layers[i], i==0);
        else if(net.types[i] == NORMALIZATION)
            print_normalization_cfg(fp, (normalization_layer *)net.layers[i], i==0);
        else if(net.types[i] == SOFTMAX)
            print_softmax_cfg(fp, (softmax_layer *)net.layers[i], i==0);
    }
    fclose(fp);
}

#ifdef GPU
void forward_network(network net, float *input, int train)
{
    cl_setup();
    size_t size = get_network_input_size(net);
    if(!net.input_cl){
        net.input_cl = clCreateBuffer(cl.context,
            CL_MEM_READ_WRITE, size*sizeof(float), 0, &cl.error);
        check_error(cl);
    }
    cl_write_array(net.input_cl, input, size);
    cl_mem input_cl = net.input_cl;
    int i;
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            forward_convolutional_layer_gpu(layer, input_cl);
            input_cl = layer.output_cl;
            input = layer.output;
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            forward_connected_layer(layer, input, train);
            input = layer.output;
        }
        else if(net.types[i] == SOFTMAX){
            softmax_layer layer = *(softmax_layer *)net.layers[i];
            forward_softmax_layer(layer, input);
            input = layer.output;
        }
        else if(net.types[i] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net.layers[i];
            forward_maxpool_layer(layer, input);
            input = layer.output;
        }
        else if(net.types[i] == NORMALIZATION){
            normalization_layer layer = *(normalization_layer *)net.layers[i];
            forward_normalization_layer(layer, input);
            input = layer.output;
        }
    }
}

#else

void forward_network(network net, float *input, int train)
{
    int i;
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            forward_convolutional_layer(layer, input);
            input = layer.output;
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            forward_connected_layer(layer, input, train);
            input = layer.output;
        }
        else if(net.types[i] == SOFTMAX){
            softmax_layer layer = *(softmax_layer *)net.layers[i];
            forward_softmax_layer(layer, input);
            input = layer.output;
        }
        else if(net.types[i] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net.layers[i];
            forward_maxpool_layer(layer, input);
            input = layer.output;
        }
        else if(net.types[i] == NORMALIZATION){
            normalization_layer layer = *(normalization_layer *)net.layers[i];
            forward_normalization_layer(layer, input);
            input = layer.output;
        }
    }
}
#endif

void update_network(network net, float step, float momentum, float decay)
{
    int i;
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            update_convolutional_layer(layer, step, momentum, decay);
        }
        else if(net.types[i] == MAXPOOL){
            //maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        }
        else if(net.types[i] == SOFTMAX){
            //maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        }
        else if(net.types[i] == NORMALIZATION){
            //maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            update_connected_layer(layer, step, momentum, decay);
        }
    }
}

float *get_network_output_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return layer.output;
    } else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        return layer.output;
    } else if(net.types[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net.layers[i];
        return layer.output;
    } else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.output;
    } else if(net.types[i] == NORMALIZATION){
        normalization_layer layer = *(normalization_layer *)net.layers[i];
        return layer.output;
    }
    return 0;
}
float *get_network_output(network net)
{
    return get_network_output_layer(net, net.n-1);
}

float *get_network_delta_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return layer.delta;
    } else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        return layer.delta;
    } else if(net.types[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net.layers[i];
        return layer.delta;
    } else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.delta;
    }
    return 0;
}

float *get_network_delta(network net)
{
    return get_network_delta_layer(net, net.n-1);
}

float calculate_error_network(network net, float *truth)
{
    float sum = 0;
    float *delta = get_network_delta(net);
    float *out = get_network_output(net);
    int i;
    for(i = 0; i < get_network_output_size(net)*net.batch; ++i){
        //if(i %get_network_output_size(net) == 0) printf("\n");
        //printf("%5.2f %5.2f, ", out[i], truth[i]);
        //if(i == get_network_output_size(net)) printf("\n");
        delta[i] = truth[i] - out[i];
        //printf("%f, ", delta[i]);
        sum += delta[i]*delta[i];
    }
    //printf("\n");
    return sum;
}

int get_predicted_class_network(network net)
{
    float *out = get_network_output(net);
    int k = get_network_output_size(net);
    return max_index(out, k);
}

float backward_network(network net, float *input, float *truth)
{
    float error = calculate_error_network(net, truth);
    int i;
    float *prev_input;
    float *prev_delta;
    for(i = net.n-1; i >= 0; --i){
        if(i == 0){
            prev_input = input;
            prev_delta = 0;
        }else{
            prev_input = get_network_output_layer(net, i-1);
            prev_delta = get_network_delta_layer(net, i-1);
        }
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            backward_convolutional_layer(layer, prev_delta);
        }
        else if(net.types[i] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net.layers[i];
            if(i != 0) backward_maxpool_layer(layer, prev_input, prev_delta);
        }
        else if(net.types[i] == NORMALIZATION){
            normalization_layer layer = *(normalization_layer *)net.layers[i];
            if(i != 0) backward_normalization_layer(layer, prev_input, prev_delta);
        }
        else if(net.types[i] == SOFTMAX){
            softmax_layer layer = *(softmax_layer *)net.layers[i];
            if(i != 0) backward_softmax_layer(layer, prev_input, prev_delta);
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            backward_connected_layer(layer, prev_input, prev_delta);
        }
    }
    return error;
}

float train_network_datum(network net, float *x, float *y, float step, float momentum, float decay)
{
    forward_network(net, x, 1);
    //int class = get_predicted_class_network(net);
    float error = backward_network(net, x, y);
    update_network(net, step, momentum, decay);
    //return (y[class]?1:0);
    return error;
}

float train_network_sgd(network net, data d, int n, float step, float momentum,float decay)
{
    int batch = net.batch;
    float *X = calloc(batch*d.X.cols, sizeof(float));
    float *y = calloc(batch*d.y.cols, sizeof(float));

    int i,j;
    float sum = 0;
    for(i = 0; i < n; ++i){
        for(j = 0; j < batch; ++j){
            int index = rand()%d.X.rows;
            memcpy(X+j*d.X.cols, d.X.vals[index], d.X.cols*sizeof(float));
            memcpy(y+j*d.y.cols, d.y.vals[index], d.y.cols*sizeof(float));
        }
        float err = train_network_datum(net, X, y, step, momentum, decay);
        sum += err;
        //train_network_datum(net, X, y, step, momentum, decay);
        /*
        float *y = d.y.vals[index];
        int class = get_predicted_class_network(net);
        correct += (y[class]?1:0);
        */

/*
        for(j = 0; j < d.y.cols*batch; ++j){
            printf("%6.3f ", y[j]);
        }
        printf("\n");
        for(j = 0; j < d.y.cols*batch; ++j){
            printf("%6.3f ", get_network_output(net)[j]);
        }
        printf("\n");
        printf("\n");
        */


        //printf("%d %f %f\n", i,net.output[0], d.y.vals[index][0]);
        //if((i+1)%10 == 0){
        //    printf("%d: %f\n", (i+1), (float)correct/(i+1));
        //}
    }
    //printf("Accuracy: %f\n",(float) correct/n);
    free(X);
    free(y);
    return (float)sum/(n*batch);
}
float train_network_batch(network net, data d, int n, float step, float momentum,float decay)
{
    int i,j;
    float sum = 0;
    int batch = 2;
    for(i = 0; i < n; ++i){
        for(j = 0; j < batch; ++j){
            int index = rand()%d.X.rows;
            float *x = d.X.vals[index];
            float *y = d.y.vals[index];
            forward_network(net, x, 1);
            sum += backward_network(net, x, y);
        }
        update_network(net, step, momentum, decay);
    }
    return (float)sum/(n*batch);
}


void train_network(network net, data d, float step, float momentum, float decay)
{
    int i;
    int correct = 0;
    for(i = 0; i < d.X.rows; ++i){
        correct += train_network_datum(net, d.X.vals[i], d.y.vals[i], step, momentum, decay);
        if(i%100 == 0){
            visualize_network(net);
            cvWaitKey(10);
        }
    }
    visualize_network(net);
    cvWaitKey(100);
    fprintf(stderr, "Accuracy: %f\n", (float)correct/d.X.rows);
}

int get_network_input_size_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return layer.h*layer.w*layer.c;
    }
    else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        return layer.h*layer.w*layer.c;
    }
    else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.inputs;
    }
    else if(net.types[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net.layers[i];
        return layer.inputs;
    }
    return 0;
}

int get_network_output_size_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        image output = get_convolutional_image(layer);
        return output.h*output.w*output.c;
    }
    else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        image output = get_maxpool_image(layer);
        return output.h*output.w*output.c;
    }
    else if(net.types[i] == CONNECTED){
        connected_layer layer = *(connected_layer *)net.layers[i];
        return layer.outputs;
    }
    else if(net.types[i] == SOFTMAX){
        softmax_layer layer = *(softmax_layer *)net.layers[i];
        return layer.inputs;
    }
    return 0;
}

int resize_network(network net, int h, int w, int c)
{
    int i;
    for (i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer *layer = (convolutional_layer *)net.layers[i];
            resize_convolutional_layer(layer, h, w, c);
            image output = get_convolutional_image(*layer);
            h = output.h;
            w = output.w;
            c = output.c;
        }else if(net.types[i] == MAXPOOL){
            maxpool_layer *layer = (maxpool_layer *)net.layers[i];
            resize_maxpool_layer(layer, h, w, c);
            image output = get_maxpool_image(*layer);
            h = output.h;
            w = output.w;
            c = output.c;
        }else if(net.types[i] == NORMALIZATION){
            normalization_layer *layer = (normalization_layer *)net.layers[i];
            resize_normalization_layer(layer, h, w, c);
            image output = get_normalization_image(*layer);
            h = output.h;
            w = output.w;
            c = output.c;
        }else{
            error("Cannot resize this type of layer");
        }
    }
    return 0;
}

int get_network_output_size(network net)
{
    int i = net.n-1;
    return get_network_output_size_layer(net, i);
}

int get_network_input_size(network net)
{
    return get_network_input_size_layer(net, 0);
}

image get_network_image_layer(network net, int i)
{
    if(net.types[i] == CONVOLUTIONAL){
        convolutional_layer layer = *(convolutional_layer *)net.layers[i];
        return get_convolutional_image(layer);
    }
    else if(net.types[i] == MAXPOOL){
        maxpool_layer layer = *(maxpool_layer *)net.layers[i];
        return get_maxpool_image(layer);
    }
    else if(net.types[i] == NORMALIZATION){
        normalization_layer layer = *(normalization_layer *)net.layers[i];
        return get_normalization_image(layer);
    }
    return make_empty_image(0,0,0);
}

image get_network_image(network net)
{
    int i;
    for(i = net.n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    return make_empty_image(0,0,0);
}

void visualize_network(network net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net.n; ++i){
        sprintf(buff, "Layer %d", i);
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            prev = visualize_convolutional_layer(layer, buff, prev);
        }
        if(net.types[i] == NORMALIZATION){
            normalization_layer layer = *(normalization_layer *)net.layers[i];
            visualize_normalization_layer(layer, buff);
        }
    } 
}

float *network_predict(network net, float *input)
{
    forward_network(net, input, 0);
    float *out = get_network_output(net);
    return out;
}

matrix network_predict_data(network net, data test)
{
    int i,j,b;
    int k = get_network_output_size(net);
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net.batch*test.X.rows, sizeof(float));
    for(i = 0; i < test.X.rows; i += net.batch){
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        float *out = network_predict(net, X);
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            for(j = 0; j < k; ++j){
                pred.vals[i+b][j] = out[j+b*k];
            }
        }
    }
    free(X);
    return pred;   
}

void print_network(network net)
{
    int i,j;
    for(i = 0; i < net.n; ++i){
        float *output = 0;
        int n = 0;
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *)net.layers[i];
            output = layer.output;
            image m = get_convolutional_image(layer);
            n = m.h*m.w*m.c;
        }
        else if(net.types[i] == MAXPOOL){
            maxpool_layer layer = *(maxpool_layer *)net.layers[i];
            output = layer.output;
            image m = get_maxpool_image(layer);
            n = m.h*m.w*m.c;
        }
        else if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *)net.layers[i];
            output = layer.output;
            n = layer.outputs;
        }
        else if(net.types[i] == SOFTMAX){
            softmax_layer layer = *(softmax_layer *)net.layers[i];
            output = layer.output;
            n = layer.inputs;
        }
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}

float network_accuracy(network net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_accuracy(d.y, guess);
    free_matrix(guess);
    return acc;
}


