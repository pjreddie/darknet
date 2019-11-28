#include "iseg_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_iseg_layer(int batch, int w, int h, int classes, int ids)
{
    layer l = {0};
    l.type = ISEG;

    l.h = h;
    l.w = w;
    l.c = classes + ids;
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.batch = batch;
    l.extra = ids;
    l.cost = calloc(1, sizeof(float));
    l.outputs = h*w*l.c;
    l.inputs = l.outputs;
    l.truths = 90*(l.w*l.h+1);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));

    l.counts = calloc(90, sizeof(int));
    l.sums = calloc(90, sizeof(float*));
    if(ids){
        int i;
        for(i = 0; i < 90; ++i){
            l.sums[i] = calloc(ids, sizeof(float));
        }
    }

    l.forward = forward_iseg_layer;
    l.backward = backward_iseg_layer;
#ifdef GPU
    l.forward_gpu = forward_iseg_layer_gpu;
    l.backward_gpu = backward_iseg_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "iseg\n");
    srand(0);

    return l;
}

void resize_iseg_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->c;
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

void forward_iseg_layer(const layer l, network net)
{

    double time = what_time_is_it_now();
    int i,b,j,k;
    int ids = l.extra;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        int index = b*l.outputs;
        activate_array(l.output + index, l.classes*l.w*l.h, LOGISTIC);
    }
#endif

    for (b = 0; b < l.batch; ++b){
        // a priori, each pixel has no class
        for(i = 0; i < l.classes; ++i){
            for(k = 0; k < l.w*l.h; ++k){
                int index = b*l.outputs + i*l.w*l.h + k;
                l.delta[index] = 0 - l.output[index];
            }
        }

        // a priori, embedding should be small magnitude
        for(i = 0; i < ids; ++i){
            for(k = 0; k < l.w*l.h; ++k){
                int index = b*l.outputs + (i+l.classes)*l.w*l.h + k;
                l.delta[index] = .1 * (0 - l.output[index]);
            }
        }


        memset(l.counts, 0, 90*sizeof(int));
        for(i = 0; i < 90; ++i){
            fill_cpu(ids, 0, l.sums[i], 1);
            
            int c = net.truth[b*l.truths + i*(l.w*l.h+1)];
            if(c < 0) break;
            // add up metric embeddings for each instance
            for(k = 0; k < l.w*l.h; ++k){
                int index = b*l.outputs + c*l.w*l.h + k;
                float v = net.truth[b*l.truths + i*(l.w*l.h + 1) + 1 + k];
                if(v){
                    l.delta[index] = v - l.output[index];
                    axpy_cpu(ids, 1, l.output + b*l.outputs + l.classes*l.w*l.h + k, l.w*l.h, l.sums[i], 1);
                    ++l.counts[i];
                }
            }
        }

        float *mse = calloc(90, sizeof(float));
        for(i = 0; i < 90; ++i){
            int c = net.truth[b*l.truths + i*(l.w*l.h+1)];
            if(c < 0) break;
            for(k = 0; k < l.w*l.h; ++k){
                float v = net.truth[b*l.truths + i*(l.w*l.h + 1) + 1 + k];
                if(v){
                    int z;
                    float sum = 0;
                    for(z = 0; z < ids; ++z){
                        int index = b*l.outputs + (l.classes + z)*l.w*l.h + k;
                        sum += pow(l.sums[i][z]/l.counts[i] - l.output[index], 2);
                    }
                    mse[i] += sum;
                }
            }
            mse[i] /= l.counts[i];
        }

        // Calculate average embedding
        for(i = 0; i < 90; ++i){
            if(!l.counts[i]) continue;
            scal_cpu(ids, 1.f/l.counts[i], l.sums[i], 1);
            if(b == 0 && net.gpu_index == 0){
                printf("%4d, %6.3f, ", l.counts[i], mse[i]);
                for(j = 0; j < ids; ++j){
                    printf("%6.3f,", l.sums[i][j]);
                }
                printf("\n");
            }
        }
        free(mse);

        // Calculate embedding loss
        for(i = 0; i < 90; ++i){
            if(!l.counts[i]) continue;
            for(k = 0; k < l.w*l.h; ++k){
                float v = net.truth[b*l.truths + i*(l.w*l.h + 1) + 1 + k];
                if(v){
                    for(j = 0; j < 90; ++j){
                        if(!l.counts[j])continue;
                        int z;
                        for(z = 0; z < ids; ++z){
                            int index = b*l.outputs + (l.classes + z)*l.w*l.h + k;
                            float diff = l.sums[j][z] - l.output[index];
                            if (j == i) l.delta[index] +=   diff < 0? -.1 : .1;
                            else        l.delta[index] += -(diff < 0? -.1 : .1);
                        }
                    }
                }
            }
        }

        for(i = 0; i < ids; ++i){
            for(k = 0; k < l.w*l.h; ++k){
                int index = b*l.outputs + (i+l.classes)*l.w*l.h + k;
                l.delta[index] *= .01;
            }
        }
    }

    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("took %lf sec\n", what_time_is_it_now() - time);
}

void backward_iseg_layer(const layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void forward_iseg_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b;
    for (b = 0; b < l.batch; ++b){
        activate_array_gpu(l.output_gpu + b*l.outputs, l.classes*l.w*l.h, LOGISTIC);
        //if(l.extra) activate_array_gpu(l.output_gpu + b*l.outputs + l.classes*l.w*l.h, l.extra*l.w*l.h, LOGISTIC);
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_iseg_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_iseg_layer_gpu(const layer l, network net)
{
    int b;
    for (b = 0; b < l.batch; ++b){
        //if(l.extra) gradient_array_gpu(l.output_gpu + b*l.outputs + l.classes*l.w*l.h, l.extra*l.w*l.h, LOGISTIC, l.delta_gpu + b*l.outputs + l.classes*l.w*l.h);
    }
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

