#include "softmax_layer.h"
#include "blas.h"
#include "dark_cuda.h"
#include "utils.h"
#include "blas.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define SECRET_NUM -1234

void softmax_tree(float *input, int batch, int inputs, float temp, tree *hierarchy, float *output)
{
	int b;
	for (b = 0; b < batch; ++b) {
		int i;
		int count = 0;
		for (i = 0; i < hierarchy->groups; ++i) {
			int group_size = hierarchy->group_size[i];
			softmax(input + b*inputs + count, group_size, temp, output + b*inputs + count, 1);
			count += group_size;
		}
	}
}

softmax_layer make_softmax_layer(int batch, int inputs, int groups)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "softmax                                        %4d\n",  inputs);
    softmax_layer l = { (LAYER_TYPE)0 };
    l.type = SOFTMAX;
    l.batch = batch;
    l.groups = groups;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = (float*)xcalloc(inputs * batch, sizeof(float));
    l.output = (float*)xcalloc(inputs * batch, sizeof(float));
    l.delta = (float*)xcalloc(inputs * batch, sizeof(float));
    l.cost = (float*)xcalloc(1, sizeof(float));

    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;
#ifdef GPU
    l.forward_gpu = forward_softmax_layer_gpu;
    l.backward_gpu = backward_softmax_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.loss_gpu = cuda_make_array(l.loss, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
#endif
    return l;
}

void forward_softmax_layer(const softmax_layer l, network_state net)
{
    if(l.softmax_tree){
        int i;
        int count = 0;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_cpu(net.input + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output + count);
            count += group_size;
        }
    } else {
        softmax_cpu(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output);
    }

    if(net.truth && !l.noloss){
        softmax_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}

void backward_softmax_layer(const softmax_layer l, network_state net)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void pull_softmax_layer_output(const softmax_layer layer)
{
    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs*layer.batch);
}

void forward_softmax_layer_gpu(const softmax_layer l, network_state net)
{
    if(l.softmax_tree){
		softmax_tree_gpu(net.input, 1, l.batch, l.inputs, l.temperature, l.output_gpu, *l.softmax_tree);
		/*
		int i;
		int count = 0;
		for (i = 0; i < l.softmax_tree->groups; ++i) {
		int group_size = l.softmax_tree->group_size[i];
		softmax_gpu(net.input_gpu + count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output_gpu + count);
		count += group_size;
		}
		*/
    } else {
        if(l.spatial){
			softmax_gpu_new_api(net.input, l.c, l.batch*l.c, l.inputs/l.c, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu);
        }else{
			softmax_gpu_new_api(net.input, l.inputs/l.groups, l.batch, l.inputs, l.groups, l.inputs/l.groups, 1, l.temperature, l.output_gpu);
        }
    }
    if(net.truth && !l.noloss){
        softmax_x_ent_gpu(l.batch*l.inputs, l.output_gpu, net.truth, l.delta_gpu, l.loss_gpu);
        if(l.softmax_tree){
			mask_gpu_new_api(l.batch*l.inputs, l.delta_gpu, SECRET_NUM, net.truth, 0);
			mask_gpu_new_api(l.batch*l.inputs, l.loss_gpu, SECRET_NUM, net.truth, 0);
        }
        cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
        l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    }
}

void backward_softmax_layer_gpu(const softmax_layer layer, network_state net)
{
	axpy_ongpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, net.delta, 1);
}

#endif

// -------------------------------------


contrastive_layer make_contrastive_layer(int batch, int w, int h, int n, int classes, int inputs)
{
    fprintf(stderr, "contrastive   %4d x%4d x%4d - batch: %4d \t classes = %4d \n", w, h, n, batch, classes);
    contrastive_layer l = { (LAYER_TYPE)0 };
    l.type = CONTRASTIVE;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.w = w;
    l.h = h;
    l.c = n;
    l.n = n;
    l.classes = classes;
    l.temperature = 1;
    l.loss = (float*)xcalloc(1, sizeof(float));
    l.output = (float*)xcalloc(inputs * batch, sizeof(float));
    l.delta = (float*)xcalloc(inputs * batch, sizeof(float));
    l.cost = (float*)xcalloc(1, sizeof(float));
    l.labels = (int*)xcalloc(l.batch, sizeof(int));
    l.cos_sim = (float*)xcalloc(l.batch*l.batch, sizeof(float));
    l.p_constrastive = (float*)xcalloc(l.batch*l.batch, sizeof(float));

    l.forward = forward_contrastive_layer;
    l.backward = backward_contrastive_layer;
#ifdef GPU
    l.forward_gpu = forward_contrastive_layer_gpu;
    l.backward_gpu = backward_contrastive_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
    //l.cos_sim_gpu = cuda_make_array(l.cos_sim, l.batch*l.batch);
#endif
    return l;
}


void forward_contrastive_layer(contrastive_layer l, network_state state)
{
    if (!state.train) return;
    const float truth_thresh = state.net.label_smooth_eps;

    int b, w, h;

    memset(l.delta, 0, l.batch*l.inputs * sizeof(float));
    for (b = 0; b < l.batch; ++b) {
        if (state.net.adversarial) l.labels[b] = b % 2;
        else l.labels[b] = b / 2;
    }

    // set labels
    for (b = 0; b < l.batch; ++b) {
        for (h = 0; h < l.h; ++h) {
            for (w = 0; w < l.w; ++w)
            {
                // find truth with max prob (only 1 label even if mosaic is used)
                float max_truth = 0;
                int n;
                for (n = 0; n < l.classes; ++n) {
                    const float truth_prob = state.truth[b*l.classes + n];
                    //printf(" truth_prob = %f, ", truth_prob);
                    //if (truth_prob > max_truth)
                    if (truth_prob > truth_thresh)
                    {
                        //printf(" truth_prob = %f, max_truth = %f, n = %d; ", truth_prob, max_truth, n);
                        max_truth = truth_prob;
                        l.labels[b] = n;
                    }
                }
                //printf(", l.labels[b] = %d ", l.labels[b]);
            }
        }
    }
    //printf("\n\n");

    // set pointers to features
    float **z = (float**)xcalloc(l.batch, sizeof(float*));
    for (b = 0; b < l.batch; ++b) {
        for (h = 0; h < l.h; ++h) {
            for (w = 0; w < l.w; ++w)
            {
                z[b] = state.input + b*l.inputs;
            }
        }
    }

    // precalculate cosine similiraty
    for (b = 0; b < l.batch; ++b) {
        int b2;
        for (b2 = 0; b2 < l.batch; ++b2) {
            const float sim = cosine_similarity(z[b], z[b2], l.n);
            l.cos_sim[b*l.batch + b2] = sim;
            //if (sim > 1.001 || sim < -1) {
            //    printf(" sim = %f, ", sim); getchar();
           //}
        }
    }

    // show near sim
    float good_contrast = 0;
    for (b = 0; b < l.batch; b += 2) {
        float same = l.cos_sim[b*l.batch + b];
        float aug = l.cos_sim[b*l.batch + b + 1];
        float diff = l.cos_sim[b*l.batch + b + 2];
        good_contrast += (aug > diff);
        //printf(" l.labels[b] = %d, l.labels[b+1] = %d, l.labels[b+2] = %d, b = %d \n", l.labels[b], l.labels[b + 1], l.labels[b + 2], b);
        //printf(" same = %f, aug = %f, diff = %f, (aug > diff) = %d \n", same, aug, diff, (aug > diff));
    }
    *l.loss = 100 * good_contrast / (l.batch / 2);
    printf(" Contrast accuracy = %f %% \n", *l.loss);

    // precalculate P_contrastive
    for (b = 0; b < l.batch; ++b) {
        int b2;
        for (b2 = 0; b2 < l.batch; ++b2) {
            if (b != b2) {
                const float P = P_constrastive(b, b2, l.labels, l.batch, z, l.n, l.temperature, l.cos_sim);
                l.p_constrastive[b*l.batch + b2] = P;
                if (P > 1 || P < -1) {
                    printf(" p = %f, ", P); getchar();
                }
            }
        }
    }

    // calc deltas
    for (b = 0; b < l.batch; ++b) {
        for (h = 0; h < l.h; ++h) {
            for (w = 0; w < l.w; ++w)
            {
                //printf(" b = %d, ", b);
                // positive
                grad_contrastive_loss_positive(b, l.labels, l.batch, z, l.n, l.temperature, l.cos_sim, l.p_constrastive, l.delta + b*l.inputs);

                // negative
                grad_contrastive_loss_negative(b, l.labels, l.batch, z, l.n, l.temperature, l.cos_sim, l.p_constrastive, l.delta + b*l.inputs);
            }
        }
    }

    *(l.cost) = pow(mag_array(l.delta, l.inputs * l.batch), 2);
    if (state.net.adversarial) {
        printf(" adversarial contrastive loss = %f \n\n", *(l.cost));
    }

    free(z);
}

void backward_contrastive_layer(contrastive_layer l, network_state state)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, state.delta, 1);
}


#ifdef GPU

void pull_contrastive_layer_output(const contrastive_layer l)
{
    cuda_pull_array(l.output_gpu, l.output, l.inputs*l.batch);
}

void push_contrastive_layer_output(const contrastive_layer l)
{
    cuda_push_array(l.delta_gpu, l.delta, l.inputs*l.batch);
}


void forward_contrastive_layer_gpu(contrastive_layer l, network_state state)
{
    simple_copy_ongpu(l.batch*l.inputs, state.input, l.output_gpu);
    if (!state.train) return;

    float *in_cpu = (float *)xcalloc(l.batch*l.inputs, sizeof(float));
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
    memcpy(in_cpu, l.output, l.batch*l.outputs * sizeof(float));
    float *truth_cpu = 0;
    if (state.truth) {
        int num_truth = l.batch*l.classes;
        truth_cpu = (float *)xcalloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    network_state cpu_state = state;
    cpu_state.net = state.net;
    cpu_state.index = state.index;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;

    forward_contrastive_layer(l, cpu_state);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);

    free(in_cpu);
    if (cpu_state.truth) free(cpu_state.truth);
}

void backward_contrastive_layer_gpu(contrastive_layer layer, network_state state)
{
    axpy_ongpu(layer.batch*layer.inputs, 1, layer.delta_gpu, 1, state.delta, 1);
}

#endif