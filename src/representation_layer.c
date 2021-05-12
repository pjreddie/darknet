#include "representation_layer.h"
#include "utils.h"
#include "dark_cuda.h"
#include "blas.h"
#include <stdio.h>
#include <assert.h>

layer make_implicit_layer(int batch, int index, float mean_init, float std_init, int filters, int atoms)
{
    fprintf(stderr,"implicit Layer: %d x %d \t mean=%.2f, std=%.2f \n", filters, atoms, mean_init, std_init);
    layer l = { (LAYER_TYPE)0 };
    l.type = IMPLICIT;
    l.batch = batch;
    l.w = 1;
    l.h = 1;
    l.c = 1;

    l.out_w = 1;
    l.out_h = atoms;
    l.out_c = filters;

    l.outputs = l.out_w*l.out_h*l.out_c;
    l.inputs = 1;
    l.index = index;

    l.nweights = l.out_w * l.out_h * l.out_c;

    l.weight_updates = (float*)xcalloc(l.nweights, sizeof(float));
    l.weights = (float*)xcalloc(l.nweights, sizeof(float));
    int i;
    for (i = 0; i < l.nweights; ++i) l.weights[i] = mean_init + rand_uniform(-std_init, std_init);


    l.delta = (float*)xcalloc(l.outputs * batch, sizeof(float));
    l.output = (float*)xcalloc(l.outputs * batch, sizeof(float));

    l.forward = forward_implicit_layer;
    l.backward = backward_implicit_layer;
    l.update = update_implicit_layer;
#ifdef GPU
    l.forward_gpu = forward_implicit_layer_gpu;
    l.backward_gpu = backward_implicit_layer_gpu;
    l.update_gpu = update_implicit_layer_gpu;

    l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);

    l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
    l.weights_gpu = cuda_make_array(l.weights, l.nweights);
#endif
    return l;
}

void resize_implicit_layer(layer *l, int w, int h)
{
}

void forward_implicit_layer(const layer l, network_state state)
{
    int i;
    #pragma omp parallel for
    for (i = 0; i < l.nweights * l.batch; ++i) {
        l.output[i] = l.weights[i % l.nweights];
    }
}

void backward_implicit_layer(const layer l, network_state state)
{
    int i;
    for (i = 0; i < l.nweights * l.batch; ++i) {
        l.weight_updates[i % l.nweights] += l.delta[i];
    }
}

void update_implicit_layer(layer l, int batch, float learning_rate_init, float momentum, float decay)
{
    float learning_rate = learning_rate_init*l.learning_rate_scale;
    //float momentum = a.momentum;
    //float decay = a.decay;
    //int batch = a.batch;

    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);

}


#ifdef GPU
void forward_implicit_layer_gpu(const layer l, network_state state)
{
    forward_implicit_gpu(l.batch, l.nweights, l.weights_gpu, l.output_gpu);
}

void backward_implicit_layer_gpu(const layer l, network_state state)
{
    backward_implicit_gpu(l.batch, l.nweights, l.weight_updates_gpu, l.delta_gpu);
}

void update_implicit_layer_gpu(layer l, int batch, float learning_rate_init, float momentum, float decay, float loss_scale)
{
    // Loss scale for Mixed-Precision on Tensor-Cores
    float learning_rate = learning_rate_init*l.learning_rate_scale / loss_scale;
    //float momentum = a.momentum;
    //float decay = a.decay;
    //int batch = a.batch;

    reset_nan_and_inf(l.weight_updates_gpu, l.nweights);
    fix_nan_and_inf(l.weights_gpu, l.nweights);

    if (l.adam) {
        //adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, l.B1, l.B2, l.eps, decay, learning_rate, l.nweights, batch, l.t);
    }
    else {
        //axpy_ongpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        //axpy_ongpu(l.nweights, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        //scal_ongpu(l.nweights, momentum, l.weight_updates_gpu, 1);

        axpy_ongpu(l.nweights, -decay*batch*loss_scale, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_ongpu(l.nweights, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);

        scal_ongpu(l.nweights, momentum, l.weight_updates_gpu, 1);
    }

    if (l.clip) {
        constrain_ongpu(l.nweights, l.clip, l.weights_gpu, 1);
    }
}

void pull_implicit_layer(layer l)
{
    cuda_pull_array_async(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array_async(l.weight_updates_gpu, l.weight_updates, l.nweights);

    if (l.adam) {
        cuda_pull_array_async(l.m_gpu, l.m, l.nweights);
        cuda_pull_array_async(l.v_gpu, l.v, l.nweights);
    }
    CHECK_CUDA(cudaPeekAtLastError());
    cudaStreamSynchronize(get_cuda_stream());
}

void push_implicit_layer(layer l)
{
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);

    if (l.train) {
        cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    }
    if (l.adam) {
        cuda_push_array(l.m_gpu, l.m, l.nweights);
        cuda_push_array(l.v_gpu, l.v, l.nweights);
    }
    CHECK_CUDA(cudaPeekAtLastError());
}
#endif


