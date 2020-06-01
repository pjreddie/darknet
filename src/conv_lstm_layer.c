// Page 4: https://arxiv.org/abs/1506.04214v2
// Page 3: https://arxiv.org/pdf/1705.06368v3.pdf
// https://wikimedia.org/api/rest_v1/media/math/render/svg/1edbece2559479959fe829e9c6657efb380debe7

#include "conv_lstm_layer.h"
#include "connected_layer.h"
#include "convolutional_layer.h"
#include "utils.h"
#include "dark_cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void increment_layer(layer *l, int steps)
{
    int num = l->outputs*l->batch*steps;
    l->output += num;
    l->delta += num;
    l->x += num;
    l->x_norm += num;

#ifdef GPU
    l->output_gpu += num;
    l->delta_gpu += num;
    l->x_gpu += num;
    l->x_norm_gpu += num;
#endif
}


layer make_conv_lstm_layer(int batch, int h, int w, int c, int output_filters, int groups, int steps, int size, int stride, int dilation, int pad, ACTIVATION activation, int batch_normalize, int peephole, int xnor, int bottleneck, int train)
{
    fprintf(stderr, "CONV_LSTM Layer: %d x %d x %d image, %d filters\n", h, w, c, output_filters);
    /*
    batch = batch / steps;
    layer l = { (LAYER_TYPE)0 };
    l.batch = batch;
    l.type = LSTM;
    l.steps = steps;
    l.inputs = inputs;
    l.out_w = 1;
    l.out_h = 1;
    l.out_c = outputs;
    */
    batch = batch / steps;
    layer l = { (LAYER_TYPE)0 };
    l.train = train;
    l.batch = batch;
    l.type = CONV_LSTM;
    l.bottleneck = bottleneck;
    l.steps = steps;
    l.size = size;
    l.stride = stride;
    l.dilation = dilation;
    l.pad = pad;
    l.h = h;
    l.w = w;
    l.c = c;
    l.groups = groups;
    l.out_c = output_filters;
    l.inputs = h * w * c;
    l.xnor = xnor;
    l.peephole = peephole;

    // U
    l.uf = (layer*)xcalloc(1, sizeof(layer));
    *(l.uf) = make_convolutional_layer(batch, steps, h, w, c, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
    l.uf->batch = batch;
    if (l.workspace_size < l.uf->workspace_size) l.workspace_size = l.uf->workspace_size;

    l.ui = (layer*)xcalloc(1, sizeof(layer));
    *(l.ui) = make_convolutional_layer(batch, steps, h, w, c, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
    l.ui->batch = batch;
    if (l.workspace_size < l.ui->workspace_size) l.workspace_size = l.ui->workspace_size;

    l.ug = (layer*)xcalloc(1, sizeof(layer));
    *(l.ug) = make_convolutional_layer(batch, steps, h, w, c, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
    l.ug->batch = batch;
    if (l.workspace_size < l.ug->workspace_size) l.workspace_size = l.ug->workspace_size;

    l.uo = (layer*)xcalloc(1, sizeof(layer));
    *(l.uo) = make_convolutional_layer(batch, steps, h, w, c, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
    l.uo->batch = batch;
    if (l.workspace_size < l.uo->workspace_size) l.workspace_size = l.uo->workspace_size;

    if (l.bottleneck) {
        // bottleneck-conv with 2x channels
        l.wf = (layer*)xcalloc(1, sizeof(layer));
        l.wi = (layer*)xcalloc(1, sizeof(layer));
        l.wg = (layer*)xcalloc(1, sizeof(layer));
        l.wo = (layer*)xcalloc(1, sizeof(layer));
        *(l.wf) = make_convolutional_layer(batch, steps, h, w, output_filters*2, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
        l.wf->batch = batch;
        if (l.workspace_size < l.wf->workspace_size) l.workspace_size = l.wf->workspace_size;
    }
    else {
        // W
        l.wf = (layer*)xcalloc(1, sizeof(layer));
        *(l.wf) = make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
        l.wf->batch = batch;
        if (l.workspace_size < l.wf->workspace_size) l.workspace_size = l.wf->workspace_size;

        l.wi = (layer*)xcalloc(1, sizeof(layer));
        *(l.wi) = make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
        l.wi->batch = batch;
        if (l.workspace_size < l.wi->workspace_size) l.workspace_size = l.wi->workspace_size;

        l.wg = (layer*)xcalloc(1, sizeof(layer));
        *(l.wg) = make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
        l.wg->batch = batch;
        if (l.workspace_size < l.wg->workspace_size) l.workspace_size = l.wg->workspace_size;

        l.wo = (layer*)xcalloc(1, sizeof(layer));
        *(l.wo) = make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
        l.wo->batch = batch;
        if (l.workspace_size < l.wo->workspace_size) l.workspace_size = l.wo->workspace_size;
    }

    // V
    l.vf = (layer*)xcalloc(1, sizeof(layer));
    if (l.peephole) {
        *(l.vf) = make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
        l.vf->batch = batch;
        if (l.workspace_size < l.vf->workspace_size) l.workspace_size = l.vf->workspace_size;
    }

    l.vi = (layer*)xcalloc(1, sizeof(layer));
    if (l.peephole) {
        *(l.vi) = make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
        l.vi->batch = batch;
        if (l.workspace_size < l.vi->workspace_size) l.workspace_size = l.vi->workspace_size;
    }

    l.vo = (layer*)xcalloc(1, sizeof(layer));
    if (l.peephole) {
        *(l.vo) = make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, 0, xnor, 0, 0, 0, 0, NULL, 0, 0, train);
        l.vo->batch = batch;
        if (l.workspace_size < l.vo->workspace_size) l.workspace_size = l.vo->workspace_size;
    }


    l.batch_normalize = batch_normalize;

    l.out_h = l.uo->out_h;
    l.out_w = l.uo->out_w;
    l.outputs = l.uo->outputs;
    int outputs = l.outputs;
    l.inputs = w*h*c;

    if (!l.bottleneck) assert(l.wo->outputs == l.uo->outputs);
    assert(l.wf->outputs == l.uf->outputs);

    l.output = (float*)xcalloc(outputs * batch * steps, sizeof(float));
    //l.state = (float*)xcalloc(outputs * batch, sizeof(float));

    l.forward = forward_conv_lstm_layer;
    l.update = update_conv_lstm_layer;
    l.backward = backward_conv_lstm_layer;

    l.prev_state_cpu =  (float*)xcalloc(batch*outputs, sizeof(float));
    l.prev_cell_cpu =   (float*)xcalloc(batch*outputs, sizeof(float));
    l.cell_cpu =        (float*)xcalloc(batch*outputs*steps, sizeof(float));

    l.f_cpu =           (float*)xcalloc(batch*outputs, sizeof(float));
    l.i_cpu =           (float*)xcalloc(batch*outputs, sizeof(float));
    l.g_cpu =           (float*)xcalloc(batch*outputs, sizeof(float));
    l.o_cpu =           (float*)xcalloc(batch*outputs, sizeof(float));
    l.c_cpu =           (float*)xcalloc(batch*outputs, sizeof(float));
    l.stored_c_cpu = (float*)xcalloc(batch*outputs, sizeof(float));
    l.h_cpu =           (float*)xcalloc(batch*outputs, sizeof(float));
    l.stored_h_cpu = (float*)xcalloc(batch*outputs, sizeof(float));
    l.temp_cpu =        (float*)xcalloc(batch*outputs, sizeof(float));
    l.temp2_cpu =       (float*)xcalloc(batch*outputs, sizeof(float));
    l.temp3_cpu =       (float*)xcalloc(batch*outputs, sizeof(float));
    l.dc_cpu =          (float*)xcalloc(batch*outputs, sizeof(float));
    l.dh_cpu =          (float*)xcalloc(batch*outputs, sizeof(float));

#ifdef GPU
    l.forward_gpu = forward_conv_lstm_layer_gpu;
    l.backward_gpu = backward_conv_lstm_layer_gpu;
    l.update_gpu = update_conv_lstm_layer_gpu;

    //l.state_gpu = cuda_make_array(l.state, batch*l.outputs);

    l.output_gpu = cuda_make_array(0, batch*outputs*steps);
    l.delta_gpu = cuda_make_array(0, batch*l.outputs*steps);

    l.prev_state_gpu = cuda_make_array(0, batch*outputs);
    l.prev_cell_gpu = cuda_make_array(0, batch*outputs);
    l.cell_gpu = cuda_make_array(0, batch*outputs*steps);

    l.f_gpu = cuda_make_array(0, batch*outputs);
    l.i_gpu = cuda_make_array(0, batch*outputs);
    l.g_gpu = cuda_make_array(0, batch*outputs);
    l.o_gpu = cuda_make_array(0, batch*outputs);
    l.c_gpu = cuda_make_array(0, batch*outputs);
    if (l.bottleneck) {
        l.bottelneck_hi_gpu = cuda_make_array(0, batch*outputs * 2);
        l.bottelneck_delta_gpu = cuda_make_array(0, batch*outputs * 2);
    }
    l.h_gpu = cuda_make_array(0, batch*outputs);
    l.stored_c_gpu = cuda_make_array(0, batch*outputs);
    l.stored_h_gpu = cuda_make_array(0, batch*outputs);
    l.temp_gpu =  cuda_make_array(0, batch*outputs);
    l.temp2_gpu = cuda_make_array(0, batch*outputs);
    l.temp3_gpu = cuda_make_array(0, batch*outputs);
    l.dc_gpu = cuda_make_array(0, batch*outputs);
    l.dh_gpu = cuda_make_array(0, batch*outputs);
    l.last_prev_state_gpu = cuda_make_array(0, l.batch*l.outputs);
    l.last_prev_cell_gpu = cuda_make_array(0, l.batch*l.outputs);
#endif

    l.bflops = l.uf->bflops + l.ui->bflops + l.ug->bflops + l.uo->bflops +
        l.wf->bflops + l.wi->bflops + l.wg->bflops + l.wo->bflops +
        l.vf->bflops + l.vi->bflops + l.vo->bflops;

    if(l.peephole) l.bflops += 12 * l.outputs*l.batch / 1000000000.;
    else l.bflops += 9 * l.outputs*l.batch / 1000000000.;

    return l;
}

void update_conv_lstm_layer(layer l, int batch, float learning_rate, float momentum, float decay)
{
    if (l.peephole) {
        update_convolutional_layer(*(l.vf), batch, learning_rate, momentum, decay);
        update_convolutional_layer(*(l.vi), batch, learning_rate, momentum, decay);
        update_convolutional_layer(*(l.vo), batch, learning_rate, momentum, decay);
    }
    update_convolutional_layer(*(l.wf), batch, learning_rate, momentum, decay);
    update_convolutional_layer(*(l.wi), batch, learning_rate, momentum, decay);
    update_convolutional_layer(*(l.wg), batch, learning_rate, momentum, decay);
    update_convolutional_layer(*(l.wo), batch, learning_rate, momentum, decay);
    update_convolutional_layer(*(l.uf), batch, learning_rate, momentum, decay);
    update_convolutional_layer(*(l.ui), batch, learning_rate, momentum, decay);
    update_convolutional_layer(*(l.ug), batch, learning_rate, momentum, decay);
    update_convolutional_layer(*(l.uo), batch, learning_rate, momentum, decay);
}

void resize_conv_lstm_layer(layer *l, int w, int h)
{
    if (l->peephole) {
        resize_convolutional_layer(l->vf, w, h);
        if (l->workspace_size < l->vf->workspace_size) l->workspace_size = l->vf->workspace_size;

        resize_convolutional_layer(l->vi, w, h);
        if (l->workspace_size < l->vi->workspace_size) l->workspace_size = l->vi->workspace_size;

        resize_convolutional_layer(l->vo, w, h);
        if (l->workspace_size < l->vo->workspace_size) l->workspace_size = l->vo->workspace_size;
    }

    resize_convolutional_layer(l->wf, w, h);
    if (l->workspace_size < l->wf->workspace_size) l->workspace_size = l->wf->workspace_size;

    resize_convolutional_layer(l->wi, w, h);
    if (l->workspace_size < l->wi->workspace_size) l->workspace_size = l->wi->workspace_size;

    resize_convolutional_layer(l->wg, w, h);
    if (l->workspace_size < l->wg->workspace_size) l->workspace_size = l->wg->workspace_size;

    resize_convolutional_layer(l->wo, w, h);
    if (l->workspace_size < l->wo->workspace_size) l->workspace_size = l->wo->workspace_size;


    resize_convolutional_layer(l->uf, w, h);
    if (l->workspace_size < l->uf->workspace_size) l->workspace_size = l->uf->workspace_size;

    resize_convolutional_layer(l->ui, w, h);
    if (l->workspace_size < l->ui->workspace_size) l->workspace_size = l->ui->workspace_size;

    resize_convolutional_layer(l->ug, w, h);
    if (l->workspace_size < l->ug->workspace_size) l->workspace_size = l->ug->workspace_size;

    resize_convolutional_layer(l->uo, w, h);
    if (l->workspace_size < l->uo->workspace_size) l->workspace_size = l->uo->workspace_size;

    l->w = w;
    l->h = h;
    l->out_h = l->wo->out_h;
    l->out_w = l->wo->out_w;
    l->outputs = l->wo->outputs;
    int outputs = l->outputs;
    l->inputs = w*h*l->c;
    int steps = l->steps;
    int batch = l->batch;

    assert(l->wo->outputs == l->uo->outputs);

    l->output = (float*)xrealloc(l->output, outputs * batch * steps * sizeof(float));
    //l->state = (float*)xrealloc(l->state, outputs * batch * sizeof(float));

    l->prev_state_cpu = (float*)xrealloc(l->prev_state_cpu, batch*outputs * sizeof(float));
    l->prev_cell_cpu = (float*)xrealloc(l->prev_cell_cpu, batch*outputs * sizeof(float));
    l->cell_cpu = (float*)xrealloc(l->cell_cpu, batch*outputs*steps * sizeof(float));

    l->f_cpu = (float*)xrealloc(l->f_cpu, batch*outputs * sizeof(float));
    l->i_cpu = (float*)xrealloc(l->i_cpu, batch*outputs * sizeof(float));
    l->g_cpu = (float*)xrealloc(l->g_cpu, batch*outputs * sizeof(float));
    l->o_cpu = (float*)xrealloc(l->o_cpu, batch*outputs * sizeof(float));
    l->c_cpu = (float*)xrealloc(l->c_cpu, batch*outputs * sizeof(float));
    l->h_cpu = (float*)xrealloc(l->h_cpu, batch*outputs * sizeof(float));
    l->temp_cpu = (float*)xrealloc(l->temp_cpu, batch*outputs * sizeof(float));
    l->temp2_cpu = (float*)xrealloc(l->temp2_cpu, batch*outputs * sizeof(float));
    l->temp3_cpu = (float*)xrealloc(l->temp3_cpu, batch*outputs * sizeof(float));
    l->dc_cpu = (float*)xrealloc(l->dc_cpu, batch*outputs * sizeof(float));
    l->dh_cpu = (float*)xrealloc(l->dh_cpu, batch*outputs * sizeof(float));
    l->stored_c_cpu = (float*)xrealloc(l->stored_c_cpu, batch*outputs * sizeof(float));
    l->stored_h_cpu = (float*)xrealloc(l->stored_h_cpu, batch*outputs * sizeof(float));

#ifdef GPU
    //if (l->state_gpu) cudaFree(l->state_gpu);
    //l->state_gpu = cuda_make_array(l->state, batch*l->outputs);

    if (l->output_gpu) cudaFree(l->output_gpu);
    l->output_gpu = cuda_make_array(0, batch*outputs*steps);

    if (l->delta_gpu) cudaFree(l->delta_gpu);
    l->delta_gpu = cuda_make_array(0, batch*outputs*steps);

    if (l->prev_state_gpu) cudaFree(l->prev_state_gpu);
    l->prev_state_gpu = cuda_make_array(0, batch*outputs);

    if (l->prev_cell_gpu) cudaFree(l->prev_cell_gpu);
    l->prev_cell_gpu = cuda_make_array(0, batch*outputs);

    if (l->cell_gpu) cudaFree(l->cell_gpu);
    l->cell_gpu = cuda_make_array(0, batch*outputs*steps);

    if (l->f_gpu) cudaFree(l->f_gpu);
    l->f_gpu = cuda_make_array(0, batch*outputs);

    if (l->i_gpu) cudaFree(l->i_gpu);
    l->i_gpu = cuda_make_array(0, batch*outputs);

    if (l->g_gpu) cudaFree(l->g_gpu);
    l->g_gpu = cuda_make_array(0, batch*outputs);

    if (l->o_gpu) cudaFree(l->o_gpu);
    l->o_gpu = cuda_make_array(0, batch*outputs);

    if (l->c_gpu) cudaFree(l->c_gpu);
    l->c_gpu = cuda_make_array(0, batch*outputs);

    if (l->h_gpu) cudaFree(l->h_gpu);
    l->h_gpu = cuda_make_array(0, batch*outputs);

    if (l->temp_gpu) cudaFree(l->temp_gpu);
    l->temp_gpu = cuda_make_array(0, batch*outputs);

    if (l->temp2_gpu) cudaFree(l->temp2_gpu);
    l->temp2_gpu = cuda_make_array(0, batch*outputs);

    if (l->temp3_gpu) cudaFree(l->temp3_gpu);
    l->temp3_gpu = cuda_make_array(0, batch*outputs);

    if (l->dc_gpu) cudaFree(l->dc_gpu);
    l->dc_gpu = cuda_make_array(0, batch*outputs);

    if (l->dh_gpu) cudaFree(l->dh_gpu);
    l->dh_gpu = cuda_make_array(0, batch*outputs);

    if (l->stored_c_gpu) cudaFree(l->stored_c_gpu);
    l->stored_c_gpu = cuda_make_array(0, batch*outputs);

    if (l->stored_h_gpu) cudaFree(l->stored_h_gpu);
    l->stored_h_gpu = cuda_make_array(0, batch*outputs);

    if (l->last_prev_state_gpu) cudaFree(l->last_prev_state_gpu);
    l->last_prev_state_gpu = cuda_make_array(0, batch*outputs);

    if (l->last_prev_cell_gpu) cudaFree(l->last_prev_cell_gpu);
    l->last_prev_cell_gpu = cuda_make_array(0, batch*outputs);
#endif
}

void free_state_conv_lstm(layer l)
{
    int i;
    for (i = 0; i < l.outputs * l.batch; ++i) l.h_cpu[i] = 0;
    for (i = 0; i < l.outputs * l.batch; ++i) l.c_cpu[i] = 0;

#ifdef GPU
    cuda_push_array(l.h_gpu, l.h_cpu, l.outputs * l.batch);
    cuda_push_array(l.c_gpu, l.c_cpu, l.outputs * l.batch);

    //fill_ongpu(l.outputs * l.batch, 0, l.dc_gpu, 1);   //  dont use
    //fill_ongpu(l.outputs * l.batch, 0, l.dh_gpu, 1);   //  dont use
#endif  // GPU
}

void randomize_state_conv_lstm(layer l)
{
    int i;
    for (i = 0; i < l.outputs * l.batch; ++i) l.h_cpu[i] = rand_uniform(-1, 1);
    for (i = 0; i < l.outputs * l.batch; ++i) l.c_cpu[i] = rand_uniform(-1, 1);

#ifdef GPU
    cuda_push_array(l.h_gpu, l.h_cpu, l.outputs * l.batch);
    cuda_push_array(l.c_gpu, l.c_cpu, l.outputs * l.batch);
#endif  // GPU
}


void remember_state_conv_lstm(layer l)
{
    memcpy(l.stored_c_cpu, l.c_cpu, l.outputs * l.batch * sizeof(float));
    memcpy(l.stored_h_cpu, l.h_cpu, l.outputs * l.batch * sizeof(float));

#ifdef GPU
    copy_ongpu(l.outputs*l.batch, l.c_gpu, 1, l.stored_c_gpu, 1);
    copy_ongpu(l.outputs*l.batch, l.h_gpu, 1, l.stored_h_gpu, 1);
#endif  // GPU
}

void restore_state_conv_lstm(layer l)
{
    memcpy(l.c_cpu, l.stored_c_cpu, l.outputs * l.batch * sizeof(float));
    memcpy(l.h_cpu, l.stored_h_cpu, l.outputs * l.batch * sizeof(float));

#ifdef GPU
    copy_ongpu(l.outputs*l.batch, l.stored_c_gpu, 1, l.c_gpu, 1);
    copy_ongpu(l.outputs*l.batch, l.stored_h_gpu, 1, l.h_gpu, 1);
#endif  // GPU
}

void forward_conv_lstm_layer(layer l, network_state state)
{
    network_state s = { 0 };
    s.train = state.train;
    s.workspace = state.workspace;
    s.net = state.net;
    int i;
    layer vf = *(l.vf);
    layer vi = *(l.vi);
    layer vo = *(l.vo);

    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    if (state.train) {
        if (l.peephole) {
            fill_cpu(l.outputs * l.batch * l.steps, 0, vf.delta, 1);
            fill_cpu(l.outputs * l.batch * l.steps, 0, vi.delta, 1);
            fill_cpu(l.outputs * l.batch * l.steps, 0, vo.delta, 1);
        }

        fill_cpu(l.outputs * l.batch * l.steps, 0, wf.delta, 1);
        fill_cpu(l.outputs * l.batch * l.steps, 0, wi.delta, 1);
        fill_cpu(l.outputs * l.batch * l.steps, 0, wg.delta, 1);
        fill_cpu(l.outputs * l.batch * l.steps, 0, wo.delta, 1);

        fill_cpu(l.outputs * l.batch * l.steps, 0, uf.delta, 1);
        fill_cpu(l.outputs * l.batch * l.steps, 0, ui.delta, 1);
        fill_cpu(l.outputs * l.batch * l.steps, 0, ug.delta, 1);
        fill_cpu(l.outputs * l.batch * l.steps, 0, uo.delta, 1);

        fill_cpu(l.outputs * l.batch * l.steps, 0, l.delta, 1);
    }

    for (i = 0; i < l.steps; ++i)
    {
        if (l.peephole) {
            assert(l.outputs == vf.out_w * vf.out_h * vf.out_c);
            s.input = l.c_cpu;
            forward_convolutional_layer(vf, s);
            forward_convolutional_layer(vi, s);
            // vo below
        }

        assert(l.outputs == wf.out_w * wf.out_h * wf.out_c);
        assert(wf.c == l.out_c && wi.c == l.out_c && wg.c == l.out_c && wo.c == l.out_c);

        s.input = l.h_cpu;
        forward_convolutional_layer(wf, s);
        forward_convolutional_layer(wi, s);
        forward_convolutional_layer(wg, s);
        forward_convolutional_layer(wo, s);

        assert(l.inputs == uf.w * uf.h * uf.c);
        assert(uf.c == l.c && ui.c == l.c && ug.c == l.c && uo.c == l.c);

        s.input = state.input;
        forward_convolutional_layer(uf, s);
        forward_convolutional_layer(ui, s);
        forward_convolutional_layer(ug, s);
        forward_convolutional_layer(uo, s);

        // f = wf + uf + vf
        copy_cpu(l.outputs*l.batch, wf.output, 1, l.f_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, uf.output, 1, l.f_cpu, 1);
        if (l.peephole) axpy_cpu(l.outputs*l.batch, 1, vf.output, 1, l.f_cpu, 1);

        // i = wi + ui + vi
        copy_cpu(l.outputs*l.batch, wi.output, 1, l.i_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, ui.output, 1, l.i_cpu, 1);
        if (l.peephole) axpy_cpu(l.outputs*l.batch, 1, vi.output, 1, l.i_cpu, 1);

        // g = wg + ug
        copy_cpu(l.outputs*l.batch, wg.output, 1, l.g_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, ug.output, 1, l.g_cpu, 1);

        activate_array(l.f_cpu, l.outputs*l.batch, LOGISTIC);
        activate_array(l.i_cpu, l.outputs*l.batch, LOGISTIC);
        activate_array(l.g_cpu, l.outputs*l.batch, TANH);

        // c = f*c + i*g
        copy_cpu(l.outputs*l.batch, l.i_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs*l.batch, l.g_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs*l.batch, l.f_cpu, 1, l.c_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, l.temp_cpu, 1, l.c_cpu, 1);

        // o = wo + uo + vo(c_new)
        if (l.peephole) {
            s.input = l.c_cpu;
            forward_convolutional_layer(vo, s);
        }
        copy_cpu(l.outputs*l.batch, wo.output, 1, l.o_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, uo.output, 1, l.o_cpu, 1);
        if (l.peephole) axpy_cpu(l.outputs*l.batch, 1, vo.output, 1, l.o_cpu, 1);
        activate_array(l.o_cpu, l.outputs*l.batch, LOGISTIC);

        // h = o * tanh(c)
        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.h_cpu, 1);
        activate_array(l.h_cpu, l.outputs*l.batch, TANH);
        mul_cpu(l.outputs*l.batch, l.o_cpu, 1, l.h_cpu, 1);

        if (l.state_constrain) constrain_cpu(l.outputs*l.batch, l.state_constrain, l.c_cpu);
        fix_nan_and_inf_cpu(l.c_cpu, l.outputs*l.batch);
        fix_nan_and_inf_cpu(l.h_cpu, l.outputs*l.batch);

        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.cell_cpu, 1);
        copy_cpu(l.outputs*l.batch, l.h_cpu, 1, l.output, 1);

        state.input += l.inputs*l.batch;
        l.output    += l.outputs*l.batch;
        l.cell_cpu      += l.outputs*l.batch;

        if (l.peephole) {
            increment_layer(&vf, 1);
            increment_layer(&vi, 1);
            increment_layer(&vo, 1);
        }

        increment_layer(&wf, 1);
        increment_layer(&wi, 1);
        increment_layer(&wg, 1);
        increment_layer(&wo, 1);

        increment_layer(&uf, 1);
        increment_layer(&ui, 1);
        increment_layer(&ug, 1);
        increment_layer(&uo, 1);
    }
}

void backward_conv_lstm_layer(layer l, network_state state)
{
    network_state s = { 0 };
    s.train = state.train;
    s.workspace = state.workspace;
    int i;
    layer vf = *(l.vf);
    layer vi = *(l.vi);
    layer vo = *(l.vo);

    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    if (l.peephole) {
        increment_layer(&vf, l.steps - 1);
        increment_layer(&vi, l.steps - 1);
        increment_layer(&vo, l.steps - 1);
    }

    increment_layer(&wf, l.steps - 1);
    increment_layer(&wi, l.steps - 1);
    increment_layer(&wg, l.steps - 1);
    increment_layer(&wo, l.steps - 1);

    increment_layer(&uf, l.steps - 1);
    increment_layer(&ui, l.steps - 1);
    increment_layer(&ug, l.steps - 1);
    increment_layer(&uo, l.steps - 1);

    state.input += l.inputs*l.batch*(l.steps - 1);
    if (state.delta) state.delta += l.inputs*l.batch*(l.steps - 1);

    l.output += l.outputs*l.batch*(l.steps - 1);
    l.cell_cpu += l.outputs*l.batch*(l.steps - 1);
    l.delta += l.outputs*l.batch*(l.steps - 1);

    for (i = l.steps - 1; i >= 0; --i) {
        if (i != 0) copy_cpu(l.outputs*l.batch, l.cell_cpu - l.outputs*l.batch, 1, l.prev_cell_cpu, 1);
        copy_cpu(l.outputs*l.batch, l.cell_cpu, 1, l.c_cpu, 1);
        if (i != 0) copy_cpu(l.outputs*l.batch, l.output - l.outputs*l.batch, 1, l.prev_state_cpu, 1);
        copy_cpu(l.outputs*l.batch, l.output, 1, l.h_cpu, 1);

        l.dh_cpu = (i == 0) ? 0 : l.delta - l.outputs*l.batch;

        // f = wf + uf + vf
        copy_cpu(l.outputs*l.batch, wf.output, 1, l.f_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, uf.output, 1, l.f_cpu, 1);
        if (l.peephole) axpy_cpu(l.outputs*l.batch, 1, vf.output, 1, l.f_cpu, 1);

        // i = wi + ui + vi
        copy_cpu(l.outputs*l.batch, wi.output, 1, l.i_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, ui.output, 1, l.i_cpu, 1);
        if (l.peephole) axpy_cpu(l.outputs*l.batch, 1, vi.output, 1, l.i_cpu, 1);

        // g = wg + ug
        copy_cpu(l.outputs*l.batch, wg.output, 1, l.g_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, ug.output, 1, l.g_cpu, 1);

        // o = wo + uo + vo
        copy_cpu(l.outputs*l.batch, wo.output, 1, l.o_cpu, 1);
        axpy_cpu(l.outputs*l.batch, 1, uo.output, 1, l.o_cpu, 1);
        if (l.peephole) axpy_cpu(l.outputs*l.batch, 1, vo.output, 1, l.o_cpu, 1);

        activate_array(l.f_cpu, l.outputs*l.batch, LOGISTIC);
        activate_array(l.i_cpu, l.outputs*l.batch, LOGISTIC);
        activate_array(l.g_cpu, l.outputs*l.batch, TANH);
        activate_array(l.o_cpu, l.outputs*l.batch, LOGISTIC);

        copy_cpu(l.outputs*l.batch, l.delta, 1, l.temp3_cpu, 1);

        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.temp_cpu, 1);
        activate_array(l.temp_cpu, l.outputs*l.batch, TANH);

        copy_cpu(l.outputs*l.batch, l.temp3_cpu, 1, l.temp2_cpu, 1);
        mul_cpu(l.outputs*l.batch, l.o_cpu, 1, l.temp2_cpu, 1);

        gradient_array(l.temp_cpu, l.outputs*l.batch, TANH, l.temp2_cpu);
        axpy_cpu(l.outputs*l.batch, 1, l.dc_cpu, 1, l.temp2_cpu, 1);
        // temp  = tanh(c)
        // temp2 = delta * o * grad_tanh(tanh(c))
        // temp3 = delta

        copy_cpu(l.outputs*l.batch, l.c_cpu, 1, l.temp_cpu, 1);
        activate_array(l.temp_cpu, l.outputs*l.batch, TANH);
        mul_cpu(l.outputs*l.batch, l.temp3_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.o_cpu, l.outputs*l.batch, LOGISTIC, l.temp_cpu);
        // delta for o(w,u,v):       temp  = delta * tanh(c) * grad_logistic(o)
        // delta for c,f,i,g(w,u,v): temp2 = delta * o * grad_tanh(tanh(c)) + delta_c(???)
        // delta for output:         temp3 = delta

        // o
        // delta for O(w,u,v):     temp  = delta * tanh(c) * grad_logistic(o)
        if (l.peephole) {
            copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, vo.delta, 1);
            s.input = l.cell_cpu;
            //s.delta = l.dc_cpu;
            backward_convolutional_layer(vo, s);
        }

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wo.delta, 1);
        s.input = l.prev_state_cpu;
        //s.delta = l.dh_cpu;
        backward_convolutional_layer(wo, s);

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, uo.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_convolutional_layer(uo, s);

        // g
        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs*l.batch, l.i_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.g_cpu, l.outputs*l.batch, TANH, l.temp_cpu);
        // delta for c,f,i,g(w,u,v): temp2 = (delta * o * grad_tanh(tanh(c)) + delta_c(???)) * g * grad_logistic(i)

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wg.delta, 1);
        s.input = l.prev_state_cpu;
        //s.delta = l.dh_cpu;
        backward_convolutional_layer(wg, s);

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, ug.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_convolutional_layer(ug, s);

        // i
        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs*l.batch, l.g_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.i_cpu, l.outputs*l.batch, LOGISTIC, l.temp_cpu);
        // delta for c,f,i,g(w,u,v): temp2 = (delta * o * grad_tanh(tanh(c)) + delta_c(???)) * g * grad_logistic(i)

        if (l.peephole) {
            copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, vi.delta, 1);
            s.input = l.prev_cell_cpu;
            //s.delta = l.dc_cpu;
            backward_convolutional_layer(vi, s);
        }

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wi.delta, 1);
        s.input = l.prev_state_cpu;
        //s.delta = l.dh_cpu;
        backward_convolutional_layer(wi, s);

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, ui.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_convolutional_layer(ui, s);

        // f
        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs*l.batch, l.prev_cell_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.f_cpu, l.outputs*l.batch, LOGISTIC, l.temp_cpu);
        // delta for c,f,i,g(w,u,v): temp2 = (delta * o * grad_tanh(tanh(c)) + delta_c(???)) * c * grad_logistic(f)

        if (l.peephole) {
            copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, vf.delta, 1);
            s.input = l.prev_cell_cpu;
            //s.delta = l.dc_cpu;
            backward_convolutional_layer(vf, s);
        }

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, wf.delta, 1);
        s.input = l.prev_state_cpu;
        //s.delta = l.dh_cpu;
        backward_convolutional_layer(wf, s);

        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, uf.delta, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_convolutional_layer(uf, s);

        copy_cpu(l.outputs*l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs*l.batch, l.f_cpu, 1, l.temp_cpu, 1);
        copy_cpu(l.outputs*l.batch, l.temp_cpu, 1, l.dc_cpu, 1);

        state.input -= l.inputs*l.batch;
        if (state.delta) state.delta -= l.inputs*l.batch;
        l.output -= l.outputs*l.batch;
        l.cell_cpu -= l.outputs*l.batch;
        l.delta -= l.outputs*l.batch;

        if (l.peephole) {
            increment_layer(&vf, -1);
            increment_layer(&vi, -1);
            increment_layer(&vo, -1);
        }

        increment_layer(&wf, -1);
        increment_layer(&wi, -1);
        increment_layer(&wg, -1);
        increment_layer(&wo, -1);

        increment_layer(&uf, -1);
        increment_layer(&ui, -1);
        increment_layer(&ug, -1);
        increment_layer(&uo, -1);
    }
}

#ifdef GPU
void pull_conv_lstm_layer(layer l)
{
    if (l.peephole) {
        pull_convolutional_layer(*(l.vf));
        pull_convolutional_layer(*(l.vi));
        pull_convolutional_layer(*(l.vo));
    }
    pull_convolutional_layer(*(l.wf));
    if (!l.bottleneck) {
        pull_convolutional_layer(*(l.wi));
        pull_convolutional_layer(*(l.wg));
        pull_convolutional_layer(*(l.wo));
    }
    pull_convolutional_layer(*(l.uf));
    pull_convolutional_layer(*(l.ui));
    pull_convolutional_layer(*(l.ug));
    pull_convolutional_layer(*(l.uo));
}

void push_conv_lstm_layer(layer l)
{
    if (l.peephole) {
        push_convolutional_layer(*(l.vf));
        push_convolutional_layer(*(l.vi));
        push_convolutional_layer(*(l.vo));
    }
    push_convolutional_layer(*(l.wf));
    if (!l.bottleneck) {
        push_convolutional_layer(*(l.wi));
        push_convolutional_layer(*(l.wg));
        push_convolutional_layer(*(l.wo));
    }
    push_convolutional_layer(*(l.uf));
    push_convolutional_layer(*(l.ui));
    push_convolutional_layer(*(l.ug));
    push_convolutional_layer(*(l.uo));
}

void update_conv_lstm_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay, float loss_scale)
{
    if (l.peephole) {
        update_convolutional_layer_gpu(*(l.vf), batch, learning_rate, momentum, decay, loss_scale);
        update_convolutional_layer_gpu(*(l.vi), batch, learning_rate, momentum, decay, loss_scale);
        update_convolutional_layer_gpu(*(l.vo), batch, learning_rate, momentum, decay, loss_scale);
    }
    update_convolutional_layer_gpu(*(l.wf), batch, learning_rate, momentum, decay, loss_scale);
    if (!l.bottleneck) {
        update_convolutional_layer_gpu(*(l.wi), batch, learning_rate, momentum, decay, loss_scale);
        update_convolutional_layer_gpu(*(l.wg), batch, learning_rate, momentum, decay, loss_scale);
        update_convolutional_layer_gpu(*(l.wo), batch, learning_rate, momentum, decay, loss_scale);
    }
    update_convolutional_layer_gpu(*(l.uf), batch, learning_rate, momentum, decay, loss_scale);
    update_convolutional_layer_gpu(*(l.ui), batch, learning_rate, momentum, decay, loss_scale);
    update_convolutional_layer_gpu(*(l.ug), batch, learning_rate, momentum, decay, loss_scale);
    update_convolutional_layer_gpu(*(l.uo), batch, learning_rate, momentum, decay, loss_scale);
}

void forward_conv_lstm_layer_gpu(layer l, network_state state)
{
    network_state s = { 0 };
    s.train = state.train;
    s.workspace = state.workspace;
    s.net = state.net;
    if (!state.train) s.index = state.index;  // don't use TC for training (especially without cuda_convert_f32_to_f16() )
    int i;
    layer vf = *(l.vf);
    layer vi = *(l.vi);
    layer vo = *(l.vo);

    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    if (state.train) {
        if (l.peephole) {
            fill_ongpu(l.outputs * l.batch * l.steps, 0, vf.delta_gpu, 1);
            fill_ongpu(l.outputs * l.batch * l.steps, 0, vi.delta_gpu, 1);
            fill_ongpu(l.outputs * l.batch * l.steps, 0, vo.delta_gpu, 1);
        }

        fill_ongpu(l.outputs * l.batch * l.steps, 0, wf.delta_gpu, 1);
        if (!l.bottleneck) {
            fill_ongpu(l.outputs * l.batch * l.steps, 0, wi.delta_gpu, 1);
            fill_ongpu(l.outputs * l.batch * l.steps, 0, wg.delta_gpu, 1);
            fill_ongpu(l.outputs * l.batch * l.steps, 0, wo.delta_gpu, 1);
        }

        fill_ongpu(l.outputs * l.batch * l.steps, 0, uf.delta_gpu, 1);
        fill_ongpu(l.outputs * l.batch * l.steps, 0, ui.delta_gpu, 1);
        fill_ongpu(l.outputs * l.batch * l.steps, 0, ug.delta_gpu, 1);
        fill_ongpu(l.outputs * l.batch * l.steps, 0, uo.delta_gpu, 1);

        fill_ongpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
    }

    for (i = 0; i < l.steps; ++i)
    {
        if (l.peephole) {
            assert(l.outputs == vf.out_w * vf.out_h * vf.out_c);
            s.input = l.c_gpu;
            forward_convolutional_layer_gpu(vf, s);
            forward_convolutional_layer_gpu(vi, s);
            // vo below
        }

        if (l.bottleneck) {
            // l.bottelneck_hi_gpu size is 2x
            simple_copy_ongpu(l.outputs*l.batch, l.h_gpu, l.bottelneck_hi_gpu);
            simple_copy_ongpu(l.outputs*l.batch, state.input, l.bottelneck_hi_gpu + l.outputs*l.batch);
            s.input = l.bottelneck_hi_gpu;
            forward_convolutional_layer_gpu(wf, s); // 2x input channels
            activate_array_ongpu(wf.output_gpu, l.outputs*l.batch, l.lstm_activation);
            s.input = wf.output_gpu;
        }
        else {
            assert(l.outputs == wf.out_w * wf.out_h * wf.out_c);
            assert(wf.c == l.out_c && wi.c == l.out_c && wg.c == l.out_c && wo.c == l.out_c);

            s.input = l.h_gpu;
            forward_convolutional_layer_gpu(wf, s);
            forward_convolutional_layer_gpu(wi, s);
            forward_convolutional_layer_gpu(wg, s);
            forward_convolutional_layer_gpu(wo, s);

            s.input = state.input;
        }

        assert(l.inputs == uf.w * uf.h * uf.c);
        assert(uf.c == l.c && ui.c == l.c && ug.c == l.c && uo.c == l.c);

        forward_convolutional_layer_gpu(uf, s);
        forward_convolutional_layer_gpu(ui, s);
        forward_convolutional_layer_gpu(ug, s);
        forward_convolutional_layer_gpu(uo, s);

        // f = wf + uf + vf
        add_3_arrays_activate((l.bottleneck)?NULL:wf.output_gpu, uf.output_gpu, (l.peephole)?vf.output_gpu:NULL, l.outputs*l.batch, LOGISTIC, l.f_gpu);
        //copy_ongpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1);
        //axpy_ongpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);
        //if (l.peephole) axpy_ongpu(l.outputs*l.batch, 1, vf.output_gpu, 1, l.f_gpu, 1);
        //activate_array_ongpu(l.f_gpu, l.outputs*l.batch, LOGISTIC);

        // i = wi + ui + vi
        add_3_arrays_activate((l.bottleneck)?NULL:wi.output_gpu, ui.output_gpu, (l.peephole) ? vi.output_gpu : NULL, l.outputs*l.batch, LOGISTIC, l.i_gpu);
        //copy_ongpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);
        //axpy_ongpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);
        //if (l.peephole) axpy_ongpu(l.outputs*l.batch, 1, vi.output_gpu, 1, l.i_gpu, 1);
        //activate_array_ongpu(l.i_gpu, l.outputs*l.batch, LOGISTIC);

        // g = wg + ug
        add_3_arrays_activate((l.bottleneck)?NULL:wg.output_gpu, ug.output_gpu, NULL, l.outputs*l.batch, l.lstm_activation, l.g_gpu);
        //copy_ongpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);
        //axpy_ongpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);
        //activate_array_ongpu(l.g_gpu, l.outputs*l.batch, TANH);

        // c = f*c + i*g
        sum_of_mults(l.f_gpu, l.c_gpu, l.i_gpu, l.g_gpu, l.outputs*l.batch, l.c_gpu);   // decreases mAP???
        //copy_ongpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);
        //mul_ongpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);
        //mul_ongpu(l.outputs*l.batch, l.f_gpu, 1, l.c_gpu, 1);
        //axpy_ongpu(l.outputs*l.batch, 1, l.temp_gpu, 1, l.c_gpu, 1);

        // o = wo + uo + vo(c_new)
        if (l.peephole) {
            s.input = l.c_gpu;
            forward_convolutional_layer_gpu(vo, s);
        }
        add_3_arrays_activate((l.bottleneck)?NULL:wo.output_gpu, uo.output_gpu, (l.peephole) ? vo.output_gpu : NULL, l.outputs*l.batch, LOGISTIC, l.o_gpu);
        //copy_ongpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);
        //axpy_ongpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);
        //if (l.peephole) axpy_ongpu(l.outputs*l.batch, 1, vo.output_gpu, 1, l.o_gpu, 1);
        //activate_array_ongpu(l.o_gpu, l.outputs*l.batch, LOGISTIC);

        // h = o * tanh(c)
        activate_and_mult(l.c_gpu, l.o_gpu, l.outputs*l.batch, l.lstm_activation, l.h_gpu);
        //simple_copy_ongpu(l.outputs*l.batch, l.c_gpu, l.h_gpu);
        //activate_array_ongpu(l.h_gpu, l.outputs*l.batch, TANH);
        //mul_ongpu(l.outputs*l.batch, l.o_gpu, 1, l.h_gpu, 1);

        fix_nan_and_inf(l.c_gpu, l.outputs*l.batch);    // should be fix_nan_and_inf()
        fix_nan_and_inf(l.h_gpu, l.outputs*l.batch);    // should be fix_nan_and_inf()
        if (l.state_constrain) constrain_ongpu(l.outputs*l.batch, l.state_constrain, l.c_gpu, 1);

        if(state.train) simple_copy_ongpu(l.outputs*l.batch, l.c_gpu, l.cell_gpu);
        simple_copy_ongpu(l.outputs*l.batch, l.h_gpu, l.output_gpu); // is required for both Detection and Training

        if (l.shortcut) {
            // partial residual connection
            if (l.bottleneck) axpy_ongpu(l.outputs*l.batch/2, 1, wf.output_gpu, 1, l.output_gpu, 1);
            //else axpy_ongpu(l.outputs*l.batch, 1, l.f_gpu, 1, l.output_gpu, 1);
        }

        state.input += l.inputs*l.batch;
        l.output_gpu    += l.outputs*l.batch;
        l.cell_gpu      += l.outputs*l.batch;

        if (l.peephole) {
            increment_layer(&vf, 1);
            increment_layer(&vi, 1);
            increment_layer(&vo, 1);
        }

        increment_layer(&wf, 1);
        increment_layer(&wi, 1);
        increment_layer(&wg, 1);
        increment_layer(&wo, 1);

        increment_layer(&uf, 1);
        increment_layer(&ui, 1);
        increment_layer(&ug, 1);
        increment_layer(&uo, 1);
    }
}

void backward_conv_lstm_layer_gpu(layer l, network_state state)
{
    float *last_output = l.output_gpu + l.outputs*l.batch*(l.steps - 1);
    float *last_cell = l.cell_gpu + l.outputs*l.batch*(l.steps - 1);

    network_state s = { 0 };
    s.train = state.train;
    s.workspace = state.workspace;
    s.net = state.net;
    int i;
    layer vf = *(l.vf);
    layer vi = *(l.vi);
    layer vo = *(l.vo);

    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    if (l.peephole) {
        increment_layer(&vf, l.steps - 1);
        increment_layer(&vi, l.steps - 1);
        increment_layer(&vo, l.steps - 1);
    }

    increment_layer(&wf, l.steps - 1);
    increment_layer(&wi, l.steps - 1);
    increment_layer(&wg, l.steps - 1);
    increment_layer(&wo, l.steps - 1);

    increment_layer(&uf, l.steps - 1);
    increment_layer(&ui, l.steps - 1);
    increment_layer(&ug, l.steps - 1);
    increment_layer(&uo, l.steps - 1);

    state.input += l.inputs*l.batch*(l.steps - 1);
    if (state.delta) state.delta += l.inputs*l.batch*(l.steps - 1);

    l.output_gpu += l.outputs*l.batch*(l.steps - 1);
    l.cell_gpu += l.outputs*l.batch*(l.steps - 1);
    l.delta_gpu += l.outputs*l.batch*(l.steps - 1);

    //fill_ongpu(l.outputs * l.batch, 0, l.dc_gpu, 1);   //  dont use
    const int sequence = get_sequence_value(state.net);

    for (i = l.steps - 1; i >= 0; --i) {
        if (i != 0) simple_copy_ongpu(l.outputs*l.batch, l.cell_gpu - l.outputs*l.batch, l.prev_cell_gpu);
        //else fill_ongpu(l.outputs * l.batch, 0, l.prev_cell_gpu, 1);   //  dont use
        else if (state.net.current_subdivision % sequence != 0) simple_copy_ongpu(l.outputs*l.batch, l.last_prev_cell_gpu, l.prev_cell_gpu);

        simple_copy_ongpu(l.outputs*l.batch, l.cell_gpu, l.c_gpu);

        if (i != 0) simple_copy_ongpu(l.outputs*l.batch, l.output_gpu - l.outputs*l.batch, l.prev_state_gpu);
        //else fill_ongpu(l.outputs * l.batch, 0, l.prev_state_gpu, 1);   //  dont use
        else if (state.net.current_subdivision % sequence != 0) simple_copy_ongpu(l.outputs*l.batch, l.last_prev_state_gpu, l.prev_state_gpu);

        simple_copy_ongpu(l.outputs*l.batch, l.output_gpu, l.h_gpu);

        l.dh_gpu = (i == 0) ? 0 : l.delta_gpu - l.outputs*l.batch;

        // f = wf + uf + vf
        add_3_arrays_activate((l.bottleneck) ? NULL : wf.output_gpu, uf.output_gpu, (l.peephole) ? vf.output_gpu : NULL, l.outputs*l.batch, LOGISTIC, l.f_gpu);
        //copy_ongpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1);
        //axpy_ongpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);
        //if (l.peephole) axpy_ongpu(l.outputs*l.batch, 1, vf.output_gpu, 1, l.f_gpu, 1);
        //activate_array_ongpu(l.f_gpu, l.outputs*l.batch, LOGISTIC);

        // i = wi + ui + vi
        add_3_arrays_activate((l.bottleneck) ? NULL : wi.output_gpu, ui.output_gpu, (l.peephole) ? vi.output_gpu : NULL, l.outputs*l.batch, LOGISTIC, l.i_gpu);
        //copy_ongpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);
        //axpy_ongpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);
        //if (l.peephole) axpy_ongpu(l.outputs*l.batch, 1, vi.output_gpu, 1, l.i_gpu, 1);
        //activate_array_ongpu(l.i_gpu, l.outputs*l.batch, LOGISTIC);

        // g = wg + ug
        add_3_arrays_activate((l.bottleneck) ? NULL : wg.output_gpu, ug.output_gpu, NULL, l.outputs*l.batch, l.lstm_activation, l.g_gpu);   // TANH
        //copy_ongpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);
        //axpy_ongpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);
        //activate_array_ongpu(l.g_gpu, l.outputs*l.batch, l.lstm_activation);

        // o = wo + uo + vo
        add_3_arrays_activate((l.bottleneck) ? NULL : wo.output_gpu, uo.output_gpu, (l.peephole) ? vo.output_gpu : NULL, l.outputs*l.batch, LOGISTIC, l.o_gpu);
        //copy_ongpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);
        //axpy_ongpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);
        //if (l.peephole) axpy_ongpu(l.outputs*l.batch, 1, vo.output_gpu, 1, l.o_gpu, 1);
        //activate_array_ongpu(l.o_gpu, l.outputs*l.batch, LOGISTIC);


        simple_copy_ongpu(l.outputs*l.batch, l.delta_gpu, l.temp3_gpu);  // temp3 = delta

        simple_copy_ongpu(l.outputs*l.batch, l.c_gpu, l.temp_gpu);
        activate_array_ongpu(l.temp_gpu, l.outputs*l.batch, l.lstm_activation);  // temp  = tanh(c)

        simple_copy_ongpu(l.outputs*l.batch, l.temp3_gpu, l.temp2_gpu);
        mul_ongpu(l.outputs*l.batch, l.o_gpu, 1, l.temp2_gpu, 1);   // temp2 = delta * o

        gradient_array_ongpu(l.temp_gpu, l.outputs*l.batch, l.lstm_activation, l.temp2_gpu); // temp2 = delta * o * grad_tanh(tanh(c))
        //???
        axpy_ongpu(l.outputs*l.batch, 1, l.dc_gpu, 1, l.temp2_gpu, 1);          // temp2 = delta * o * grad_tanh(tanh(c)) + delta_c(???)
        // temp  = tanh(c)
        // temp2 = delta * o * grad_tanh(tanh(c)) + delta_c(???)
        // temp3 = delta

        simple_copy_ongpu(l.outputs*l.batch, l.c_gpu, l.temp_gpu);
        activate_array_ongpu(l.temp_gpu, l.outputs*l.batch, l.lstm_activation);    // temp  = tanh(c)

        mul_ongpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp_gpu, 1);  // temp  = delta * tanh(c)
        gradient_array_ongpu(l.o_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);  // temp  = delta * tanh(c) * grad_logistic(o)
        // delta for o(w,u,v):       temp  = delta * tanh(c) * grad_logistic(o)
        // delta for c,f,i,g(w,u,v): temp2 = delta * o * grad_tanh(tanh(c)) + delta_c(???)
        // delta for output:         temp3 = delta

        // o
        // delta for O(w,u,v):     temp  = delta * tanh(c) * grad_logistic(o)
        if (l.peephole) {
            simple_copy_ongpu(l.outputs*l.batch, l.temp_gpu, vo.delta_gpu);
            s.input = l.cell_gpu;
            //s.delta = l.dc_gpu;
            backward_convolutional_layer_gpu(vo, s);
        }

        if (!l.bottleneck) {
            simple_copy_ongpu(l.outputs*l.batch, l.temp_gpu, wo.delta_gpu);
            s.input = l.prev_state_gpu;
            s.delta = l.temp3_gpu;// s.delta = l.dh_gpu;
            fill_ongpu(l.outputs * l.batch, 0, l.temp3_gpu, 1);
            backward_convolutional_layer_gpu(wo, s);
        }

        simple_copy_ongpu(l.outputs*l.batch, l.temp_gpu, uo.delta_gpu);
        if (l.bottleneck) {
            s.input = wf.output_gpu;
            s.delta = wf.delta_gpu;
        }
        else {
            s.input = state.input;
            s.delta = state.delta;
        }
        backward_convolutional_layer_gpu(uo, s);

        // g
        simple_copy_ongpu(l.outputs*l.batch, l.temp2_gpu, l.temp_gpu);
        mul_ongpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);
        gradient_array_ongpu(l.g_gpu, l.outputs*l.batch, l.lstm_activation, l.temp_gpu);
        // delta for c,f,i,g(w,u,v): temp = (delta * o * grad_tanh(tanh(c)) + delta_c(???)) * i * grad_tanh(g)

        if (!l.bottleneck) {
            simple_copy_ongpu(l.outputs*l.batch, l.temp_gpu, wg.delta_gpu);
            s.input = l.prev_state_gpu;
            s.delta = l.temp3_gpu;// s.delta = l.dh_gpu;   // comment this
            backward_convolutional_layer_gpu(wg, s);  // lead to nan
        }

        simple_copy_ongpu(l.outputs*l.batch, l.temp_gpu, ug.delta_gpu);
        if (l.bottleneck) {
            s.input = wf.output_gpu;
            s.delta = wf.delta_gpu;
        }
        else {
            s.input = state.input;
            s.delta = state.delta;
        }
        backward_convolutional_layer_gpu(ug, s);

        // i
        simple_copy_ongpu(l.outputs*l.batch, l.temp2_gpu, l.temp_gpu);
        mul_ongpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);
        gradient_array_ongpu(l.i_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);
        // delta for c,f,i,g(w,u,v): temp = (delta * o * grad_tanh(tanh(c)) + delta_c(???)) * g * grad_logistic(i)

        if (l.peephole) {
            simple_copy_ongpu(l.outputs*l.batch, l.temp_gpu, vi.delta_gpu);
            s.input = l.prev_cell_gpu;
            //s.delta = l.dc_gpu;
            backward_convolutional_layer_gpu(vi, s);
        }

        if (!l.bottleneck) {
            simple_copy_ongpu(l.outputs*l.batch, l.temp_gpu, wi.delta_gpu);
            s.input = l.prev_state_gpu;
            s.delta = l.temp3_gpu;// s.delta = l.dh_gpu;   // comment this
            backward_convolutional_layer_gpu(wi, s);  // lead to nan (after 1000 it)
        }

        simple_copy_ongpu(l.outputs*l.batch, l.temp_gpu, ui.delta_gpu);
        if (l.bottleneck) {
            s.input = wf.output_gpu;
            s.delta = wf.delta_gpu;
        }
        else {
            s.input = state.input;
            s.delta = state.delta;
        }
        backward_convolutional_layer_gpu(ui, s);

        // f
        simple_copy_ongpu(l.outputs*l.batch, l.temp2_gpu, l.temp_gpu);
        mul_ongpu(l.outputs*l.batch, l.prev_cell_gpu, 1, l.temp_gpu, 1);
        gradient_array_ongpu(l.f_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);
        // delta for c,f,i,g(w,u,v): temp = (delta * o * grad_tanh(tanh(c)) + delta_c(???)) * c * grad_logistic(f)

        if (l.peephole) {
            simple_copy_ongpu(l.outputs*l.batch, l.temp_gpu, vf.delta_gpu);
            s.input = l.prev_cell_gpu;
            //s.delta = l.dc_gpu;
            backward_convolutional_layer_gpu(vf, s);
        }

        simple_copy_ongpu(l.outputs*l.batch, l.temp_gpu, uf.delta_gpu);
        if (l.bottleneck) {
            s.input = wf.output_gpu;
            s.delta = wf.delta_gpu;
        }
        else {
            s.input = state.input;
            s.delta = state.delta;
        }
        backward_convolutional_layer_gpu(uf, s);


        if (l.bottleneck) {
            // l.bottelneck_hi_gpu size is 2x
            simple_copy_ongpu(l.outputs*l.batch, l.prev_state_gpu, l.bottelneck_hi_gpu);
            simple_copy_ongpu(l.outputs*l.batch, state.input, l.bottelneck_hi_gpu + l.outputs*l.batch);
            fill_ongpu(l.outputs * l.batch * 2, 0, l.bottelneck_delta_gpu, 1);
            s.input = l.bottelneck_hi_gpu;
            s.delta = l.bottelneck_delta_gpu;
            if (l.shortcut) axpy_ongpu(l.outputs*l.batch/2, 1, l.delta_gpu, 1, wf.delta_gpu, 1);    // partial residual connection
            gradient_array_ongpu(wf.output_gpu, l.outputs*l.batch, l.lstm_activation, wf.delta_gpu);

            reset_nan_and_inf(wf.delta_gpu, l.outputs*l.batch);
            constrain_ongpu(l.outputs*l.batch, 1, wf.delta_gpu, 1);
        }
        else {
            s.input = l.prev_state_gpu;
            simple_copy_ongpu(l.outputs*l.batch, l.temp_gpu, wf.delta_gpu);
            s.delta = l.temp3_gpu;// s.delta = l.dh_gpu;
        }

        // WF
        backward_convolutional_layer_gpu(wf, s);

        if (l.bottleneck) {
            reset_nan_and_inf(l.bottelneck_delta_gpu, l.outputs*l.batch*2);
            //constrain_ongpu(l.outputs*l.batch*2, 1, l.bottelneck_delta_gpu, 1);
            if (l.dh_gpu) axpy_ongpu(l.outputs*l.batch, l.time_normalizer, l.bottelneck_delta_gpu, 1, l.dh_gpu, 1);
            axpy_ongpu(l.outputs*l.batch, 1, l.bottelneck_delta_gpu + l.outputs*l.batch, 1, state.delta, 1);    // lead to nan
        }
        else {
            axpy_ongpu(l.outputs*l.batch, l.time_normalizer, l.temp3_gpu, 1, l.dh_gpu, 1);
        }

        // c
        simple_copy_ongpu(l.outputs*l.batch, l.temp2_gpu, l.temp_gpu);
        mul_ongpu(l.outputs*l.batch, l.f_gpu, 1, l.temp_gpu, 1);
        simple_copy_ongpu(l.outputs*l.batch, l.temp_gpu, l.dc_gpu);
        reset_nan_and_inf(l.dc_gpu, l.outputs*l.batch);
        if (i != 0) reset_nan_and_inf(l.dh_gpu, l.outputs*l.batch);
        // delta for c,f,i,g(w,u,v): delta_c = temp = (delta * o * grad_tanh(tanh(c)) + delta_c(???)) * f    // (grad_linear(c)==1)

        state.input -= l.inputs*l.batch;
        if (state.delta) state.delta -= l.inputs*l.batch;   // new delta: state.delta = prev_layer.delta_gpu;
        l.output_gpu -= l.outputs*l.batch;
        l.cell_gpu -= l.outputs*l.batch;
        l.delta_gpu -= l.outputs*l.batch;

        if (l.peephole) {
            increment_layer(&vf, -1);
            increment_layer(&vi, -1);
            increment_layer(&vo, -1);
        }

        increment_layer(&wf, -1);
        increment_layer(&wi, -1);
        increment_layer(&wg, -1);
        increment_layer(&wo, -1);

        increment_layer(&uf, -1);
        increment_layer(&ui, -1);
        increment_layer(&ug, -1);
        increment_layer(&uo, -1);
    }

    simple_copy_ongpu(l.outputs*l.batch, last_output, l.last_prev_state_gpu);
    simple_copy_ongpu(l.outputs*l.batch, last_cell, l.last_prev_cell_gpu);

    // free state after each 100 iterations
    //if (get_current_batch(state.net) % 100) free_state_conv_lstm(l);  // dont use
}
#endif
