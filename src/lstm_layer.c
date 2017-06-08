#include "lstm_layer.h"
#include "connected_layer.h"
#include "utils.h"
#include "cuda.h"
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

layer make_lstm_layer(int batch, int inputs, int outputs, int steps, int batch_normalize)
{
    fprintf(stderr, "LSTM Layer: %d inputs, %d outputs\n", inputs, outputs);
    batch = batch / steps;
    layer l = { 0 };
    l.batch = batch;
    l.type = LSTM;
    l.steps = steps;
    l.inputs = inputs;

    l.uf = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.uf) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize);
    l.uf->batch = batch;

    l.wf = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wf) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize);
    l.wf->batch = batch;

    l.ui = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.ui) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize);
    l.ui->batch = batch;

    l.wi = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wi) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize);
    l.wi->batch = batch;

    l.ug = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.ug) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize);
    l.ug->batch = batch;

    l.wg = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wg) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize);
    l.wg->batch = batch;

    l.uo = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.uo) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize);
    l.uo->batch = batch;

    l.wo = malloc(sizeof(layer));
    fprintf(stderr, "\t\t");
    *(l.wo) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize);
    l.wo->batch = batch;

    l.batch_normalize = batch_normalize;
    l.outputs = outputs;

    l.output = calloc(outputs*batch*steps, sizeof(float));
    l.state = calloc(outputs*batch, sizeof(float)); 

    l.forward = forward_lstm_layer;
    l.update = update_lstm_layer;

#ifdef GPU
    l.forward_gpu = forward_lstm_layer_gpu;
    l.backward_gpu = backward_lstm_layer_gpu;
    l.update_gpu = update_lstm_layer_gpu;

    l.prev_state_gpu = cuda_make_array(0, batch*outputs);
    l.prev_cell_gpu = cuda_make_array(0, batch*outputs);

    l.output_gpu = cuda_make_array(0, batch*outputs*steps);
    l.cell_gpu = cuda_make_array(0, batch*outputs*steps);
    l.delta_gpu = cuda_make_array(0, batch*l.outputs*steps); 

    l.f_gpu = cuda_make_array(l.output, batch*outputs);
    l.i_gpu = cuda_make_array(l.output, batch*outputs);
    l.g_gpu = cuda_make_array(l.output, batch*outputs);
    l.o_gpu = cuda_make_array(l.output, batch*outputs);
    l.c_gpu = cuda_make_array(l.output, batch*outputs);
    l.h_gpu = cuda_make_array(l.output, batch*outputs);
    l.temp_gpu = cuda_make_array(l.output, batch*outputs);
    l.temp2_gpu = cuda_make_array(l.output, batch*outputs);
    l.temp3_gpu = cuda_make_array(l.output, batch*outputs);
    l.dc_gpu = cuda_make_array(l.output, batch*outputs);
    l.dh_gpu = cuda_make_array(l.output, batch*outputs);
#endif

    return l;
}

void update_lstm_layer(layer l, int batch, float learning_rate, float momentum, float decay)
{
}

void forward_lstm_layer(layer l, network state)
{
}

#ifdef GPU 
void update_lstm_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay)
{
    update_connected_layer_gpu(*(l.wf), batch, learning_rate, momentum, decay);
    update_connected_layer_gpu(*(l.wi), batch, learning_rate, momentum, decay);
    update_connected_layer_gpu(*(l.wg), batch, learning_rate, momentum, decay);
    update_connected_layer_gpu(*(l.wo), batch, learning_rate, momentum, decay);
    update_connected_layer_gpu(*(l.uf), batch, learning_rate, momentum, decay);
    update_connected_layer_gpu(*(l.ui), batch, learning_rate, momentum, decay);
    update_connected_layer_gpu(*(l.ug), batch, learning_rate, momentum, decay);
    update_connected_layer_gpu(*(l.uo), batch, learning_rate, momentum, decay);
}

void forward_lstm_layer_gpu(layer l, network state)
{ 
    network s = { 0 };
    s.train = state.train;
    int i;
    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

    fill_ongpu(l.outputs * l.batch * l.steps, 0, wf.delta_gpu, 1);
    fill_ongpu(l.outputs * l.batch * l.steps, 0, wi.delta_gpu, 1);
    fill_ongpu(l.outputs * l.batch * l.steps, 0, wg.delta_gpu, 1);
    fill_ongpu(l.outputs * l.batch * l.steps, 0, wo.delta_gpu, 1);

    fill_ongpu(l.outputs * l.batch * l.steps, 0, uf.delta_gpu, 1);
    fill_ongpu(l.outputs * l.batch * l.steps, 0, ui.delta_gpu, 1);
    fill_ongpu(l.outputs * l.batch * l.steps, 0, ug.delta_gpu, 1);
    fill_ongpu(l.outputs * l.batch * l.steps, 0, uo.delta_gpu, 1);
    if (state.train) {
        fill_ongpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1); 
    }

    for (i = 0; i < l.steps; ++i) {
        s.input = l.h_gpu;
        forward_connected_layer_gpu(wf, s);							 
        forward_connected_layer_gpu(wi, s);							 
        forward_connected_layer_gpu(wg, s);							 
        forward_connected_layer_gpu(wo, s);							 

        s.input = state.input;
        forward_connected_layer_gpu(uf, s);							 
        forward_connected_layer_gpu(ui, s);							 
        forward_connected_layer_gpu(ug, s);							 
        forward_connected_layer_gpu(uo, s);							 

        copy_ongpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1); 
        axpy_ongpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1); 

        copy_ongpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);	 
        axpy_ongpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);	 

        copy_ongpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);	 
        axpy_ongpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);	 

        copy_ongpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);	 
        axpy_ongpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);	 

        activate_array_ongpu(l.f_gpu, l.outputs*l.batch, LOGISTIC);		 
        activate_array_ongpu(l.i_gpu, l.outputs*l.batch, LOGISTIC);		 
        activate_array_ongpu(l.g_gpu, l.outputs*l.batch, TANH);			 
        activate_array_ongpu(l.o_gpu, l.outputs*l.batch, LOGISTIC);		 

        copy_ongpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);		 
        mul_ongpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);		 
        mul_ongpu(l.outputs*l.batch, l.f_gpu, 1, l.c_gpu, 1);			 
        axpy_ongpu(l.outputs*l.batch, 1, l.temp_gpu, 1, l.c_gpu, 1);	 

        copy_ongpu(l.outputs*l.batch, l.c_gpu, 1, l.h_gpu, 1);			 
        activate_array_ongpu(l.h_gpu, l.outputs*l.batch, TANH);		 
        mul_ongpu(l.outputs*l.batch, l.o_gpu, 1, l.h_gpu, 1);	 

        copy_ongpu(l.outputs*l.batch, l.c_gpu, 1, l.cell_gpu, 1);		
        copy_ongpu(l.outputs*l.batch, l.h_gpu, 1, l.output_gpu, 1); 

        state.input += l.inputs*l.batch;
        l.output_gpu += l.outputs*l.batch;
        l.cell_gpu += l.outputs*l.batch;

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

void backward_lstm_layer_gpu(layer l, network state)
{  
    network s = { 0 };
    s.train = state.train;
    int i;
    layer wf = *(l.wf);
    layer wi = *(l.wi);
    layer wg = *(l.wg);
    layer wo = *(l.wo);

    layer uf = *(l.uf);
    layer ui = *(l.ui);
    layer ug = *(l.ug);
    layer uo = *(l.uo);

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

    for (i = l.steps - 1; i >= 0; --i) { 
        if (i != 0) copy_ongpu(l.outputs*l.batch, l.cell_gpu - l.outputs*l.batch, 1, l.prev_cell_gpu, 1);
        copy_ongpu(l.outputs*l.batch, l.cell_gpu, 1, l.c_gpu, 1);
        if (i != 0) copy_ongpu(l.outputs*l.batch, l.output_gpu - l.outputs*l.batch, 1, l.prev_state_gpu, 1);
        copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.h_gpu, 1);

        l.dh_gpu = (i == 0) ? 0 : l.delta_gpu - l.outputs*l.batch;

        copy_ongpu(l.outputs*l.batch, wf.output_gpu, 1, l.f_gpu, 1);			 
        axpy_ongpu(l.outputs*l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);			 

        copy_ongpu(l.outputs*l.batch, wi.output_gpu, 1, l.i_gpu, 1);			 
        axpy_ongpu(l.outputs*l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);			 

        copy_ongpu(l.outputs*l.batch, wg.output_gpu, 1, l.g_gpu, 1);			 
        axpy_ongpu(l.outputs*l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);			 

        copy_ongpu(l.outputs*l.batch, wo.output_gpu, 1, l.o_gpu, 1);			 
        axpy_ongpu(l.outputs*l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);			 

        activate_array_ongpu(l.f_gpu, l.outputs*l.batch, LOGISTIC);			 
        activate_array_ongpu(l.i_gpu, l.outputs*l.batch, LOGISTIC);		 
        activate_array_ongpu(l.g_gpu, l.outputs*l.batch, TANH);			 
        activate_array_ongpu(l.o_gpu, l.outputs*l.batch, LOGISTIC);		 

        copy_ongpu(l.outputs*l.batch, l.delta_gpu, 1, l.temp3_gpu, 1);		 

        copy_ongpu(l.outputs*l.batch, l.c_gpu, 1, l.temp_gpu, 1);			 
        activate_array_ongpu(l.temp_gpu, l.outputs*l.batch, TANH);			 

        copy_ongpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp2_gpu, 1);		 
        mul_ongpu(l.outputs*l.batch, l.o_gpu, 1, l.temp2_gpu, 1);			 

        gradient_array_ongpu(l.temp_gpu, l.outputs*l.batch, TANH, l.temp2_gpu); 
        axpy_ongpu(l.outputs*l.batch, 1, l.dc_gpu, 1, l.temp2_gpu, 1);		 

        copy_ongpu(l.outputs*l.batch, l.c_gpu, 1, l.temp_gpu, 1);			 
        activate_array_ongpu(l.temp_gpu, l.outputs*l.batch, TANH);			 
        mul_ongpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp_gpu, 1);		 
        gradient_array_ongpu(l.o_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu); 
        copy_ongpu(l.outputs*l.batch, l.temp_gpu, 1, wo.delta_gpu, 1);
        s.input = l.prev_state_gpu;
        s.delta = l.dh_gpu;															
        backward_connected_layer_gpu(wo, s);	

        copy_ongpu(l.outputs*l.batch, l.temp_gpu, 1, uo.delta_gpu, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_connected_layer_gpu(uo, s);									

        copy_ongpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);			 
        mul_ongpu(l.outputs*l.batch, l.i_gpu, 1, l.temp_gpu, 1);				 
        gradient_array_ongpu(l.g_gpu, l.outputs*l.batch, TANH, l.temp_gpu);		 
        copy_ongpu(l.outputs*l.batch, l.temp_gpu, 1, wg.delta_gpu, 1);
        s.input = l.prev_state_gpu;
        s.delta = l.dh_gpu;														
        backward_connected_layer_gpu(wg, s);	

        copy_ongpu(l.outputs*l.batch, l.temp_gpu, 1, ug.delta_gpu, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_connected_layer_gpu(ug, s);																

        copy_ongpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);			 
        mul_ongpu(l.outputs*l.batch, l.g_gpu, 1, l.temp_gpu, 1);				 
        gradient_array_ongpu(l.i_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu);	 
        copy_ongpu(l.outputs*l.batch, l.temp_gpu, 1, wi.delta_gpu, 1);
        s.input = l.prev_state_gpu;
        s.delta = l.dh_gpu;
        backward_connected_layer_gpu(wi, s);						

        copy_ongpu(l.outputs*l.batch, l.temp_gpu, 1, ui.delta_gpu, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_connected_layer_gpu(ui, s);									

        copy_ongpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);		 
        mul_ongpu(l.outputs*l.batch, l.prev_cell_gpu, 1, l.temp_gpu, 1); 
        gradient_array_ongpu(l.f_gpu, l.outputs*l.batch, LOGISTIC, l.temp_gpu); 
        copy_ongpu(l.outputs*l.batch, l.temp_gpu, 1, wf.delta_gpu, 1);
        s.input = l.prev_state_gpu;
        s.delta = l.dh_gpu;
        backward_connected_layer_gpu(wf, s);						

        copy_ongpu(l.outputs*l.batch, l.temp_gpu, 1, uf.delta_gpu, 1);
        s.input = state.input;
        s.delta = state.delta;
        backward_connected_layer_gpu(uf, s);									

        copy_ongpu(l.outputs*l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);			 
        mul_ongpu(l.outputs*l.batch, l.f_gpu, 1, l.temp_gpu, 1);				 
        copy_ongpu(l.outputs*l.batch, l.temp_gpu, 1, l.dc_gpu, 1);				 

        state.input -= l.inputs*l.batch;
        if (state.delta) state.delta -= l.inputs*l.batch;
        l.output_gpu -= l.outputs*l.batch;
        l.cell_gpu -= l.outputs*l.batch;
        l.delta_gpu -= l.outputs*l.batch;

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
#endif
