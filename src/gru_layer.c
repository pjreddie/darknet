#include "gru_layer.h"
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

layer make_gru_layer(int batch, int inputs, int outputs, int steps, int batch_normalize)
{
	fprintf(stderr, "GRU Layer: %d inputs, %d outputs\n", inputs, outputs);
	batch = batch / steps;
	layer l = { 0 };
	l.batch = batch;
	l.type = GRU;
	l.steps = steps;
	l.inputs = inputs;

	l.wz = malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.wz) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize);
	l.wz->batch = batch;

	l.uz = malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.uz) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize);
	l.uz->batch = batch;

	l.wr = malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.wr) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize);
	l.wr->batch = batch;

	l.ur = malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.ur) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize);
	l.ur->batch = batch;

	l.wh = malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.wh) = make_connected_layer(batch*steps, inputs, outputs, LINEAR, batch_normalize);
	l.wh->batch = batch;

	l.uh = malloc(sizeof(layer));
	fprintf(stderr, "\t\t");
	*(l.uh) = make_connected_layer(batch*steps, outputs, outputs, LINEAR, batch_normalize);
	l.uh->batch = batch;

	l.batch_normalize = batch_normalize;
	l.outputs = outputs;

	l.output = calloc(outputs*batch*steps, sizeof(float));
	l.state = calloc(outputs*batch, sizeof(float));

	l.forward = forward_gru_layer;
	l.backward = backward_gru_layer;
	l.update = update_gru_layer;

#ifdef GPU
	l.forward_gpu = forward_gru_layer_gpu;
	l.backward_gpu = backward_gru_layer_gpu;
	l.update_gpu = update_gru_layer_gpu;

	l.prev_state_gpu = cuda_make_array(0, batch*outputs);
	l.output_gpu = cuda_make_array(0, batch*outputs*steps);
	l.delta_gpu = cuda_make_array(0, batch*outputs*steps);

	l.r_gpu = cuda_make_array(l.output, batch*outputs);
	l.z_gpu = cuda_make_array(l.output, batch*outputs);
	l.hh_gpu = cuda_make_array(l.output, batch*outputs);
	l.h_gpu = cuda_make_array(l.output, batch*outputs);
	l.temp_gpu = cuda_make_array(l.output, batch*outputs);
	l.temp2_gpu = cuda_make_array(l.output, batch*outputs);
	l.temp3_gpu = cuda_make_array(l.output, batch*outputs);
	l.dh_gpu = cuda_make_array(l.output, batch*outputs);
#endif
	return l;
}

void update_gru_layer(layer l, int batch, float learning_rate, float momentum, float decay)
{
}

void forward_gru_layer(layer l, network state)
{
}

void backward_gru_layer(layer l, network state)
{
}

#ifdef GPU

void pull_gru_layer(layer l)
{
}

void push_gru_layer(layer l)
{
}

void update_gru_layer_gpu(layer l, int batch, float learning_rate, float momentum, float decay)
{
	update_connected_layer_gpu(*(l.wr), batch, learning_rate, momentum, decay);
	update_connected_layer_gpu(*(l.wz), batch, learning_rate, momentum, decay);
	update_connected_layer_gpu(*(l.wh), batch, learning_rate, momentum, decay);
	update_connected_layer_gpu(*(l.ur), batch, learning_rate, momentum, decay);
	update_connected_layer_gpu(*(l.uz), batch, learning_rate, momentum, decay);
	update_connected_layer_gpu(*(l.uh), batch, learning_rate, momentum, decay);
}

void forward_gru_layer_gpu(layer l, network state)
{
	network s = { 0 };
	s.train = state.train;
	int i;
	layer wz = *(l.wz);
	layer wr = *(l.wr);
	layer wh = *(l.wh);

	layer uz = *(l.uz);
	layer ur = *(l.ur);
	layer uh = *(l.uh);

	fill_ongpu(l.outputs * l.batch * l.steps, 0, wz.delta_gpu, 1);
	fill_ongpu(l.outputs * l.batch * l.steps, 0, wr.delta_gpu, 1);
	fill_ongpu(l.outputs * l.batch * l.steps, 0, wh.delta_gpu, 1);

	fill_ongpu(l.outputs * l.batch * l.steps, 0, uz.delta_gpu, 1);
	fill_ongpu(l.outputs * l.batch * l.steps, 0, ur.delta_gpu, 1);
	fill_ongpu(l.outputs * l.batch * l.steps, 0, uh.delta_gpu, 1);

	if (state.train) {
		fill_ongpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
	}

	for (i = 0; i < l.steps; ++i) {
		s.input = l.h_gpu;
		forward_connected_layer_gpu(uz, s);									 
		forward_connected_layer_gpu(ur, s);									 

		s.input = state.input;
		forward_connected_layer_gpu(wz, s);									 
		forward_connected_layer_gpu(wr, s);									 
		forward_connected_layer_gpu(wh, s);									 

		copy_ongpu(l.outputs*l.batch, wz.output_gpu, 1, l.z_gpu, 1);		 
		axpy_ongpu(l.outputs*l.batch, 1, uz.output_gpu, 1, l.z_gpu, 1);		 

		copy_ongpu(l.outputs*l.batch, wr.output_gpu, 1, l.r_gpu, 1);		 
		axpy_ongpu(l.outputs*l.batch, 1, ur.output_gpu, 1, l.r_gpu, 1);		 

		activate_array_ongpu(l.z_gpu, l.outputs*l.batch, LOGISTIC);			 
		activate_array_ongpu(l.r_gpu, l.outputs*l.batch, LOGISTIC);			 

		copy_ongpu(l.outputs*l.batch, l.h_gpu, 1, l.hh_gpu, 1);				 
		mul_ongpu(l.outputs*l.batch, l.r_gpu, 1, l.hh_gpu, 1);				 

		s.input = l.hh_gpu;
		forward_connected_layer_gpu(uh, s);									 

		copy_ongpu(l.outputs*l.batch, wh.output_gpu, 1, l.hh_gpu, 1);		 
		axpy_ongpu(l.outputs*l.batch, 1, uh.output_gpu, 1, l.hh_gpu, 1);	 

		activate_array_ongpu(l.hh_gpu, l.outputs*l.batch, TANH);			 

		weighted_sum_gpu(l.h_gpu, l.hh_gpu, l.z_gpu, l.outputs*l.batch, l.output_gpu);
		//ht = z .* ht-1 + (1-z) .* hh
		copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.h_gpu, 1);

		state.input += l.inputs*l.batch;
		l.output_gpu += l.outputs*l.batch;

		increment_layer(&wz, 1);
		increment_layer(&wr, 1);
		increment_layer(&wh, 1);

		increment_layer(&uz, 1);
		increment_layer(&ur, 1);
		increment_layer(&uh, 1);
	}
}

void backward_gru_layer_gpu(layer l, network state)
{
	network s = { 0 };
	s.train = state.train;
	int i;
	layer wz = *(l.wz);
	layer wr = *(l.wr);
	layer wh = *(l.wh);

	layer uz = *(l.uz);
	layer ur = *(l.ur);
	layer uh = *(l.uh);

	increment_layer(&wz, l.steps - 1);
	increment_layer(&wr, l.steps - 1);
	increment_layer(&wh, l.steps - 1);

	increment_layer(&uz, l.steps - 1);
	increment_layer(&ur, l.steps - 1);
	increment_layer(&uh, l.steps - 1);

	state.input += l.inputs*l.batch*(l.steps - 1);
	if (state.delta) state.delta += l.inputs*l.batch*(l.steps - 1);

	l.output_gpu += l.outputs*l.batch*(l.steps - 1);
	l.delta_gpu += l.outputs*l.batch*(l.steps - 1);

	for (i = l.steps - 1; i >= 0; --i) {
		if (i>0) copy_ongpu(l.outputs*l.batch, l.output_gpu - l.outputs*l.batch, 1, l.prev_state_gpu, 1);
		l.dh_gpu = (i == 0) ? 0 : l.delta_gpu - l.outputs*l.batch;

		copy_ongpu(l.outputs*l.batch, wz.output_gpu, 1, l.z_gpu, 1);		 
		axpy_ongpu(l.outputs*l.batch, 1, uz.output_gpu, 1, l.z_gpu, 1);		 

		copy_ongpu(l.outputs*l.batch, wr.output_gpu, 1, l.r_gpu, 1);		 
		axpy_ongpu(l.outputs*l.batch, 1, ur.output_gpu, 1, l.r_gpu, 1);		 

		activate_array_ongpu(l.z_gpu, l.outputs*l.batch, LOGISTIC);			 
		activate_array_ongpu(l.r_gpu, l.outputs*l.batch, LOGISTIC);			 

		copy_ongpu(l.outputs*l.batch, wh.output_gpu, 1, l.hh_gpu, 1);		 
		axpy_ongpu(l.outputs*l.batch, 1, uh.output_gpu, 1, l.hh_gpu, 1);	 

		activate_array_ongpu(l.hh_gpu, l.outputs*l.batch, TANH);			 

		copy_ongpu(l.outputs*l.batch, l.delta_gpu, 1, l.temp3_gpu, 1);		 

		fill_ongpu(l.outputs*l.batch, 1, l.temp_gpu, 1);					 
		axpy_ongpu(l.outputs*l.batch, -1, l.z_gpu, 1, l.temp_gpu, 1);		 
		mul_ongpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp_gpu, 1);		 
		gradient_array_ongpu(l.hh_gpu, l.outputs*l.batch, TANH, l.temp_gpu); 

		copy_ongpu(l.outputs*l.batch, l.temp_gpu, 1, wh.delta_gpu, 1);
		s.input = state.input;
		s.delta = state.delta;
		backward_connected_layer_gpu(wh, s);

		copy_ongpu(l.outputs*l.batch, l.prev_state_gpu, 1, l.temp2_gpu, 1);	 
		mul_ongpu(l.outputs*l.batch, l.r_gpu, 1, l.temp2_gpu, 1);			 

		copy_ongpu(l.outputs*l.batch, l.temp_gpu, 1, uh.delta_gpu, 1);
		fill_ongpu(l.outputs*l.batch, 0, l.temp_gpu, 1);					 
		s.input = l.temp2_gpu;
		s.delta = l.temp_gpu;
		backward_connected_layer_gpu(uh, s);								 

		copy_ongpu(l.outputs*l.batch, l.temp_gpu, 1, l.temp2_gpu, 1);		 
		mul_ongpu(l.outputs*l.batch, l.prev_state_gpu, 1, l.temp2_gpu, 1);	 
		gradient_array_ongpu(l.r_gpu, l.outputs*l.batch, LOGISTIC, l.temp2_gpu); 

		copy_ongpu(l.outputs*l.batch, l.temp2_gpu, 1, wr.delta_gpu, 1);
		s.input = state.input;
		s.delta = state.delta;
		backward_connected_layer_gpu(wr, s);

		copy_ongpu(l.outputs*l.batch, l.temp2_gpu, 1, ur.delta_gpu, 1);
		s.input = l.prev_state_gpu;
		s.delta = l.dh_gpu;
		backward_connected_layer_gpu(ur, s);									 

		copy_ongpu(l.outputs*l.batch, l.temp_gpu, 1, l.temp2_gpu, 1);			 
		mul_ongpu(l.outputs*l.batch, l.r_gpu, 1, l.temp2_gpu, 1);				 
		if (l.dh_gpu) axpy_ongpu(l.outputs*l.batch, 1, l.temp2_gpu, 1, l.dh_gpu, 1); 

		copy_ongpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp2_gpu, 1); 
		mul_ongpu(l.outputs*l.batch, l.z_gpu, 1, l.temp2_gpu, 1);		 
		if (l.dh_gpu) axpy_ongpu(l.outputs*l.batch, 1, l.temp2_gpu, 1, l.dh_gpu, 1);	 

		copy_ongpu(l.outputs*l.batch, l.temp3_gpu, 1, l.temp2_gpu, 1);			 
		mul_ongpu(l.outputs*l.batch, l.prev_state_gpu, 1, l.temp3_gpu, 1);		 
		mul_ongpu(l.outputs*l.batch, l.hh_gpu, 1, l.temp2_gpu, 1);				 
		axpy_ongpu(l.outputs*l.batch, -1, l.temp2_gpu, 1, l.temp3_gpu, 1);		 
		gradient_array_ongpu(l.z_gpu, l.outputs*l.batch, LOGISTIC, l.temp3_gpu); 

		copy_ongpu(l.outputs*l.batch, l.temp3_gpu, 1, wz.delta_gpu, 1);
		s.input = state.input;
		s.delta = state.delta;
		backward_connected_layer_gpu(wz, s);

		copy_ongpu(l.outputs*l.batch, l.temp3_gpu, 1, uz.delta_gpu, 1);
		s.input = l.prev_state_gpu;
		s.delta = l.dh_gpu;
		backward_connected_layer_gpu(uz, s);									 

		state.input -= l.inputs*l.batch;
		if (state.delta) state.delta -= l.inputs*l.batch;
		l.output_gpu -= l.outputs*l.batch;
		l.delta_gpu -= l.outputs*l.batch;

		increment_layer(&wz, -1);
		increment_layer(&wr, -1);
		increment_layer(&wh, -1);

		increment_layer(&uz, -1);
		increment_layer(&ur, -1);
		increment_layer(&uh, -1);
	}
}
#endif