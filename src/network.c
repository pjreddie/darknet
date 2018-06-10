#include <stdio.h>
#include <time.h>
#include <assert.h>
#include "network.h"
#include "image.h"
#include "data.h"
#include "utils.h"
#include "blas.h"

#include "crop_layer.h"
#include "connected_layer.h"
#include "gru_layer.h"
#include "rnn_layer.h"
#include "crnn_layer.h"
#include "local_layer.h"
#include "convolutional_layer.h"
#include "activation_layer.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "normalization_layer.h"
#include "batchnorm_layer.h"
#include "maxpool_layer.h"
#include "reorg_layer.h"
#include "avgpool_layer.h"
#include "cost_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "yolo_layer.h"
#include "upsample_layer.h"
#include "parser.h"

network *load_network_custom(char *cfg, char *weights, int clear, int batch)
{
	printf(" Try to load cfg: %s, weights: %s, clear = %d \n", cfg, weights, clear);
	network *net = calloc(1, sizeof(network));
	*net = parse_network_cfg_custom(cfg, batch);
	if (weights && weights[0] != 0) {
		load_weights(net, weights);
	}
	if (clear) (*net->seen) = 0;
	return net;
}

network *load_network(char *cfg, char *weights, int clear)
{
	return load_network_custom(cfg, weights, clear, 0);
}

int get_current_batch(network net)
{
    int batch_num = (*net.seen)/(net.batch*net.subdivisions);
    return batch_num;
}

void reset_momentum(network net)
{
    if (net.momentum == 0) return;
    net.learning_rate = 0;
    net.momentum = 0;
    net.decay = 0;
    #ifdef GPU
        //if(net.gpu_index >= 0) update_network_gpu(net);
    #endif
}

void reset_network_state(network *net, int b)
{
	int i;
	for (i = 0; i < net->n; ++i) {
#ifdef GPU
		layer l = net->layers[i];
		if (l.state_gpu) {
			fill_ongpu(l.outputs, 0, l.state_gpu + l.outputs*b, 1);
		}
		if (l.h_gpu) {
			fill_ongpu(l.outputs, 0, l.h_gpu + l.outputs*b, 1);
		}
#endif
	}
}

void reset_rnn(network *net)
{
	reset_network_state(net, 0);
}

float get_current_rate(network net)
{
    int batch_num = get_current_batch(net);
    int i;
    float rate;
	if (batch_num < net.burn_in) return net.learning_rate * pow((float)batch_num / net.burn_in, net.power);
    switch (net.policy) {
        case CONSTANT:
            return net.learning_rate;
        case STEP:
            return net.learning_rate * pow(net.scale, batch_num/net.step);
        case STEPS:
            rate = net.learning_rate;
            for(i = 0; i < net.num_steps; ++i){
                if(net.steps[i] > batch_num) return rate;
                rate *= net.scales[i];
                //if(net.steps[i] > batch_num - 1 && net.scales[i] > 1) reset_momentum(net);
            }
            return rate;
        case EXP:
            return net.learning_rate * pow(net.gamma, batch_num);
        case POLY:
			return net.learning_rate * pow(1 - (float)batch_num / net.max_batches, net.power);
            //if (batch_num < net.burn_in) return net.learning_rate * pow((float)batch_num / net.burn_in, net.power);
            //return net.learning_rate * pow(1 - (float)batch_num / net.max_batches, net.power);
        case RANDOM:
            return net.learning_rate * pow(rand_uniform(0,1), net.power);
        case SIG:
            return net.learning_rate * (1./(1.+exp(net.gamma*(batch_num - net.step))));
        default:
            fprintf(stderr, "Policy is weird!\n");
            return net.learning_rate;
    }
}

char *get_layer_string(LAYER_TYPE a)
{
    switch(a){
        case CONVOLUTIONAL:
            return "convolutional";
        case ACTIVE:
            return "activation";
        case LOCAL:
            return "local";
        case DECONVOLUTIONAL:
            return "deconvolutional";
        case CONNECTED:
            return "connected";
        case RNN:
            return "rnn";
        case GRU:
            return "gru";
        case CRNN:
            return "crnn";
        case MAXPOOL:
            return "maxpool";
        case REORG:
            return "reorg";
        case AVGPOOL:
            return "avgpool";
        case SOFTMAX:
            return "softmax";
        case DETECTION:
            return "detection";
        case REGION:
            return "region";
        case DROPOUT:
            return "dropout";
        case CROP:
            return "crop";
        case COST:
            return "cost";
        case ROUTE:
            return "route";
        case SHORTCUT:
            return "shortcut";
        case NORMALIZATION:
            return "normalization";
        case BATCHNORM:
            return "batchnorm";
        default:
            break;
    }
    return "none";
}

network make_network(int n)
{
    network net = {0};
    net.n = n;
    net.layers = calloc(net.n, sizeof(layer));
    net.seen = calloc(1, sizeof(int));
#ifdef GPU
    net.input_gpu = calloc(1, sizeof(float *));
    net.truth_gpu = calloc(1, sizeof(float *));

	net.input16_gpu = calloc(1, sizeof(float *));
	net.output16_gpu = calloc(1, sizeof(float *));
	net.max_input16_size = calloc(1, sizeof(size_t));
	net.max_output16_size = calloc(1, sizeof(size_t));
#endif
    return net;
}

void forward_network(network net, network_state state)
{
    state.workspace = net.workspace;
    int i;
    for(i = 0; i < net.n; ++i){
        state.index = i;
        layer l = net.layers[i];
        if(l.delta){
            scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
        }
        l.forward(l, state);
        state.input = l.output;
    }
}

void update_network(network net)
{
    int i;
    int update_batch = net.batch*net.subdivisions;
    float rate = get_current_rate(net);
    for(i = 0; i < net.n; ++i){
        layer l = net.layers[i];
        if(l.update){
            l.update(l, update_batch, rate, net.momentum, net.decay);
        }
    }
}

float *get_network_output(network net)
{
#ifdef GPU
    if (gpu_index >= 0) return get_network_output_gpu(net);
#endif 
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].output;
}

float get_network_cost(network net)
{
    int i;
    float sum = 0;
    int count = 0;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].cost){
            sum += net.layers[i].cost[0];
            ++count;
        }
    }
    return sum/count;
}

int get_predicted_class_network(network net)
{
    float *out = get_network_output(net);
    int k = get_network_output_size(net);
    return max_index(out, k);
}

void backward_network(network net, network_state state)
{
    int i;
    float *original_input = state.input;
    float *original_delta = state.delta;
    state.workspace = net.workspace;
    for(i = net.n-1; i >= 0; --i){
        state.index = i;
        if(i == 0){
            state.input = original_input;
            state.delta = original_delta;
        }else{
            layer prev = net.layers[i-1];
            state.input = prev.output;
            state.delta = prev.delta;
        }
        layer l = net.layers[i];
        if (l.stopbackward) break;
        l.backward(l, state);
    }
}

float train_network_datum(network net, float *x, float *y)
{
#ifdef GPU
    if(gpu_index >= 0) return train_network_datum_gpu(net, x, y);
#endif
    network_state state;
    *net.seen += net.batch;
    state.index = 0;
    state.net = net;
    state.input = x;
    state.delta = 0;
    state.truth = y;
    state.train = 1;
    forward_network(net, state);
    backward_network(net, state);
    float error = get_network_cost(net);
    if(((*net.seen)/net.batch)%net.subdivisions == 0) update_network(net);
    return error;
}

float train_network_sgd(network net, data d, int n)
{
    int batch = net.batch;
    float *X = calloc(batch*d.X.cols, sizeof(float));
    float *y = calloc(batch*d.y.cols, sizeof(float));

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_random_batch(d, batch, X, y);
        float err = train_network_datum(net, X, y);
        sum += err;
    }
    free(X);
    free(y);
    return (float)sum/(n*batch);
}

float train_network(network net, data d)
{
    assert(d.X.rows % net.batch == 0);
    int batch = net.batch;
    int n = d.X.rows / batch;
    float *X = calloc(batch*d.X.cols, sizeof(float));
    float *y = calloc(batch*d.y.cols, sizeof(float));

    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        get_next_batch(d, batch, i*batch, X, y);
        float err = train_network_datum(net, X, y);
        sum += err;
    }
    free(X);
    free(y);
    return (float)sum/(n*batch);
}


float train_network_batch(network net, data d, int n)
{
    int i,j;
    network_state state;
    state.index = 0;
    state.net = net;
    state.train = 1;
    state.delta = 0;
    float sum = 0;
    int batch = 2;
    for(i = 0; i < n; ++i){
        for(j = 0; j < batch; ++j){
            int index = rand()%d.X.rows;
            state.input = d.X.vals[index];
            state.truth = d.y.vals[index];
            forward_network(net, state);
            backward_network(net, state);
            sum += get_network_cost(net);
        }
        update_network(net);
    }
    return (float)sum/(n*batch);
}

void set_batch_network(network *net, int b)
{
    net->batch = b;
    int i;
    for(i = 0; i < net->n; ++i){
        net->layers[i].batch = b;
#ifdef CUDNN
        if(net->layers[i].type == CONVOLUTIONAL){
			cudnn_convolutional_setup(net->layers + i, cudnn_fastest);
			/*
			layer *l = net->layers + i;
            cudnn_convolutional_setup(l, cudnn_fastest);
			// check for excessive memory consumption 
			size_t free_byte;
			size_t total_byte;
			check_error(cudaMemGetInfo(&free_byte, &total_byte));
			if (l->workspace_size > free_byte || l->workspace_size >= total_byte / 2) {
				printf(" used slow CUDNN algo without Workspace! \n");
				cudnn_convolutional_setup(l, cudnn_smallest);
				l->workspace_size = get_workspace_size(*l);
			}
			*/
        }
#endif
    }
}

int resize_network(network *net, int w, int h)
{
#ifdef GPU
    cuda_set_device(net->gpu_index);
    if(gpu_index >= 0){
        cuda_free(net->workspace);
		if (net->input_gpu) {
			cuda_free(*net->input_gpu);
			*net->input_gpu = 0;
			cuda_free(*net->truth_gpu);
			*net->truth_gpu = 0;
		}
    }
#endif
    int i;
    //if(w == net->w && h == net->h) return 0;
    net->w = w;
    net->h = h;
    int inputs = 0;
    size_t workspace_size = 0;
    //fprintf(stderr, "Resizing to %d x %d...\n", w, h);
    //fflush(stderr);
    for (i = 0; i < net->n; ++i){
        layer l = net->layers[i];
		//printf(" %d: layer = %d,", i, l.type);
        if(l.type == CONVOLUTIONAL){
            resize_convolutional_layer(&l, w, h);
        }else if(l.type == CROP){
            resize_crop_layer(&l, w, h);
        }else if(l.type == MAXPOOL){
            resize_maxpool_layer(&l, w, h);
        }else if(l.type == REGION){
            resize_region_layer(&l, w, h);
		}else if (l.type == YOLO) {
			resize_yolo_layer(&l, w, h);
        }else if(l.type == ROUTE){
            resize_route_layer(&l, net);
		}else if (l.type == SHORTCUT) {
			resize_shortcut_layer(&l, w, h);
		}else if (l.type == UPSAMPLE) {
			resize_upsample_layer(&l, w, h);
        }else if(l.type == REORG){
            resize_reorg_layer(&l, w, h);
        }else if(l.type == AVGPOOL){
            resize_avgpool_layer(&l, w, h);
        }else if(l.type == NORMALIZATION){
            resize_normalization_layer(&l, w, h);
        }else if(l.type == COST){
            resize_cost_layer(&l, inputs);
        }else{
			fprintf(stderr, "Resizing type %d \n", (int)l.type);
            error("Cannot resize this type of layer");
        }
        if(l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        inputs = l.outputs;
        net->layers[i] = l;
        w = l.out_w;
        h = l.out_h;
        if(l.type == AVGPOOL) break;
    }
#ifdef GPU
    if(gpu_index >= 0){
		printf(" try to allocate workspace = %zu * sizeof(float), ", workspace_size / sizeof(float) + 1);
        net->workspace = cuda_make_array(0, workspace_size/sizeof(float) + 1);
		printf(" CUDA allocate done! \n");
    }else {
        free(net->workspace);
        net->workspace = calloc(1, workspace_size);
    }
#else
    free(net->workspace);
    net->workspace = calloc(1, workspace_size);
#endif
    //fprintf(stderr, " Done!\n");
    return 0;
}

int get_network_output_size(network net)
{
    int i;
    for(i = net.n-1; i > 0; --i) if(net.layers[i].type != COST) break;
    return net.layers[i].outputs;
}

int get_network_input_size(network net)
{
    return net.layers[0].inputs;
}

detection_layer get_network_detection_layer(network net)
{
    int i;
    for(i = 0; i < net.n; ++i){
        if(net.layers[i].type == DETECTION){
            return net.layers[i];
        }
    }
    fprintf(stderr, "Detection layer not found!!\n");
    detection_layer l = {0};
    return l;
}

image get_network_image_layer(network net, int i)
{
    layer l = net.layers[i];
    if (l.out_w && l.out_h && l.out_c){
        return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
    }
    image def = {0};
    return def;
}

layer* get_network_layer(network* net, int i)
{
    return net->layers + i;
}

image get_network_image(network net)
{
    int i;
    for(i = net.n-1; i >= 0; --i){
        image m = get_network_image_layer(net, i);
        if(m.h != 0) return m;
    }
    image def = {0};
    return def;
}

void visualize_network(network net)
{
    image *prev = 0;
    int i;
    char buff[256];
    for(i = 0; i < net.n; ++i){
        sprintf(buff, "Layer %d", i);
        layer l = net.layers[i];
        if(l.type == CONVOLUTIONAL){
            prev = visualize_convolutional_layer(l, buff, prev);
        }
    } 
}

void top_predictions(network net, int k, int *index)
{
    int size = get_network_output_size(net);
    float *out = get_network_output(net);
    top_k(out, size, k, index);
}


float *network_predict(network net, float *input)
{
#ifdef GPU
    if(gpu_index >= 0)  return network_predict_gpu(net, input);
#endif

    network_state state;
    state.net = net;
    state.index = 0;
    state.input = input;
    state.truth = 0;
    state.train = 0;
    state.delta = 0;
    forward_network(net, state);
    float *out = get_network_output(net);
    return out;
}

int num_detections(network *net, float thresh)
{
	int i;
	int s = 0;
	for (i = 0; i < net->n; ++i) {
		layer l = net->layers[i];
		if (l.type == YOLO) {
			s += yolo_num_detections(l, thresh);
		}
		if (l.type == DETECTION || l.type == REGION) {
			s += l.w*l.h*l.n;
		}
	}
	return s;
}

detection *make_network_boxes(network *net, float thresh, int *num)
{
	layer l = net->layers[net->n - 1];
	int i;
	int nboxes = num_detections(net, thresh);
	if (num) *num = nboxes;
	detection *dets = calloc(nboxes, sizeof(detection));
	for (i = 0; i < nboxes; ++i) {
		dets[i].prob = calloc(l.classes, sizeof(float));
		if (l.coords > 4) {
			dets[i].mask = calloc(l.coords - 4, sizeof(float));
		}
	}
	return dets;
}


void custom_get_region_detections(layer l, int w, int h, int net_w, int net_h, float thresh, int *map, float hier, int relative, detection *dets, int letter)
{
	box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
	float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
	int i, j;
	for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float));
	get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, map);
	for (j = 0; j < l.w*l.h*l.n; ++j) {
		dets[j].classes = l.classes;
		dets[j].bbox = boxes[j];
		dets[j].objectness = 1;
		for (i = 0; i < l.classes; ++i) {
			dets[j].prob[i] = probs[j][i];
		}
	}

	free(boxes);
	free_ptrs((void **)probs, l.w*l.h*l.n);

	//correct_region_boxes(dets, l.w*l.h*l.n, w, h, net_w, net_h, relative);
	correct_yolo_boxes(dets, l.w*l.h*l.n, w, h, net_w, net_h, relative, letter);
}

void fill_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, detection *dets, int letter)
{
	int prev_classes = -1;
	int j;
	for (j = 0; j < net->n; ++j) {
		layer l = net->layers[j];
		if (l.type == YOLO) {
			int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets, letter);
			dets += count;
			if (prev_classes < 0) prev_classes = l.classes;
			else if (prev_classes != l.classes) {
				printf(" Error: Different [yolo] layers have different number of classes = %d and %d - check your cfg-file! \n",
					prev_classes, l.classes);
			}
		}
		if (l.type == REGION) {
			custom_get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets, letter);
			//get_region_detections(l, w, h, net->w, net->h, thresh, map, hier, relative, dets);
			dets += l.w*l.h*l.n;
		}
		if (l.type == DETECTION) {
			get_detection_detections(l, w, h, thresh, dets);
			dets += l.w*l.h*l.n;
		}
	}
}

detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter)
{
	detection *dets = make_network_boxes(net, thresh, num);
	fill_network_boxes(net, w, h, thresh, hier, map, relative, dets, letter);
	return dets;
}

void free_detections(detection *dets, int n)
{
	int i;
	for (i = 0; i < n; ++i) {
		free(dets[i].prob);
		if (dets[i].mask) free(dets[i].mask);
	}
	free(dets);
}

float *network_predict_image(network *net, image im)
{
	//image imr = letterbox_image(im, net->w, net->h);
	image imr = resize_image(im, net->w, net->h);
	set_batch_network(net, 1);
	float *p = network_predict(*net, imr.data);
	free_image(imr);
	return p;
}

int network_width(network *net) { return net->w; }
int network_height(network *net) { return net->h; }

matrix network_predict_data_multi(network net, data test, int n)
{
    int i,j,b,m;
    int k = get_network_output_size(net);
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net.batch*test.X.rows, sizeof(float));
    for(i = 0; i < test.X.rows; i += net.batch){
        for(b = 0; b < net.batch; ++b){
            if(i+b == test.X.rows) break;
            memcpy(X+b*test.X.cols, test.X.vals[i+b], test.X.cols*sizeof(float));
        }
        for(m = 0; m < n; ++m){
            float *out = network_predict(net, X);
            for(b = 0; b < net.batch; ++b){
                if(i+b == test.X.rows) break;
                for(j = 0; j < k; ++j){
                    pred.vals[i+b][j] += out[j+b*k]/n;
                }
            }
        }
    }
    free(X);
    return pred;   
}

matrix network_predict_data(network net, data test)
{
    int i,j,b;
    int k = get_network_output_size(net);
    matrix pred = make_matrix(test.X.rows, k);
    float *X = calloc(net.batch*test.X.cols, sizeof(float));
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
        layer l = net.layers[i];
        float *output = l.output;
        int n = l.outputs;
        float mean = mean_array(output, n);
        float vari = variance_array(output, n);
        fprintf(stderr, "Layer %d - Mean: %f, Variance: %f\n",i,mean, vari);
        if(n > 100) n = 100;
        for(j = 0; j < n; ++j) fprintf(stderr, "%f, ", output[j]);
        if(n == 100)fprintf(stderr,".....\n");
        fprintf(stderr, "\n");
    }
}

void compare_networks(network n1, network n2, data test)
{
    matrix g1 = network_predict_data(n1, test);
    matrix g2 = network_predict_data(n2, test);
    int i;
    int a,b,c,d;
    a = b = c = d = 0;
    for(i = 0; i < g1.rows; ++i){
        int truth = max_index(test.y.vals[i], test.y.cols);
        int p1 = max_index(g1.vals[i], g1.cols);
        int p2 = max_index(g2.vals[i], g2.cols);
        if(p1 == truth){
            if(p2 == truth) ++d;
            else ++c;
        }else{
            if(p2 == truth) ++b;
            else ++a;
        }
    }
    printf("%5d %5d\n%5d %5d\n", a, b, c, d);
    float num = pow((abs(b - c) - 1.), 2.);
    float den = b + c;
    printf("%f\n", num/den); 
}

float network_accuracy(network net, data d)
{
    matrix guess = network_predict_data(net, d);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

float *network_accuracies(network net, data d, int n)
{
    static float acc[2];
    matrix guess = network_predict_data(net, d);
    acc[0] = matrix_topk_accuracy(d.y, guess, 1);
    acc[1] = matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    return acc;
}

float network_accuracy_multi(network net, data d, int n)
{
    matrix guess = network_predict_data_multi(net, d, n);
    float acc = matrix_topk_accuracy(d.y, guess,1);
    free_matrix(guess);
    return acc;
}

void free_network(network net)
{
	int i;
	for (i = 0; i < net.n; ++i) {
		free_layer(net.layers[i]);
	}
	free(net.layers);

	free(net.scales);
	free(net.steps);
	free(net.seen);

#ifdef GPU
	if (gpu_index >= 0) cuda_free(net.workspace);
	else free(net.workspace);
	if (*net.input_gpu) cuda_free(*net.input_gpu);
	if (*net.truth_gpu) cuda_free(*net.truth_gpu);
	if (net.input_gpu) free(net.input_gpu);
	if (net.truth_gpu) free(net.truth_gpu);

	if (*net.input16_gpu) cuda_free(*net.input16_gpu);
	if (*net.output16_gpu) cuda_free(*net.output16_gpu);
	if (net.input16_gpu) free(net.input16_gpu);
	if (net.output16_gpu) free(net.output16_gpu);
	if (net.max_input16_size) free(net.max_input16_size);
	if (net.max_output16_size) free(net.max_output16_size);
#else
	free(net.workspace);
#endif
}


void fuse_conv_batchnorm(network net)
{
	int j;
	for (j = 0; j < net.n; ++j) {
		layer *l = &net.layers[j];

		if (l->type == CONVOLUTIONAL) {
			//printf(" Merges Convolutional-%d and batch_norm \n", j);

			if (l->batch_normalize) {
				int f;
				for (f = 0; f < l->n; ++f)
				{
					l->biases[f] = l->biases[f] - (double)l->scales[f] * l->rolling_mean[f] / (sqrt((double)l->rolling_variance[f]) + .000001f);

					const size_t filter_size = l->size*l->size*l->c;
					int i;
					for (i = 0; i < filter_size; ++i) {
						int w_index = f*filter_size + i;

						l->weights[w_index] = (double)l->weights[w_index] * l->scales[f] / (sqrt((double)l->rolling_variance[f]) + .000001f);
					}
				}

				l->batch_normalize = 0;
#ifdef GPU
				if (gpu_index >= 0) {
					push_convolutional_layer(*l);
				}
#endif
			}
		}
		else {
			//printf(" Fusion skip layer type: %d \n", l->type);
		}
	}
}
