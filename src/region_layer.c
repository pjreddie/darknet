#include "region_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#define DOABS 1

region_layer make_region_layer(int batch, int w, int h, int n, int classes, int coords, int max_boxes)
{
    region_layer l = {0};
    l.type = REGION;

    l.n = n;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.classes = classes;
    l.coords = coords;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(n*2, sizeof(float));
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + coords + 1);
    l.inputs = l.outputs;
	l.max_boxes = max_boxes;
    l.truths = max_boxes*(5);
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    int i;
    for(i = 0; i < n*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_region_layer;
    l.backward = backward_region_layer;
#ifdef GPU
    l.forward_gpu = forward_region_layer_gpu;
    l.backward_gpu = backward_region_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "detection\n");
    srand(0);

    return l;
}

void resize_region_layer(layer *l, int w, int h)
{
	int old_w = l->w;
	int old_h = l->h;
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + l->coords + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
	if (old_w < w || old_h < h) {
		cuda_free(l->delta_gpu);
		cuda_free(l->output_gpu);

		l->delta_gpu = cuda_make_array(l->delta, l->batch*l->outputs);
		l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
	}
#endif
}

box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h)
{
    box b;
    b.x = (i + logistic_activate(x[index + 0])) / w;
    b.y = (j + logistic_activate(x[index + 1])) / h;
    b.w = exp(x[index + 2]) * biases[2*n];
    b.h = exp(x[index + 3]) * biases[2*n+1];
    if(DOABS){
        b.w = exp(x[index + 2]) * biases[2*n]   / w;
        b.h = exp(x[index + 3]) * biases[2*n+1] / h;
    }
    return b;
}

float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w / biases[2*n]);
    float th = log(truth.h / biases[2*n + 1]);
    if(DOABS){
        tw = log(truth.w*w / biases[2*n]);
        th = log(truth.h*h / biases[2*n + 1]);
    }

    delta[index + 0] = scale * (tx - logistic_activate(x[index + 0])) * logistic_gradient(logistic_activate(x[index + 0]));
    delta[index + 1] = scale * (ty - logistic_activate(x[index + 1])) * logistic_gradient(logistic_activate(x[index + 1]));
    delta[index + 2] = scale * (tw - x[index + 2]);
    delta[index + 3] = scale * (th - x[index + 3]);
    return iou;
}

void delta_region_class(float *output, float *delta, int index, int class_id, int classes, tree *hier, float scale, float *avg_cat, int focal_loss)
{
    int i, n;
    if(hier){
        float pred = 1;
        while(class_id >= 0){
            pred *= output[index + class_id];
            int g = hier->group[class_id];
            int offset = hier->group_offset[g];
            for(i = 0; i < hier->group_size[g]; ++i){
                delta[index + offset + i] = scale * (0 - output[index + offset + i]);
            }
            delta[index + class_id] = scale * (1 - output[index + class_id]);

            class_id = hier->parent[class_id];
        }
        *avg_cat += pred;
    } else {		
		// Focal loss
		if (focal_loss) {
			// Focal Loss
			float alpha = 0.5;	// 0.25 or 0.5
			//float gamma = 2;	// hardcoded in many places of the grad-formula	

			int ti = index + class_id;
			float pt = output[ti] + 0.000000000000001F;
			// http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
			float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);	// http://blog.csdn.net/linmingan/article/details/77885832	
			//float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);	// https://github.com/unsky/focal-loss

			for (n = 0; n < classes; ++n) {
				delta[index + n] = scale * (((n == class_id) ? 1 : 0) - output[index + n]);

				delta[index + n] *= alpha*grad;

				if (n == class_id) *avg_cat += output[index + n];
			}
		}
		else {
			// default
			for (n = 0; n < classes; ++n) {
				delta[index + n] = scale * (((n == class_id) ? 1 : 0) - output[index + n]);
				if (n == class_id) *avg_cat += output[index + n];
			}
		}
    }
}

float logit(float x)
{
    return log(x/(1.-x));
}

float tisnan(float x)
{
    return (x != x);
}

static int entry_index(layer l, int batch, int location, int entry)
{
	int n = location / (l.w*l.h);
	int loc = location % (l.w*l.h);
	return batch*l.outputs + n*l.w*l.h*(l.coords + l.classes + 1) + entry*l.w*l.h + loc;
}

void softmax_tree(float *input, int batch, int inputs, float temp, tree *hierarchy, float *output);
void forward_region_layer(const region_layer l, network_state state)
{
    int i,j,b,t,n;
    int size = l.coords + l.classes + 1;
    memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));
    #ifndef GPU
    flatten(l.output, l.w*l.h, size*l.n, l.batch, 1);
    #endif
    for (b = 0; b < l.batch; ++b){
        for(i = 0; i < l.h*l.w*l.n; ++i){
            int index = size*i + b*l.outputs;
            l.output[index + 4] = logistic_activate(l.output[index + 4]);
        }
    }


#ifndef GPU
    if (l.softmax_tree){
        for (b = 0; b < l.batch; ++b){
            for(i = 0; i < l.h*l.w*l.n; ++i){
                int index = size*i + b*l.outputs;
                softmax_tree(l.output + index + 5, 1, 0, 1, l.softmax_tree, l.output + index + 5);
            }
        }
    } else if (l.softmax){
        for (b = 0; b < l.batch; ++b){
            for(i = 0; i < l.h*l.w*l.n; ++i){
                int index = size*i + b*l.outputs;
                softmax(l.output + index + 5, l.classes, 1, l.output + index + 5, 1);
            }
        }
    }
#endif
    if(!state.train) return;
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    float avg_iou = 0;
    float recall = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        if(l.softmax_tree){
            int onlyclass_id = 0;
            for(t = 0; t < l.max_boxes; ++t){
                box truth = float_to_box(state.truth + t*5 + b*l.truths);
                if(!truth.x) break;
                int class_id = state.truth[t*5 + b*l.truths + 4];
                float maxp = 0;
                int maxi = 0;
                if(truth.x > 100000 && truth.y > 100000){
                    for(n = 0; n < l.n*l.w*l.h; ++n){
                        int index = size*n + b*l.outputs + 5;
                        float scale =  l.output[index-1];
                        float p = scale*get_hierarchy_probability(l.output + index, l.softmax_tree, class_id);
                        if(p > maxp){
                            maxp = p;
                            maxi = n;
                        }
                    }
                    int index = size*maxi + b*l.outputs + 5;
                    delta_region_class(l.output, l.delta, index, class_id, l.classes, l.softmax_tree, l.class_scale, &avg_cat, l.focal_loss);
                    ++class_count;
                    onlyclass_id = 1;
                    break;
                }
            }
            if(onlyclass_id) continue;
        }
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int index = size*(j*l.w*l.n + i*l.n + n) + b*l.outputs;
                    box pred = get_region_box(l.output, l.biases, n, index, i, j, l.w, l.h);
                    float best_iou = 0;
                    int best_class_id = -1;
                    for(t = 0; t < l.max_boxes; ++t){
                        box truth = float_to_box(state.truth + t*5 + b*l.truths);
						int class_id = state.truth[t * 5 + b*l.truths + 4];
						if (class_id >= l.classes) continue; // if label contains class_id more than number of classes in the cfg-file
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_class_id = state.truth[t*5 + b*l.truths + 4];
                            best_iou = iou;
                        }
                    }
                    avg_anyobj += l.output[index + 4];
                    l.delta[index + 4] = l.noobject_scale * ((0 - l.output[index + 4]) * logistic_gradient(l.output[index + 4]));
                    if(l.classfix == -1) l.delta[index + 4] = l.noobject_scale * ((best_iou - l.output[index + 4]) * logistic_gradient(l.output[index + 4]));
                    else{
                        if (best_iou > l.thresh) {
                            l.delta[index + 4] = 0;
                            if(l.classfix > 0){
                                delta_region_class(l.output, l.delta, index + 5, best_class_id, l.classes, l.softmax_tree, l.class_scale*(l.classfix == 2 ? l.output[index + 4] : 1), &avg_cat, l.focal_loss);
                                ++class_count;
                            }
                        }
                    }

                    if(*(state.net.seen) < 12800){
                        box truth = {0};
                        truth.x = (i + .5)/l.w;
                        truth.y = (j + .5)/l.h;
                        truth.w = l.biases[2*n];
                        truth.h = l.biases[2*n+1];
                        if(DOABS){
                            truth.w = l.biases[2*n]/l.w;
                            truth.h = l.biases[2*n+1]/l.h;
                        }
                        delta_region_box(truth, l.output, l.biases, n, index, i, j, l.w, l.h, l.delta, .01);
                    }
                }
            }
        }
        for(t = 0; t < l.max_boxes; ++t){
            box truth = float_to_box(state.truth + t*5 + b*l.truths);
			int class_id = state.truth[t * 5 + b*l.truths + 4];
			if (class_id >= l.classes) {
				printf(" Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] \n", class_id, l.classes, l.classes-1);
				getchar();
				continue; // if label contains class_id more than number of classes in the cfg-file
			}

            if(!truth.x) break;
            float best_iou = 0;
            int best_index = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            //printf("%d %f %d %f\n", i, truth.x*l.w, j, truth.y*l.h);
            box truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            //printf("index %d %d\n",i, j);
            for(n = 0; n < l.n; ++n){
                int index = size*(j*l.w*l.n + i*l.n + n) + b*l.outputs;
                box pred = get_region_box(l.output, l.biases, n, index, i, j, l.w, l.h);
                if(l.bias_match){
                    pred.w = l.biases[2*n];
                    pred.h = l.biases[2*n+1];
                    if(DOABS){
                        pred.w = l.biases[2*n]/l.w;
                        pred.h = l.biases[2*n+1]/l.h;
                    }
                }
                //printf("pred: (%f, %f) %f x %f\n", pred.x, pred.y, pred.w, pred.h);
                pred.x = 0;
                pred.y = 0;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_index = index;
                    best_iou = iou;
                    best_n = n;
                }
            }
            //printf("%d %f (%f, %f) %f x %f\n", best_n, best_iou, truth.x, truth.y, truth.w, truth.h);

            float iou = delta_region_box(truth, l.output, l.biases, best_n, best_index, i, j, l.w, l.h, l.delta, l.coord_scale);
            if(iou > .5) recall += 1;
            avg_iou += iou;

            //l.delta[best_index + 4] = iou - l.output[best_index + 4];
            avg_obj += l.output[best_index + 4];
            l.delta[best_index + 4] = l.object_scale * (1 - l.output[best_index + 4]) * logistic_gradient(l.output[best_index + 4]);
            if (l.rescore) {
                l.delta[best_index + 4] = l.object_scale * (iou - l.output[best_index + 4]) * logistic_gradient(l.output[best_index + 4]);
            }

            if (l.map) class_id = l.map[class_id];
            delta_region_class(l.output, l.delta, best_index + 5, class_id, l.classes, l.softmax_tree, l.class_scale, &avg_cat, l.focal_loss);
            ++count;
            ++class_count;
        }
    }
    //printf("\n");
    #ifndef GPU
    flatten(l.delta, l.w*l.h, size*l.n, l.batch, 0);
    #endif
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, count);
}

void backward_region_layer(const region_layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

void get_region_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map)
{
    int i,j,n;
    float *predictions = l.output;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = i*l.n + n;
            int p_index = index * (l.classes + 5) + 4;
            float scale = predictions[p_index];
            if(l.classfix == -1 && scale < .5) scale = 0;
            int box_index = index * (l.classes + 5);
            boxes[index] = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h);
            boxes[index].x *= w;
            boxes[index].y *= h;
            boxes[index].w *= w;
            boxes[index].h *= h;

            int class_index = index * (l.classes + 5) + 5;
            if(l.softmax_tree){

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0);
                int found = 0;
                if(map){
                    for(j = 0; j < 200; ++j){
                        float prob = scale*predictions[class_index+map[j]];
                        probs[index][j] = (prob > thresh) ? prob : 0;
                    }
                } else {
                    for(j = l.classes - 1; j >= 0; --j){
                        if(!found && predictions[class_index + j] > .5){
                            found = 1;
                        } else {
                            predictions[class_index + j] = 0;
                        }
                        float prob = predictions[class_index+j];
                        probs[index][j] = (scale > thresh) ? prob : 0;
                    }
                }
            } else {
                for(j = 0; j < l.classes; ++j){
                    float prob = scale*predictions[class_index+j];
                    probs[index][j] = (prob > thresh) ? prob : 0;
                }
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}

#ifdef GPU

void forward_region_layer_gpu(const region_layer l, network_state state)
{
    /*
       if(!state.train){
       copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
       return;
       }
     */
    flatten_ongpu(state.input, l.h*l.w, l.n*(l.coords + l.classes + 1), l.batch, 1, l.output_gpu);
    if(l.softmax_tree){
        int i;
        int count = 5;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_gpu(l.output_gpu+count, group_size, l.classes + 5, l.w*l.h*l.n*l.batch, 1, l.output_gpu + count);
            count += group_size;
        }
    }else if (l.softmax){
        softmax_gpu(l.output_gpu+5, l.classes, l.classes + 5, l.w*l.h*l.n*l.batch, 1, l.output_gpu + 5);
    }

    float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));
    float *truth_cpu = 0;
    if(state.truth){
        int num_truth = l.batch*l.truths;
        truth_cpu = calloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    cuda_pull_array(l.output_gpu, in_cpu, l.batch*l.inputs);
	//cudaStreamSynchronize(get_cuda_stream());
    network_state cpu_state = state;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_region_layer(l, cpu_state);
    //cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    free(cpu_state.input);
    if(!state.train) return;
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
	//cudaStreamSynchronize(get_cuda_stream());
    if(cpu_state.truth) free(cpu_state.truth);
}

void backward_region_layer_gpu(region_layer l, network_state state)
{
    flatten_ongpu(l.delta_gpu, l.h*l.w, l.n*(l.coords + l.classes + 1), l.batch, 0, state.delta);
}
#endif


void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
	int i;
	int new_w = 0;
	int new_h = 0;
	if (((float)netw / w) < ((float)neth / h)) {
		new_w = netw;
		new_h = (h * netw) / w;
	}
	else {
		new_h = neth;
		new_w = (w * neth) / h;
	}
	for (i = 0; i < n; ++i) {
		box b = dets[i].bbox;
		b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
		b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
		b.w *= (float)netw / new_w;
		b.h *= (float)neth / new_h;
		if (!relative) {
			b.x *= w;
			b.w *= w;
			b.y *= h;
			b.h *= h;
		}
		dets[i].bbox = b;
	}
}


void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets)
{
	int i, j, n, z;
	float *predictions = l.output;
	if (l.batch == 2) {
		float *flip = l.output + l.outputs;
		for (j = 0; j < l.h; ++j) {
			for (i = 0; i < l.w / 2; ++i) {
				for (n = 0; n < l.n; ++n) {
					for (z = 0; z < l.classes + l.coords + 1; ++z) {
						int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
						int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
						float swap = flip[i1];
						flip[i1] = flip[i2];
						flip[i2] = swap;
						if (z == 0) {
							flip[i1] = -flip[i1];
							flip[i2] = -flip[i2];
						}
					}
				}
			}
		}
		for (i = 0; i < l.outputs; ++i) {
			l.output[i] = (l.output[i] + flip[i]) / 2.;
		}
	}
	for (i = 0; i < l.w*l.h; ++i) {
		int row = i / l.w;
		int col = i % l.w;
		for (n = 0; n < l.n; ++n) {
			int index = n*l.w*l.h + i;
			for (j = 0; j < l.classes; ++j) {
				dets[index].prob[j] = 0;
			}
			int obj_index = entry_index(l, 0, n*l.w*l.h + i, l.coords);
			int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
			int mask_index = entry_index(l, 0, n*l.w*l.h + i, 4);
			float scale = l.background ? 1 : predictions[obj_index];
			dets[index].bbox = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h);// , l.w*l.h);
			dets[index].objectness = scale > thresh ? scale : 0;
			if (dets[index].mask) {
				for (j = 0; j < l.coords - 4; ++j) {
					dets[index].mask[j] = l.output[mask_index + j*l.w*l.h];
				}
			}

			int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + !l.background);
			if (l.softmax_tree) {

				hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0);// , l.w*l.h);
				if (map) {
					for (j = 0; j < 200; ++j) {
						int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + map[j]);
						float prob = scale*predictions[class_index];
						dets[index].prob[j] = (prob > thresh) ? prob : 0;
					}
				}
				else {
					int j = hierarchy_top_prediction(predictions + class_index, l.softmax_tree, tree_thresh, l.w*l.h);
					dets[index].prob[j] = (scale > thresh) ? scale : 0;
				}
			}
			else {
				if (dets[index].objectness) {
					for (j = 0; j < l.classes; ++j) {
						int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + j);
						float prob = scale*predictions[class_index];
						dets[index].prob[j] = (prob > thresh) ? prob : 0;
					}
				}
			}
		}
	}
	correct_region_boxes(dets, l.w*l.h*l.n, w, h, netw, neth, relative);
}
