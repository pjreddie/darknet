#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "parser.h"
#include "activations.h"
#include "crop_layer.h"
#include "cost_layer.h"
#include "convolutional_layer.h"
#include "deconvolutional_layer.h"
#include "connected_layer.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "detection_layer.h"
#include "list.h"
#include "option_list.h"
#include "utils.h"

typedef struct{
    char *type;
    list *options;
}section;

int is_network(section *s);
int is_convolutional(section *s);
int is_deconvolutional(section *s);
int is_connected(section *s);
int is_maxpool(section *s);
int is_dropout(section *s);
int is_softmax(section *s);
int is_crop(section *s);
int is_cost(section *s);
int is_detection(section *s);
int is_normalization(section *s);
list *read_cfg(char *filename);

void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

void parse_data(char *data, float *a, int n)
{
    int i;
    if(!data) return;
    char *curr = data;
    char *next = data;
    int done = 0;
    for(i = 0; i < n && !done; ++i){
        while(*++next !='\0' && *next != ',');
        if(*next == '\0') done = 1;
        *next = '\0';
        sscanf(curr, "%g", &a[i]);
        curr = next+1;
    }
}

typedef struct size_params{
    int batch;
    int inputs;
    int h;
    int w;
    int c;
} size_params;

deconvolutional_layer *parse_deconvolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before deconvolutional layer must output image.");

    deconvolutional_layer *layer = make_deconvolutional_layer(batch,h,w,c,n,size,stride,activation);

    char *weights = option_find_str(options, "weights", 0);
    char *biases = option_find_str(options, "biases", 0);
    parse_data(weights, layer->filters, c*n*size*size);
    parse_data(biases, layer->biases, n);
    #ifdef GPU
    if(weights || biases) push_deconvolutional_layer(*layer);
    #endif
    option_unused(options);
    return layer;
}

convolutional_layer *parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int(options, "pad",0);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.");

    convolutional_layer *layer = make_convolutional_layer(batch,h,w,c,n,size,stride,pad,activation);

    char *weights = option_find_str(options, "weights", 0);
    char *biases = option_find_str(options, "biases", 0);
    parse_data(weights, layer->filters, c*n*size*size);
    parse_data(biases, layer->biases, n);
    #ifdef GPU
    if(weights || biases) push_convolutional_layer(*layer);
    #endif
    option_unused(options);
    return layer;
}

connected_layer *parse_connected(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    connected_layer *layer = make_connected_layer(params.batch, params.inputs, output, activation);

    char *weights = option_find_str(options, "weights", 0);
    char *biases = option_find_str(options, "biases", 0);
    parse_data(biases, layer->biases, output);
    parse_data(weights, layer->weights, params.inputs*output);
    #ifdef GPU
    if(weights || biases) push_connected_layer(*layer);
    #endif
    option_unused(options);
    return layer;
}

softmax_layer *parse_softmax(list *options, size_params params)
{
    int groups = option_find_int(options, "groups",1);
    softmax_layer *layer = make_softmax_layer(params.batch, params.inputs, groups);
    option_unused(options);
    return layer;
}

detection_layer *parse_detection(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 1);
    int classes = option_find_int(options, "classes", 1);
    int rescore = option_find_int(options, "rescore", 1);
    int nuisance = option_find_int(options, "nuisance", 0);
    int background = option_find_int(options, "background", 1);
    detection_layer *layer = make_detection_layer(params.batch, params.inputs, classes, coords, rescore, background, nuisance);
    option_unused(options);
    return layer;
}

cost_layer *parse_cost(list *options, size_params params)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    cost_layer *layer = make_cost_layer(params.batch, params.inputs, type);
    option_unused(options);
    return layer;
}

crop_layer *parse_crop(list *options, size_params params)
{
    int crop_height = option_find_int(options, "crop_height",1);
    int crop_width = option_find_int(options, "crop_width",1);
    int flip = option_find_int(options, "flip",0);
    float angle = option_find_float(options, "angle",0);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before crop layer must output image.");

    crop_layer *layer = make_crop_layer(batch,h,w,c,crop_height,crop_width,flip, angle);
    option_unused(options);
    return layer;
}

maxpool_layer *parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before maxpool layer must output image.");

    maxpool_layer *layer = make_maxpool_layer(batch,h,w,c,size,stride);
    option_unused(options);
    return layer;
}

dropout_layer *parse_dropout(list *options, size_params params)
{
    float probability = option_find_float(options, "probability", .5);
    dropout_layer *layer = make_dropout_layer(params.batch, params.inputs, probability);
    option_unused(options);
    return layer;
}

normalization_layer *parse_normalization(list *options, size_params params)
{
    int size = option_find_int(options, "size",1);
    float alpha = option_find_float(options, "alpha", 0.);
    float beta = option_find_float(options, "beta", 1.);
    float kappa = option_find_float(options, "kappa", 1.);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before normalization layer must output image.");

    normalization_layer *layer = make_normalization_layer(batch,h,w,c,size, alpha, beta, kappa);
    option_unused(options);
    return layer;
}

void parse_net_options(list *options, network *net)
{
    net->batch = option_find_int(options, "batch",1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    net->seen = option_find_int(options, "seen",0);
    int subdivs = option_find_int(options, "subdivisions",1);
    net->batch /= subdivs;
    net->subdivisions = subdivs;

    net->h = option_find_int_quiet(options, "height",0);
    net->w = option_find_int_quiet(options, "width",0);
    net->c = option_find_int_quiet(options, "channels",0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    if(!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");
    option_unused(options);
}

network parse_network_cfg(char *filename)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n) error("Config file has no sections");
    network net = make_network(sections->size - 1);
    size_params params;

    section *s = (section *)n->val;
    list *options = s->options;
    if(!is_network(s)) error("First section must be [net] or [network]");
    parse_net_options(options, &net);

    params.h = net.h;
    params.w = net.w;
    params.c = net.c;
    params.inputs = net.inputs;
    params.batch = net.batch;

    n = n->next;
    int count = 0;
    while(n){
        fprintf(stderr, "%d: ", count);
        s = (section *)n->val;
        options = s->options;
        if(is_convolutional(s)){
            convolutional_layer *layer = parse_convolutional(options, params);
            net.types[count] = CONVOLUTIONAL;
            net.layers[count] = layer;
        }else if(is_deconvolutional(s)){
            deconvolutional_layer *layer = parse_deconvolutional(options, params);
            net.types[count] = DECONVOLUTIONAL;
            net.layers[count] = layer;
        }else if(is_connected(s)){
            connected_layer *layer = parse_connected(options, params);
            net.types[count] = CONNECTED;
            net.layers[count] = layer;
        }else if(is_crop(s)){
            crop_layer *layer = parse_crop(options, params);
            net.types[count] = CROP;
            net.layers[count] = layer;
        }else if(is_cost(s)){
            cost_layer *layer = parse_cost(options, params);
            net.types[count] = COST;
            net.layers[count] = layer;
        }else if(is_detection(s)){
            detection_layer *layer = parse_detection(options, params);
            net.types[count] = DETECTION;
            net.layers[count] = layer;
        }else if(is_softmax(s)){
            softmax_layer *layer = parse_softmax(options, params);
            net.types[count] = SOFTMAX;
            net.layers[count] = layer;
        }else if(is_maxpool(s)){
            maxpool_layer *layer = parse_maxpool(options, params);
            net.types[count] = MAXPOOL;
            net.layers[count] = layer;
        }else if(is_normalization(s)){
            normalization_layer *layer = parse_normalization(options, params);
            net.types[count] = NORMALIZATION;
            net.layers[count] = layer;
        }else if(is_dropout(s)){
            dropout_layer *layer = parse_dropout(options, params);
            net.types[count] = DROPOUT;
            net.layers[count] = layer;
        }else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        free_section(s);
        n = n->next;
        if(n){
            image im = get_network_image_layer(net, count);
            params.h = im.h;
            params.w = im.w;
            params.c = im.c;
            params.inputs = get_network_output_size_layer(net, count);
        }
        ++count;
    }   
    free_list(sections);
    net.outputs = get_network_output_size(net);
    net.output = get_network_output(net);
    return net;
}

int is_crop(section *s)
{
    return (strcmp(s->type, "[crop]")==0);
}
int is_cost(section *s)
{
    return (strcmp(s->type, "[cost]")==0);
}
int is_detection(section *s)
{
    return (strcmp(s->type, "[detection]")==0);
}
int is_deconvolutional(section *s)
{
    return (strcmp(s->type, "[deconv]")==0
            || strcmp(s->type, "[deconvolutional]")==0);
}
int is_convolutional(section *s)
{
    return (strcmp(s->type, "[conv]")==0
            || strcmp(s->type, "[convolutional]")==0);
}
int is_network(section *s)
{
    return (strcmp(s->type, "[net]")==0
            || strcmp(s->type, "[network]")==0);
}
int is_connected(section *s)
{
    return (strcmp(s->type, "[conn]")==0
            || strcmp(s->type, "[connected]")==0);
}
int is_maxpool(section *s)
{
    return (strcmp(s->type, "[max]")==0
            || strcmp(s->type, "[maxpool]")==0);
}
int is_dropout(section *s)
{
    return (strcmp(s->type, "[dropout]")==0);
}

int is_softmax(section *s)
{
    return (strcmp(s->type, "[soft]")==0
            || strcmp(s->type, "[softmax]")==0);
}
int is_normalization(section *s)
{
    return (strcmp(s->type, "[lrnorm]")==0
            || strcmp(s->type, "[localresponsenormalization]")==0);
}

int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for(i = 0; i < len; ++i){
        if(s[i] == '='){
            s[i] = '\0';
            val = s+i+1;
            break;
        }
    }
    if(i == len-1) return 0;
    char *key = s;
    option_insert(options, key, val);
    return 1;
}

list *read_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *sections = make_list();
    section *current = 0;
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '[':
                current = malloc(sizeof(section));
                list_insert(sections, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, current->options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return sections;
}

void print_convolutional_cfg(FILE *fp, convolutional_layer *l, network net, int count)
{
#ifdef GPU
    if(gpu_index >= 0)  pull_convolutional_layer(*l);
#endif
    int i;
    fprintf(fp, "[convolutional]\n");
    fprintf(fp, "filters=%d\n"
            "size=%d\n"
            "stride=%d\n"
            "pad=%d\n"
            "activation=%s\n",
            l->n, l->size, l->stride, l->pad,
            get_activation_string(l->activation));
    fprintf(fp, "biases=");
    for(i = 0; i < l->n; ++i) fprintf(fp, "%g,", l->biases[i]);
    fprintf(fp, "\n");
    fprintf(fp, "weights=");
    for(i = 0; i < l->n*l->c*l->size*l->size; ++i) fprintf(fp, "%g,", l->filters[i]);
    fprintf(fp, "\n\n");
}

void print_deconvolutional_cfg(FILE *fp, deconvolutional_layer *l, network net, int count)
{
#ifdef GPU
    if(gpu_index >= 0)  pull_deconvolutional_layer(*l);
#endif
    int i;
    fprintf(fp, "[deconvolutional]\n");
    fprintf(fp, "filters=%d\n"
            "size=%d\n"
            "stride=%d\n"
            "activation=%s\n",
            l->n, l->size, l->stride,
            get_activation_string(l->activation));
    fprintf(fp, "biases=");
    for(i = 0; i < l->n; ++i) fprintf(fp, "%g,", l->biases[i]);
    fprintf(fp, "\n");
    fprintf(fp, "weights=");
    for(i = 0; i < l->n*l->c*l->size*l->size; ++i) fprintf(fp, "%g,", l->filters[i]);
    fprintf(fp, "\n\n");
}

void print_dropout_cfg(FILE *fp, dropout_layer *l, network net, int count)
{
    fprintf(fp, "[dropout]\n");
    fprintf(fp, "probability=%g\n\n", l->probability);
}

void print_connected_cfg(FILE *fp, connected_layer *l, network net, int count)
{
#ifdef GPU
    if(gpu_index >= 0) pull_connected_layer(*l);
#endif
    int i;
    fprintf(fp, "[connected]\n");
    fprintf(fp, "output=%d\n"
            "activation=%s\n",
            l->outputs,
            get_activation_string(l->activation));
    fprintf(fp, "biases=");
    for(i = 0; i < l->outputs; ++i) fprintf(fp, "%g,", l->biases[i]);
    fprintf(fp, "\n");
    fprintf(fp, "weights=");
    for(i = 0; i < l->outputs*l->inputs; ++i) fprintf(fp, "%g,", l->weights[i]);
    fprintf(fp, "\n\n");
}

void print_crop_cfg(FILE *fp, crop_layer *l, network net, int count)
{
    fprintf(fp, "[crop]\n");
    fprintf(fp, "crop_height=%d\ncrop_width=%d\nflip=%d\n\n", l->crop_height, l->crop_width, l->flip);
}

void print_maxpool_cfg(FILE *fp, maxpool_layer *l, network net, int count)
{
    fprintf(fp, "[maxpool]\n");
    fprintf(fp, "size=%d\nstride=%d\n\n", l->size, l->stride);
}

void print_normalization_cfg(FILE *fp, normalization_layer *l, network net, int count)
{
    fprintf(fp, "[localresponsenormalization]\n");
    fprintf(fp, "size=%d\n"
            "alpha=%g\n"
            "beta=%g\n"
            "kappa=%g\n\n", l->size, l->alpha, l->beta, l->kappa);
}

void print_softmax_cfg(FILE *fp, softmax_layer *l, network net, int count)
{
    fprintf(fp, "[softmax]\n");
    fprintf(fp, "\n");
}

void print_detection_cfg(FILE *fp, detection_layer *l, network net, int count)
{
    fprintf(fp, "[detection]\n");
    fprintf(fp, "classes=%d\ncoords=%d\nrescore=%d\nnuisance=%d\n", l->classes, l->coords, l->rescore, l->nuisance);
    fprintf(fp, "\n");
}

void print_cost_cfg(FILE *fp, cost_layer *l, network net, int count)
{
    fprintf(fp, "[cost]\ntype=%s\n", get_cost_string(l->type));
    fprintf(fp, "\n");
}

void save_weights(network net, char *filename)
{
    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "w");
    if(!fp) file_error(filename);

    fwrite(&net.learning_rate, sizeof(float), 1, fp);
    fwrite(&net.momentum, sizeof(float), 1, fp);
    fwrite(&net.decay, sizeof(float), 1, fp);
    fwrite(&net.seen, sizeof(int), 1, fp);

    int i;
    for(i = 0; i < net.n; ++i){
        if(net.types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *) net.layers[i];
#ifdef GPU
            if(gpu_index >= 0){
                pull_convolutional_layer(layer);
            }
#endif
            int num = layer.n*layer.c*layer.size*layer.size;
            fwrite(layer.biases, sizeof(float), layer.n, fp);
            fwrite(layer.filters, sizeof(float), num, fp);
        }
        if(net.types[i] == DECONVOLUTIONAL){
            deconvolutional_layer layer = *(deconvolutional_layer *) net.layers[i];
#ifdef GPU
            if(gpu_index >= 0){
                pull_deconvolutional_layer(layer);
            }
#endif
            int num = layer.n*layer.c*layer.size*layer.size;
            fwrite(layer.biases, sizeof(float), layer.n, fp);
            fwrite(layer.filters, sizeof(float), num, fp);
        }
        if(net.types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *) net.layers[i];
#ifdef GPU
            if(gpu_index >= 0){
                pull_connected_layer(layer);
            }
#endif
            fwrite(layer.biases, sizeof(float), layer.outputs, fp);
            fwrite(layer.weights, sizeof(float), layer.outputs*layer.inputs, fp);
        }
    }
    fclose(fp);
}

void load_weights_upto(network *net, char *filename, int cutoff)
{
    fprintf(stderr, "Loading weights from %s\n", filename);
    FILE *fp = fopen(filename, "r");
    if(!fp) file_error(filename);

    fread(&net->learning_rate, sizeof(float), 1, fp);
    fread(&net->momentum, sizeof(float), 1, fp);
    fread(&net->decay, sizeof(float), 1, fp);
    fread(&net->seen, sizeof(int), 1, fp);

    int i;
    for(i = 0; i < net->n && i < cutoff; ++i){
        if(net->types[i] == CONVOLUTIONAL){
            convolutional_layer layer = *(convolutional_layer *) net->layers[i];
            int num = layer.n*layer.c*layer.size*layer.size;
            fread(layer.biases, sizeof(float), layer.n, fp);
            fread(layer.filters, sizeof(float), num, fp);
#ifdef GPU
            if(gpu_index >= 0){
                push_convolutional_layer(layer);
            }
#endif
        }
        if(net->types[i] == DECONVOLUTIONAL){
            deconvolutional_layer layer = *(deconvolutional_layer *) net->layers[i];
            int num = layer.n*layer.c*layer.size*layer.size;
            fread(layer.biases, sizeof(float), layer.n, fp);
            fread(layer.filters, sizeof(float), num, fp);
#ifdef GPU
            if(gpu_index >= 0){
                push_deconvolutional_layer(layer);
            }
#endif
        }
        if(net->types[i] == CONNECTED){
            connected_layer layer = *(connected_layer *) net->layers[i];
            fread(layer.biases, sizeof(float), layer.outputs, fp);
            fread(layer.weights, sizeof(float), layer.outputs*layer.inputs, fp);
#ifdef GPU
            if(gpu_index >= 0){
                push_connected_layer(layer);
            }
#endif
        }
    }
    fclose(fp);
}

void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, net->n);
}

void save_network(network net, char *filename)
{
    FILE *fp = fopen(filename, "w");
    if(!fp) file_error(filename);
    int i;
    for(i = 0; i < net.n; ++i)
    {
        if(net.types[i] == CONVOLUTIONAL)
            print_convolutional_cfg(fp, (convolutional_layer *)net.layers[i], net, i);
        else if(net.types[i] == DECONVOLUTIONAL)
            print_deconvolutional_cfg(fp, (deconvolutional_layer *)net.layers[i], net, i);
        else if(net.types[i] == CONNECTED)
            print_connected_cfg(fp, (connected_layer *)net.layers[i], net, i);
        else if(net.types[i] == CROP)
            print_crop_cfg(fp, (crop_layer *)net.layers[i], net, i);
        else if(net.types[i] == MAXPOOL)
            print_maxpool_cfg(fp, (maxpool_layer *)net.layers[i], net, i);
        else if(net.types[i] == DROPOUT)
            print_dropout_cfg(fp, (dropout_layer *)net.layers[i], net, i);
        else if(net.types[i] == NORMALIZATION)
            print_normalization_cfg(fp, (normalization_layer *)net.layers[i], net, i);
        else if(net.types[i] == SOFTMAX)
            print_softmax_cfg(fp, (softmax_layer *)net.layers[i], net, i);
        else if(net.types[i] == DETECTION)
            print_detection_cfg(fp, (detection_layer *)net.layers[i], net, i);
        else if(net.types[i] == COST)
            print_cost_cfg(fp, (cost_layer *)net.layers[i], net, i);
    }
    fclose(fp);
}

