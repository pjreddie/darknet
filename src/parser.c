#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "parser.h"
#include "activations.h"
#include "crop_layer.h"
#include "cost_layer.h"
#include "convolutional_layer.h"
#include "connected_layer.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "softmax_layer.h"
#include "dropout_layer.h"
#include "freeweight_layer.h"
#include "list.h"
#include "option_list.h"
#include "utils.h"

typedef struct{
    char *type;
    list *options;
}section;

int is_convolutional(section *s);
int is_connected(section *s);
int is_maxpool(section *s);
int is_dropout(section *s);
int is_freeweight(section *s);
int is_softmax(section *s);
int is_crop(section *s);
int is_cost(section *s);
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

convolutional_layer *parse_convolutional(list *options, network *net, int count)
{
    int h,w,c;
    float learning_rate, momentum, decay;
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int(options, "pad",0);
    char *activation_s = option_find_str(options, "activation", "sigmoid");
    ACTIVATION activation = get_activation(activation_s);
    if(count == 0){
        learning_rate = option_find_float(options, "learning_rate", .001);
        momentum = option_find_float(options, "momentum", .9);
        decay = option_find_float(options, "decay", .0001);
        h = option_find_int(options, "height",1);
        w = option_find_int(options, "width",1);
        c = option_find_int(options, "channels",1);
        net->batch = option_find_int(options, "batch",1);
        net->learning_rate = learning_rate;
        net->momentum = momentum;
        net->decay = decay;
    }else{
        learning_rate = option_find_float_quiet(options, "learning_rate", net->learning_rate);
        momentum = option_find_float_quiet(options, "momentum", net->momentum);
        decay = option_find_float_quiet(options, "decay", net->decay);
        image m =  get_network_image_layer(*net, count-1);
        h = m.h;
        w = m.w;
        c = m.c;
        if(h == 0) error("Layer before convolutional layer must output image.");
    }
    convolutional_layer *layer = make_convolutional_layer(net->batch,h,w,c,n,size,stride,pad,activation,learning_rate,momentum,decay);
    char *weights = option_find_str(options, "weights", 0);
    char *biases = option_find_str(options, "biases", 0);
    parse_data(weights, layer->filters, c*n*size*size);
    parse_data(biases, layer->biases, n);
    #ifdef GPU
    push_convolutional_layer(*layer);
    #endif
    option_unused(options);
    return layer;
}

connected_layer *parse_connected(list *options, network *net, int count)
{
    int input;
    float learning_rate, momentum, decay;
    int output = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "sigmoid");
    ACTIVATION activation = get_activation(activation_s);
    if(count == 0){
        input = option_find_int(options, "input",1);
        net->batch = option_find_int(options, "batch",1);
        learning_rate = option_find_float(options, "learning_rate", .001);
        momentum = option_find_float(options, "momentum", .9);
        decay = option_find_float(options, "decay", .0001);
        net->learning_rate = learning_rate;
        net->momentum = momentum;
        net->decay = decay;
    }else{
        learning_rate = option_find_float_quiet(options, "learning_rate", net->learning_rate);
        momentum = option_find_float_quiet(options, "momentum", net->momentum);
        decay = option_find_float_quiet(options, "decay", net->decay);
        input =  get_network_output_size_layer(*net, count-1);
    }
    connected_layer *layer = make_connected_layer(net->batch, input, output, activation,learning_rate,momentum,decay);
    char *weights = option_find_str(options, "weights", 0);
    char *biases = option_find_str(options, "biases", 0);
    parse_data(biases, layer->biases, output);
    parse_data(weights, layer->weights, input*output);
    #ifdef GPU
    push_connected_layer(*layer);
    #endif
    option_unused(options);
    return layer;
}

softmax_layer *parse_softmax(list *options, network *net, int count)
{
    int input;
    if(count == 0){
        input = option_find_int(options, "input",1);
        net->batch = option_find_int(options, "batch",1);
    }else{
        input =  get_network_output_size_layer(*net, count-1);
    }
    softmax_layer *layer = make_softmax_layer(net->batch, input);
    option_unused(options);
    return layer;
}

cost_layer *parse_cost(list *options, network *net, int count)
{
    int input;
    if(count == 0){
        input = option_find_int(options, "input",1);
        net->batch = option_find_int(options, "batch",1);
    }else{
        input =  get_network_output_size_layer(*net, count-1);
    }
    cost_layer *layer = make_cost_layer(net->batch, input);
    option_unused(options);
    return layer;
}

crop_layer *parse_crop(list *options, network *net, int count)
{
    float learning_rate, momentum, decay;
    int h,w,c;
    int crop_height = option_find_int(options, "crop_height",1);
    int crop_width = option_find_int(options, "crop_width",1);
    int flip = option_find_int(options, "flip",0);
    if(count == 0){
        h = option_find_int(options, "height",1);
        w = option_find_int(options, "width",1);
        c = option_find_int(options, "channels",1);
        net->batch = option_find_int(options, "batch",1);
        learning_rate = option_find_float(options, "learning_rate", .001);
        momentum = option_find_float(options, "momentum", .9);
        decay = option_find_float(options, "decay", .0001);
        net->learning_rate = learning_rate;
        net->momentum = momentum;
        net->decay = decay;
    }else{
        image m =  get_network_image_layer(*net, count-1);
        h = m.h;
        w = m.w;
        c = m.c;
        if(h == 0) error("Layer before crop layer must output image.");
    }
    crop_layer *layer = make_crop_layer(net->batch,h,w,c,crop_height,crop_width,flip);
    option_unused(options);
    return layer;
}

maxpool_layer *parse_maxpool(list *options, network *net, int count)
{
    int h,w,c;
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);
    if(count == 0){
        h = option_find_int(options, "height",1);
        w = option_find_int(options, "width",1);
        c = option_find_int(options, "channels",1);
        net->batch = option_find_int(options, "batch",1);
    }else{
        image m =  get_network_image_layer(*net, count-1);
        h = m.h;
        w = m.w;
        c = m.c;
        if(h == 0) error("Layer before convolutional layer must output image.");
    }
    maxpool_layer *layer = make_maxpool_layer(net->batch,h,w,c,size,stride);
    option_unused(options);
    return layer;
}

freeweight_layer *parse_freeweight(list *options, network *net, int count)
{
    int input;
    if(count == 0){
        net->batch = option_find_int(options, "batch",1);
        input = option_find_int(options, "input",1);
    }else{
        input =  get_network_output_size_layer(*net, count-1);
    }
    freeweight_layer *layer = make_freeweight_layer(net->batch,input);
    option_unused(options);
    return layer;
}

dropout_layer *parse_dropout(list *options, network *net, int count)
{
    int input;
    float probability = option_find_float(options, "probability", .5);
    if(count == 0){
        net->batch = option_find_int(options, "batch",1);
        input = option_find_int(options, "input",1);
    }else{
        input =  get_network_output_size_layer(*net, count-1);
    }
    dropout_layer *layer = make_dropout_layer(net->batch,input,probability);
    option_unused(options);
    return layer;
}

normalization_layer *parse_normalization(list *options, network *net, int count)
{
    int h,w,c;
    int size = option_find_int(options, "size",1);
    float alpha = option_find_float(options, "alpha", 0.);
    float beta = option_find_float(options, "beta", 1.);
    float kappa = option_find_float(options, "kappa", 1.);
    if(count == 0){
        h = option_find_int(options, "height",1);
        w = option_find_int(options, "width",1);
        c = option_find_int(options, "channels",1);
        net->batch = option_find_int(options, "batch",1);
    }else{
        image m =  get_network_image_layer(*net, count-1);
        h = m.h;
        w = m.w;
        c = m.c;
        if(h == 0) error("Layer before convolutional layer must output image.");
    }
    normalization_layer *layer = make_normalization_layer(net->batch,h,w,c,size, alpha, beta, kappa);
    option_unused(options);
    return layer;
}

network parse_network_cfg(char *filename)
{
    list *sections = read_cfg(filename);
    network net = make_network(sections->size, 0);

    node *n = sections->front;
    int count = 0;
    while(n){
        section *s = (section *)n->val;
        list *options = s->options;
        if(is_convolutional(s)){
            convolutional_layer *layer = parse_convolutional(options, &net, count);
            net.types[count] = CONVOLUTIONAL;
            net.layers[count] = layer;
        }else if(is_connected(s)){
            connected_layer *layer = parse_connected(options, &net, count);
            net.types[count] = CONNECTED;
            net.layers[count] = layer;
        }else if(is_crop(s)){
            crop_layer *layer = parse_crop(options, &net, count);
            net.types[count] = CROP;
            net.layers[count] = layer;
        }else if(is_cost(s)){
            cost_layer *layer = parse_cost(options, &net, count);
            net.types[count] = COST;
            net.layers[count] = layer;
        }else if(is_softmax(s)){
            softmax_layer *layer = parse_softmax(options, &net, count);
            net.types[count] = SOFTMAX;
            net.layers[count] = layer;
        }else if(is_maxpool(s)){
            maxpool_layer *layer = parse_maxpool(options, &net, count);
            net.types[count] = MAXPOOL;
            net.layers[count] = layer;
        }else if(is_normalization(s)){
            normalization_layer *layer = parse_normalization(options, &net, count);
            net.types[count] = NORMALIZATION;
            net.layers[count] = layer;
        }else if(is_dropout(s)){
            dropout_layer *layer = parse_dropout(options, &net, count);
            net.types[count] = DROPOUT;
            net.layers[count] = layer;
        }else if(is_freeweight(s)){
            freeweight_layer *layer = parse_freeweight(options, &net, count);
            net.types[count] = FREEWEIGHT;
            net.layers[count] = layer;
        }else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        free_section(s);
        ++count;
        n = n->next;
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
int is_convolutional(section *s)
{
    return (strcmp(s->type, "[conv]")==0
            || strcmp(s->type, "[convolutional]")==0);
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
int is_freeweight(section *s)
{
    return (strcmp(s->type, "[freeweight]")==0);
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
    int i;
    int len = strlen(s);
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
                    printf("Config file error line %d, could parse: %s\n", nu, line);
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
    int i;
    fprintf(fp, "[convolutional]\n");
    if(count == 0) {
        fprintf(fp,   "batch=%d\n"
                "height=%d\n"
                "width=%d\n"
                "channels=%d\n"
                "learning_rate=%g\n"
                "momentum=%g\n"
                "decay=%g\n",
                l->batch,l->h, l->w, l->c, l->learning_rate, l->momentum, l->decay);
    } else {
        if(l->learning_rate != net.learning_rate)
            fprintf(fp, "learning_rate=%g\n", l->learning_rate);
        if(l->momentum != net.momentum)
            fprintf(fp, "momentum=%g\n", l->momentum);
        if(l->decay != net.decay)
            fprintf(fp, "decay=%g\n", l->decay);
    }
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

void print_freeweight_cfg(FILE *fp, freeweight_layer *l, network net, int count)
{
    fprintf(fp, "[freeweight]\n");
    if(count == 0){
        fprintf(fp, "batch=%d\ninput=%d\n",l->batch, l->inputs);
    }
    fprintf(fp, "\n");
}

void print_dropout_cfg(FILE *fp, dropout_layer *l, network net, int count)
{
    fprintf(fp, "[dropout]\n");
    if(count == 0){
        fprintf(fp, "batch=%d\ninput=%d\n", l->batch, l->inputs);
    }
    fprintf(fp, "probability=%g\n\n", l->probability);
}

void print_connected_cfg(FILE *fp, connected_layer *l, network net, int count)
{
    int i;
    fprintf(fp, "[connected]\n");
    if(count == 0){
        fprintf(fp, "batch=%d\n"
                "input=%d\n"
                "learning_rate=%g\n"
                "momentum=%g\n"
                "decay=%g\n",
                l->batch, l->inputs, l->learning_rate, l->momentum, l->decay);
    } else {
        if(l->learning_rate != net.learning_rate)
            fprintf(fp, "learning_rate=%g\n", l->learning_rate);
        if(l->momentum != net.momentum)
            fprintf(fp, "momentum=%g\n", l->momentum);
        if(l->decay != net.decay)
            fprintf(fp, "decay=%g\n", l->decay);
    }
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
    if(count == 0) {
        fprintf(fp,   "batch=%d\n"
                "height=%d\n"
                "width=%d\n"
                "channels=%d\n"
                "learning_rate=%g\n"
                "momentum=%g\n"
                "decay=%g\n",
                l->batch,l->h, l->w, l->c, net.learning_rate, net.momentum, net.decay);
    }
    fprintf(fp, "crop_height=%d\ncrop_width=%d\nflip=%d\n\n", l->crop_height, l->crop_width, l->flip);
}

void print_maxpool_cfg(FILE *fp, maxpool_layer *l, network net, int count)
{
    fprintf(fp, "[maxpool]\n");
    if(count == 0) fprintf(fp,   "batch=%d\n"
            "height=%d\n"
            "width=%d\n"
            "channels=%d\n",
            l->batch,l->h, l->w, l->c);
    fprintf(fp, "size=%d\nstride=%d\n\n", l->size, l->stride);
}

void print_normalization_cfg(FILE *fp, normalization_layer *l, network net, int count)
{
    fprintf(fp, "[localresponsenormalization]\n");
    if(count == 0) fprintf(fp,   "batch=%d\n"
            "height=%d\n"
            "width=%d\n"
            "channels=%d\n",
            l->batch,l->h, l->w, l->c);
    fprintf(fp, "size=%d\n"
            "alpha=%g\n"
            "beta=%g\n"
            "kappa=%g\n\n", l->size, l->alpha, l->beta, l->kappa);
}

void print_softmax_cfg(FILE *fp, softmax_layer *l, network net, int count)
{
    fprintf(fp, "[softmax]\n");
    if(count == 0) fprintf(fp, "batch=%d\ninput=%d\n", l->batch, l->inputs);
    fprintf(fp, "\n");
}

void print_cost_cfg(FILE *fp, cost_layer *l, network net, int count)
{
    fprintf(fp, "[cost]\n");
    if(count == 0) fprintf(fp, "batch=%d\ninput=%d\n", l->batch, l->inputs);
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
            print_convolutional_cfg(fp, (convolutional_layer *)net.layers[i], net, i);
        else if(net.types[i] == CONNECTED)
            print_connected_cfg(fp, (connected_layer *)net.layers[i], net, i);
        else if(net.types[i] == CROP)
            print_crop_cfg(fp, (crop_layer *)net.layers[i], net, i);
        else if(net.types[i] == MAXPOOL)
            print_maxpool_cfg(fp, (maxpool_layer *)net.layers[i], net, i);
        else if(net.types[i] == FREEWEIGHT)
            print_freeweight_cfg(fp, (freeweight_layer *)net.layers[i], net, i);
        else if(net.types[i] == DROPOUT)
            print_dropout_cfg(fp, (dropout_layer *)net.layers[i], net, i);
        else if(net.types[i] == NORMALIZATION)
            print_normalization_cfg(fp, (normalization_layer *)net.layers[i], net, i);
        else if(net.types[i] == SOFTMAX)
            print_softmax_cfg(fp, (softmax_layer *)net.layers[i], net, i);
        else if(net.types[i] == COST)
            print_cost_cfg(fp, (cost_layer *)net.layers[i], net, i);
    }
    fclose(fp);
}

