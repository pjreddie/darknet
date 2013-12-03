#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "parser.h"
#include "activations.h"
#include "convolutional_layer.h"
#include "connected_layer.h"
#include "maxpool_layer.h"
#include "softmax_layer.h"
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
int is_softmax(section *s);
list *read_cfg(char *filename);


network parse_network_cfg(char *filename)
{
    list *sections = read_cfg(filename);
    network net = make_network(sections->size);

    node *n = sections->front;
    int count = 0;
    while(n){
        section *s = (section *)n->val;
        list *options = s->options;
        if(is_convolutional(s)){
            int h,w,c;
            int n = option_find_int(options, "filters",1);
            int size = option_find_int(options, "size",1);
            int stride = option_find_int(options, "stride",1);
            char *activation_s = option_find_str(options, "activation", "sigmoid");
            ACTIVATION activation = get_activation(activation_s);
            if(count == 0){
                h = option_find_int(options, "height",1);
                w = option_find_int(options, "width",1);
                c = option_find_int(options, "channels",1);
            }else{
                image m =  get_network_image_layer(net, count-1);
                h = m.h;
                w = m.w;
                c = m.c;
                if(h == 0) error("Layer before convolutional layer must output image.");
            }
            convolutional_layer *layer = make_convolutional_layer(h,w,c,n,size,stride, activation);
            net.types[count] = CONVOLUTIONAL;
            net.layers[count] = layer;
            option_unused(options);
        }
        else if(is_connected(s)){
            int input;
            int output = option_find_int(options, "output",1);
            char *activation_s = option_find_str(options, "activation", "sigmoid");
            ACTIVATION activation = get_activation(activation_s);
            if(count == 0){
                input = option_find_int(options, "input",1);
            }else{
                input =  get_network_output_size_layer(net, count-1);
            }
            connected_layer *layer = make_connected_layer(input, output, activation);
            net.types[count] = CONNECTED;
            net.layers[count] = layer;
            option_unused(options);
        }else if(is_softmax(s)){
            int input;
            if(count == 0){
                input = option_find_int(options, "input",1);
            }else{
                input =  get_network_output_size_layer(net, count-1);
            }
            softmax_layer *layer = make_softmax_layer(input);
            net.types[count] = SOFTMAX;
            net.layers[count] = layer;
            option_unused(options);
        }else if(is_maxpool(s)){
            int h,w,c;
            int stride = option_find_int(options, "stride",1);
            //char *activation_s = option_find_str(options, "activation", "sigmoid");
            if(count == 0){
                h = option_find_int(options, "height",1);
                w = option_find_int(options, "width",1);
                c = option_find_int(options, "channels",1);
            }else{
                image m =  get_network_image_layer(net, count-1);
                h = m.h;
                w = m.w;
                c = m.c;
                if(h == 0) error("Layer before convolutional layer must output image.");
            }
            maxpool_layer *layer = make_maxpool_layer(h,w,c,stride);
            net.types[count] = MAXPOOL;
            net.layers[count] = layer;
            option_unused(options);
        }else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        ++count;
        n = n->next;
    }   
    return net;
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

int is_softmax(section *s)
{
    return (strcmp(s->type, "[soft]")==0
            || strcmp(s->type, "[softmax]")==0);
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

