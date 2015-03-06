#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "parser.h"
#include "utils.h"
#include "cuda.h"

#define _GNU_SOURCE
#include <fenv.h>

extern void run_imagenet(int argc, char **argv);
extern void run_detection(int argc, char **argv);
extern void run_captcha(int argc, char **argv);

void convert(char *cfgfile, char *outfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    save_network(net, outfile);
}

void del_arg(int argc, char **argv, int index)
{
    int i;
    for(i = index; i < argc-1; ++i) argv[i] = argv[i+1];
    argv[i] = 0;
}

int find_arg(int argc, char* argv[], char *arg)
{
    int i;
    for(i = 0; i < argc; ++i) {
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)) {
            del_arg(argc, argv, i);
            return 1;
        }
    }
    return 0;
}

int find_int_arg(int argc, char **argv, char *arg, int def)
{
    int i;
    for(i = 0; i < argc-1; ++i){
        if(!argv[i]) continue;
        if(0==strcmp(argv[i], arg)){
            def = atoi(argv[i+1]);
            del_arg(argc, argv, i);
            del_arg(argc, argv, i);
            break;
        }
    }
    return def;
}

void scale_rate(char *filename, float scale)
{
    // Ready for some weird shit??
    FILE *fp = fopen(filename, "r+b");
    if(!fp) file_error(filename);
    float rate = 0;
    fread(&rate, sizeof(float), 1, fp);
    printf("Scaling learning rate from %f to %f\n", rate, rate*scale);
    rate = rate*scale;
    fseek(fp, 0, SEEK_SET);
    fwrite(&rate, sizeof(float), 1, fp);
    fclose(fp);
}

int main(int argc, char **argv)
{
    //test_convolutional_layer();
    if(argc < 2){
        fprintf(stderr, "usage: %s <function>\n", argv[0]);
        return 0;
    }
    gpu_index = find_int_arg(argc, argv, "-i", 0);
    if(find_arg(argc, argv, "-nogpu")) gpu_index = -1;

#ifndef GPU
    gpu_index = -1;
#else
    if(gpu_index >= 0){
        cudaSetDevice(gpu_index);
    }
#endif

    if(0==strcmp(argv[1], "imagenet")){
        run_imagenet(argc, argv);   
    } else if (0 == strcmp(argv[1], "detection")){
        run_detection(argc, argv);   
    } else if (0 == strcmp(argv[1], "captcha")){
        run_captcha(argc, argv);   
    }
    return 0;
}

