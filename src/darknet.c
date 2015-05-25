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
extern void run_writing(int argc, char **argv);
extern void run_captcha(int argc, char **argv);

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

void change_rate(char *filename, float scale, float add)
{
    // Ready for some weird shit??
    FILE *fp = fopen(filename, "r+b");
    if(!fp) file_error(filename);
    float rate = 0;
    fread(&rate, sizeof(float), 1, fp);
    printf("Scaling learning rate from %f to %f\n", rate, rate*scale+add);
    rate = rate*scale + add;
    fseek(fp, 0, SEEK_SET);
    fwrite(&rate, sizeof(float), 1, fp);
    fclose(fp);
}

void partial(char *cfgfile, char *weightfile, char *outfile, int max)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights_upto(&net, weightfile, max);
    }
    net.seen = 0;
    save_weights(net, outfile);
}

void visualize(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    visualize_network(net);
    cvWaitKey(0);
}

int main(int argc, char **argv)
{
    //test_box();
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
    } else if (0 == strcmp(argv[1], "writing")){
        run_writing(argc, argv);
    } else if (0 == strcmp(argv[1], "test")){
        test_resize(argv[2]);
    } else if (0 == strcmp(argv[1], "captcha")){
        run_captcha(argc, argv);
    } else if (0 == strcmp(argv[1], "change")){
        change_rate(argv[2], atof(argv[3]), (argc > 4) ? atof(argv[4]) : 0);
    } else if (0 == strcmp(argv[1], "partial")){
        partial(argv[2], argv[3], argv[4], atoi(argv[5]));
    } else if (0 == strcmp(argv[1], "visualize")){
        visualize(argv[2], (argc > 3) ? argv[3] : 0);
    } else {
        fprintf(stderr, "Not an option: %s\n", argv[1]);
    }
    return 0;
}

