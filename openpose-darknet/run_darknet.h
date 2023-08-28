#ifndef RUN_DARKNET_H
#define RUN_DARKNET_H

extern "C"
{

void init_net
    (
    char *cfgfile,
    char *weightfile,
    int *inw,
    int *inh,
    int *outw,
    int *outh
    );

float *run_net
    (
    float *indata
    );

}
#endif // RUN_DARKNET_H
