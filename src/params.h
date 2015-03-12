#ifndef PARAMS_H
#define PARAMS_H

typedef struct {
    float *truth;
    float *input;
    float *delta;
    int train;
} network_state;

#endif

