#ifndef COST_LAYER_H
#define COST_LAYER_H

typedef enum{
    SSE
} COST_TYPE;

typedef struct {
    int inputs;
    int batch;
    int coords;
    int classes;
    float *delta;
    float *output;
    COST_TYPE type;
    #ifdef GPU
    float * delta_gpu;
    #endif
} cost_layer;

COST_TYPE get_cost_type(char *s);
char *get_cost_string(COST_TYPE a);
cost_layer *make_cost_layer(int batch, int inputs, COST_TYPE type);
void forward_cost_layer(const cost_layer layer, float *input, float *truth);
void backward_cost_layer(const cost_layer layer, float *input, float *delta);

#ifdef GPU
void forward_cost_layer_gpu(cost_layer layer, float * input, float * truth);
void backward_cost_layer_gpu(const cost_layer layer, float * input, float * delta);
#endif

#endif
