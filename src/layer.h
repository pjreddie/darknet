#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include "activations.h"
#include "stddef.h"
#include "tree.h"
#ifdef __cplusplus
extern "C" {
#endif

//struct network_state;

//struct layer;
//typedef struct layer layer;

//typedef enum {
//    CONVOLUTIONAL,
//    DECONVOLUTIONAL,
//    CONNECTED,
//    MAXPOOL,
//    SOFTMAX,
//    DETECTION,
//    DROPOUT,
//    CROP,
//    ROUTE,
//    COST,
//    NORMALIZATION,
//    AVGPOOL,
//    LOCAL,
//    SHORTCUT,
//    ACTIVE,
//    RNN,
//    GRU,
//    CRNN,
//    BATCHNORM,
//    NETWORK,
//    XNOR,
//    REGION,
//    YOLO,
//    REORG,
//    UPSAMPLE,
//    REORG_OLD,
//    BLANK
//} LAYER_TYPE;

//typedef enum{
//    SSE, MASKED, SMOOTH
//} COST_TYPE;

//typedef struct {
//    int batch;
//    float learning_rate;
//    float momentum;
//    float decay;
//    int adam;
//    float B1;
//    float B2;
//    float eps;
//    int t;
//} update_args;

/*
struct layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    void (*forward)   (struct layer, struct network_state);
    void (*backward)  (struct layer, struct network_state);
    void (*update)    (struct layer, int, float, float, float);
    void (*forward_gpu)   (struct layer, struct network_state);
    void (*backward_gpu)  (struct layer, struct network_state);
    void (*update_gpu)    (struct layer, int, float, float, float);
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int truths;
    int h,w,c;
    int out_h, out_w, out_c;
    int n;
    int max_boxes;
    int groups;
    int size;
    int side;
    int stride;
    int reverse;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int use_bin_output;
    int steps;
    int hidden;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    int focal_loss;
    int noloss;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int does_cost;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;
    float bflops;

    int adam;
    float B1;
    float B2;
    float eps;

    int t;
    float *m;
    float *v;
    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;

    tree *softmax_tree;
    int  *map;

    float alpha;
    float beta;
    float kappa;

    float coord_scale;
    float object_scale;
    float noobject_scale;
    float mask_scale;
    float class_scale;
    int bias_match;
    int random;
    float ignore_thresh;
    float truth_thresh;
    float thresh;
    float focus;
    int classfix;
    int absolute;

    int onlyforward;
    int stopbackward;
    int dontload;
    int dontloadscales;

    float temperature;
    float probability;
    float scale;

    int *indexes;
    float *rand;
    float *cost;
    char  *cweights;
    float *state;
    float *prev_state;
    float *forgot_state;
    float *forgot_delta;
    float *state_delta;

    float *concat;
    float *concat_delta;

    float *binary_weights;

    float *biases;
    float *bias_updates;

    float *scales;
    float *scale_updates;

    float *weights;
    float *weight_updates;

    char *align_bit_weights_gpu;
    float *mean_arr_gpu;
    float *align_workspace_gpu;
    float *transposed_align_workspace_gpu;
    int align_workspace_size;

    char *align_bit_weights;
    float *mean_arr;
    int align_bit_weights_size;
    int lda_align;
    int new_lda;
    int bit_align;

    float *col_image;
    int   * input_layers;
    int   * input_sizes;
    float * delta;
    float * output;
    float * loss;
    float * squared;
    float * norms;

    float * spatial_mean;
    float * mean;
    float * variance;

    float * mean_delta;
    float * variance_delta;

    float * rolling_mean;
    float * rolling_variance;

    float * x;
    float * x_norm;

    struct layer *input_layer;
    struct layer *self_layer;
    struct layer *output_layer;

    struct layer *input_gate_layer;
    struct layer *state_gate_layer;
    struct layer *input_save_layer;
    struct layer *state_save_layer;
    struct layer *input_state_layer;
    struct layer *state_state_layer;

    struct layer *input_z_layer;
    struct layer *state_z_layer;

    struct layer *input_r_layer;
    struct layer *state_r_layer;

    struct layer *input_h_layer;
    struct layer *state_h_layer;

    float *z_cpu;
    float *r_cpu;
    float *h_cpu;

    float *binary_input;

    size_t workspace_size;

#ifdef GPU
    float *z_gpu;
    float *r_gpu;
    float *h_gpu;

    int *indexes_gpu;
    float * prev_state_gpu;
    float * forgot_state_gpu;
    float * forgot_delta_gpu;
    float * state_gpu;
    float * state_delta_gpu;
    float * gate_gpu;
    float * gate_delta_gpu;
    float * save_gpu;
    float * save_delta_gpu;
    float * concat_gpu;
    float * concat_delta_gpu;

    // adam
    float *m_gpu;
    float *v_gpu;
    float *bias_m_gpu;
    float *scale_m_gpu;
    float *bias_v_gpu;
    float *scale_v_gpu;

    float *binary_input_gpu;
    float *binary_weights_gpu;

    float * mean_gpu;
    float * variance_gpu;

    float * rolling_mean_gpu;
    float * rolling_variance_gpu;

    float * variance_delta_gpu;
    float * mean_delta_gpu;

    float * col_image_gpu;

    float * x_gpu;
    float * x_norm_gpu;
    float * weights_gpu;
    float * weight_updates_gpu;

    float * weights_gpu16;
    float * weight_updates_gpu16;

    float * biases_gpu;
    float * bias_updates_gpu;

    float * scales_gpu;
    float * scale_updates_gpu;

    float * output_gpu;
    float * loss_gpu;
    float * delta_gpu;
    float * rand_gpu;
    float * squared_gpu;
    float * norms_gpu;
    #ifdef CUDNN
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t srcTensorDesc16, dstTensorDesc16;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc16, ddstTensorDesc16;
    cudnnTensorDescriptor_t normTensorDesc, normDstTensorDesc, normDstTensorDescF16;
    cudnnFilterDescriptor_t weightDesc, weightDesc16;
    cudnnFilterDescriptor_t dweightDesc, dweightDesc16;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fw_algo, fw_algo16;
    cudnnConvolutionBwdDataAlgo_t bd_algo, bd_algo16;
    cudnnConvolutionBwdFilterAlgo_t bf_algo, bf_algo16;
    cudnnPoolingDescriptor_t poolingDesc;
    #endif  // CUDNN
#endif  // GPU
};
*/
//void free_layer(layer);

#ifdef __cplusplus
}
#endif
#endif
