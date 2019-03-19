#ifndef DARKNET_API
#define DARKNET_API
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#ifdef GPU
    #define BLOCK 512

    #include "cuda_runtime.h"
    #include "curand.h"
    #include "cublas_v2.h"

    #ifdef CUDNN
    #include "cudnn.h"
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SECRET_NUM -1234
extern int gpu_index;

typedef struct{
    int classes;
    char **names;
} metadata;

metadata get_metadata(char *file);

typedef struct{
    int *leaf;
    int n;
    int *parent;
    int *child;
    int *group;
    char **name;

    int groups;
    int *group_size;
    int *group_offset;
} dn_tree;
dn_tree *read_tree(const char *filename);

typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
} ACTIVATION;

typedef enum{
    PNG, BMP, TGA, JPG
} IMTYPE;

typedef enum{
    MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;

typedef enum {
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLO,
    ISEG,
    REORG,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    BLANK
} LAYER_TYPE;

typedef enum{
    SSE, MASKED, L1, SEG, SMOOTH,WGAN
} COST_TYPE;

typedef struct{
    int batch;
    float learning_rate;
    float momentum;
    float decay;
    int adam;
    float B1;
    float B2;
    float eps;
    int t;
} update_args;

struct dn_network;
typedef struct dn_network dn_network;

struct dn_layer;
typedef struct dn_layer dn_layer;

struct dn_layer{
    LAYER_TYPE type;
    ACTIVATION activation;
    COST_TYPE cost_type;
    void (*forward)   (struct dn_layer, struct dn_network);
    void (*backward)  (struct dn_layer, struct dn_network);
    void (*update)    (struct dn_layer, update_args);
    void (*forward_gpu)   (struct dn_layer, struct dn_network);
    void (*backward_gpu)  (struct dn_layer, struct dn_network);
    void (*update_gpu)    (struct dn_layer, update_args);
    int batch_normalize;
    int shortcut;
    int batch;
    int forced;
    int flipped;
    int inputs;
    int outputs;
    int nweights;
    int nbiases;
    int extra;
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
    int flatten;
    int spatial;
    int pad;
    int sqrt;
    int flip;
    int index;
    int binary;
    int xnor;
    int steps;
    int hidden;
    int truth;
    float smooth;
    float dot;
    float angle;
    float jitter;
    float saturation;
    float exposure;
    float shift;
    float ratio;
    float learning_rate_scale;
    float clip;
    int noloss;
    int softmax;
    int classes;
    int coords;
    int background;
    int rescore;
    int objectness;
    int joint;
    int noadjust;
    int reorg;
    int log;
    int tanh;
    int *mask;
    int total;

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
    int dontsave;
    int dontloadscales;
    int numload;

    float temperature;
    float probability;
    float scale;

    char  * cweights;
    int   * indexes;
    int   * input_layers;
    int   * input_sizes;
    int   * map;
    int   * counts;
    float ** sums;
    float * rand;
    float * cost;
    float * state;
    float * prev_state;
    float * forgot_state;
    float * forgot_delta;
    float * state_delta;
    float * combine_cpu;
    float * combine_delta_cpu;

    float * concat;
    float * concat_delta;

    float * binary_weights;

    float * biases;
    float * bias_updates;

    float * scales;
    float * scale_updates;

    float * weights;
    float * weight_updates;

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

    float * m;
    float * v;
    
    float * bias_m;
    float * bias_v;
    float * scale_m;
    float * scale_v;


    float *z_cpu;
    float *r_cpu;
    float *h_cpu;
    float * prev_state_cpu;

    float *temp_cpu;
    float *temp2_cpu;
    float *temp3_cpu;

    float *dh_cpu;
    float *hh_cpu;
    float *prev_cell_cpu;
    float *cell_cpu;
    float *f_cpu;
    float *i_cpu;
    float *g_cpu;
    float *o_cpu;
    float *c_cpu;
    float *dc_cpu; 

    float * binary_input;

    struct dn_layer *input_layer;
    struct dn_layer *self_layer;
    struct dn_layer *output_layer;

    struct dn_layer *reset_layer;
    struct dn_layer *update_layer;
    struct dn_layer *state_layer;

    struct dn_layer *input_gate_layer;
    struct dn_layer *state_gate_layer;
    struct dn_layer *input_save_layer;
    struct dn_layer *state_save_layer;
    struct dn_layer *input_state_layer;
    struct dn_layer *state_state_layer;

    struct dn_layer *input_z_layer;
    struct dn_layer *state_z_layer;

    struct dn_layer *input_r_layer;
    struct dn_layer *state_r_layer;

    struct dn_layer *input_h_layer;
    struct dn_layer *state_h_layer;
	
    struct dn_layer *wz;
    struct dn_layer *uz;
    struct dn_layer *wr;
    struct dn_layer *ur;
    struct dn_layer *wh;
    struct dn_layer *uh;
    struct dn_layer *uo;
    struct dn_layer *wo;
    struct dn_layer *uf;
    struct dn_layer *wf;
    struct dn_layer *ui;
    struct dn_layer *wi;
    struct dn_layer *ug;
    struct dn_layer *wg;

    dn_tree *softmax_tree;

    size_t workspace_size;

#ifdef GPU
    int *indexes_gpu;

    float *z_gpu;
    float *r_gpu;
    float *h_gpu;

    float *temp_gpu;
    float *temp2_gpu;
    float *temp3_gpu;

    float *dh_gpu;
    float *hh_gpu;
    float *prev_cell_gpu;
    float *cell_gpu;
    float *f_gpu;
    float *i_gpu;
    float *g_gpu;
    float *o_gpu;
    float *c_gpu;
    float *dc_gpu; 

    float *m_gpu;
    float *v_gpu;
    float *bias_m_gpu;
    float *scale_m_gpu;
    float *bias_v_gpu;
    float *scale_v_gpu;

    float * combine_gpu;
    float * combine_delta_gpu;

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

    float * binary_input_gpu;
    float * binary_weights_gpu;

    float * mean_gpu;
    float * variance_gpu;

    float * rolling_mean_gpu;
    float * rolling_variance_gpu;

    float * variance_delta_gpu;
    float * mean_delta_gpu;

    float * x_gpu;
    float * x_norm_gpu;
    float * weights_gpu;
    float * weight_updates_gpu;
    float * weight_change_gpu;

    float * biases_gpu;
    float * bias_updates_gpu;
    float * bias_change_gpu;

    float * scales_gpu;
    float * scale_updates_gpu;
    float * scale_change_gpu;

    float * output_gpu;
    float * loss_gpu;
    float * delta_gpu;
    float * rand_gpu;
    float * squared_gpu;
    float * norms_gpu;
#ifdef CUDNN
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
    cudnnTensorDescriptor_t normTensorDesc;
    cudnnFilterDescriptor_t weightDesc;
    cudnnFilterDescriptor_t dweightDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fw_algo;
    cudnnConvolutionBwdDataAlgo_t bd_algo;
    cudnnConvolutionBwdFilterAlgo_t bf_algo;
#endif
#endif
};

void free_layer(dn_layer);

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct dn_network{
    int n;
    int batch;
    size_t *seen;
    int *t;
    float epoch;
    int subdivisions;
    dn_layer *layers;
    float *output;
    learning_rate_policy policy;

    float learning_rate;
    float momentum;
    float decay;
    float gamma;
    float scale;
    float power;
    int time_steps;
    int step;
    int max_batches;
    float *scales;
    int   *steps;
    int num_steps;
    int burn_in;

    int adam;
    float B1;
    float B2;
    float eps;

    int inputs;
    int outputs;
    int truths;
    int notruth;
    int h, w, c;
    int max_crop;
    int min_crop;
    float max_ratio;
    float min_ratio;
    int center;
    float angle;
    float aspect;
    float exposure;
    float saturation;
    float hue;
    int random;

    int gpu_index;
    dn_tree *hierarchy;

    float *input;
    float *truth;
    float *delta;
    float *workspace;
    int train;
    int index;
    float *cost;
    float clip;

#ifdef GPU
    float *input_gpu;
    float *truth_gpu;
    float *delta_gpu;
    float *output_gpu;
#endif

} dn_network;

typedef struct {
    int w;
    int h;
    float scale;
    float rad;
    float dx;
    float dy;
    float aspect;
} augment_args;

typedef struct {
    int w;
    int h;
    int c;
    float *data;
} dn_image;

typedef struct{
    float x, y, w, h;
} dn_box;

typedef struct detection{
    dn_box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

typedef struct {
    int rows, cols;
    float **vals;
} dn_matrix;


typedef struct{
    int w, h;
    dn_matrix X;
    dn_matrix y;
    int shallow;
    int *num_boxes;
    dn_box **boxes;
} dn_data;

typedef enum {
    CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA, SEGMENTATION_DATA, INSTANCE_DATA, ISEG_DATA
} dn_data_type;

typedef struct dn_load_args{
    int threads;
    char **paths;
    char *path;
    int n;
    int m;
    char **labels;
    int h;
    int w;
    int out_w;
    int out_h;
    int nh;
    int nw;
    int num_boxes;
    int min, max, size;
    int classes;
    int background;
    int scale;
    int center;
    int coords;
    float jitter;
    float angle;
    float aspect;
    float saturation;
    float exposure;
    float hue;
    dn_data *d;
    dn_image *im;
    dn_image *resized;
    dn_data_type type;
    dn_tree *hierarchy;
} dn_load_args;

typedef struct{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} dn_box_label;


dn_network *load_network(const char *cfg, const char *weights, int clear);
dn_load_args get_base_args(dn_network *net);

void free_data(dn_data d);

typedef struct dn_node{
    void *val;
    struct dn_node *next;
    struct dn_node *prev;
} dn_node;

typedef struct dn_list{
    int size;
    dn_node *front;
    dn_node *back;
} dn_list;

pthread_t load_data(dn_load_args args);
dn_list *read_data_cfg(const char *filename);
dn_list *read_cfg(const char *filename);
unsigned char *read_file(const char *filename);
dn_data resize_data(dn_data orig, int w, int h);
dn_data *tile_data(dn_data orig, int divs, int size);
dn_data select_data(dn_data *orig, int *inds);

void forward_network(dn_network *net);
void backward_network(dn_network *net);
void update_network(dn_network *net);


float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
void fill_cpu(int N, float ALPHA, float * X, int INCX);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void softmax(float *input, int n, float temp, int stride, float *output);

int best_3d_shift_r(dn_image a, dn_image b, int min, int max);
#ifdef GPU
void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
void fill_gpu(int N, float ALPHA, float * X, int INCX);
void scal_gpu(int N, float ALPHA, float * X, int INCX);
void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);

void cuda_set_device(int n);
void cuda_free(float *x_gpu);
float *cuda_make_array(float *x, size_t n);
void cuda_pull_array(float *x_gpu, float *x, size_t n);
float cuda_mag_array(float *x_gpu, size_t n);
void cuda_push_array(float *x_gpu, float *x, size_t n);

void forward_network_gpu(network *net);
void backward_network_gpu(network *net);
void update_network_gpu(network *net);

float train_networks(network **nets, int n, data d, int interval);
void sync_nets(network **nets, int n, int interval);
void harmless_update_network_gpu(network *net);
#endif
dn_image get_label(dn_image **characters, char *string, int size);
void draw_label(dn_image a, int r, int c, dn_image label, const float *rgb);
void save_image(dn_image im, const char *name);
void save_image_options(dn_image im, const char *name, IMTYPE f, int quality);
void get_next_batch(dn_data d, int n, int offset, float *X, float *y);
void grayscale_image_3c(dn_image im);
void normalize_image(dn_image p);
void matrix_to_csv(dn_matrix m);
float train_network_sgd(dn_network *net, dn_data d, int n);
void rgbgr_image(dn_image im);
dn_data copy_data(dn_data d);
dn_data concat_data(dn_data d1, dn_data d2);
dn_data load_cifar10_data(const char *filename);
float matrix_topk_accuracy(dn_matrix truth, dn_matrix guess, int k);
void matrix_add_matrix(dn_matrix from, dn_matrix to);
void scale_matrix(dn_matrix m, float scale);
dn_matrix csv_to_matrix(const char *filename);
float *network_accuracies(dn_network *net, dn_data d, int n);
float train_network_datum(dn_network *net);
dn_image make_random_image(int w, int h, int c);

void denormalize_connected_layer(dn_layer l);
void denormalize_convolutional_layer(dn_layer l);
void statistics_connected_layer(dn_layer l);
void rescale_weights(dn_layer l, float scale, float trans);
void rgbgr_weights(dn_layer l);
dn_image *get_weights(dn_layer l);

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, int avg, float hier_thresh, int w, int h, int fps, int fullscreen);
void get_detection_detections(dn_layer l, int w, int h, float thresh, detection *dets);

char *option_find_str(dn_list *l, char *key, char *def);
int option_find_int(dn_list *l, char *key, int def);
int option_find_int_quiet(dn_list *l, char *key, int def);

dn_network *parse_network_cfg(const char *filename);
void save_weights(dn_network *net, const char *filename);
void load_weights(dn_network *net, const char *filename);
void save_weights_upto(dn_network *net, const char *filename, int cutoff);
void load_weights_upto(dn_network *net, const char *filename, int start, int cutoff);

void zero_objectness(dn_layer l);
void get_region_detections(dn_layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
int get_yolo_detections(dn_layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets);
void free_network(dn_network *net);
void set_batch_network(dn_network *net, int b);
void set_temp_network(dn_network *net, float t);
dn_image load_image(const char *filename, int w, int h, int c);
dn_image load_image_color(const char *filename, int w, int h);
dn_image make_image(int w, int h, int c);
dn_image resize_image(dn_image im, int w, int h);
void censor_image(dn_image im, int dx, int dy, int w, int h);
dn_image letterbox_image(dn_image im, int w, int h);
dn_image crop_image(dn_image im, int dx, int dy, int w, int h);
dn_image center_crop_image(dn_image im, int w, int h);
dn_image resize_min(dn_image im, int min);
dn_image resize_max(dn_image im, int max);
dn_image threshold_image(dn_image im, float thresh);
dn_image mask_to_rgb(dn_image mask);
int resize_network(dn_network *net, int w, int h);
void free_matrix(dn_matrix m);
void test_resize(const char *filename);
int show_image(dn_image p, const char *name, int ms);
dn_image copy_image(dn_image p);
void draw_box_width(dn_image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
float get_current_rate(dn_network *net);
void composite_3d(char *f1, char *f2, char *out, int delta);
dn_data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h);
size_t get_current_batch(dn_network *net);
void constrain_image(dn_image im);
dn_image get_network_image_layer(dn_network *net, int i);
dn_layer get_network_output_layer(dn_network *net);
void top_predictions(dn_network *net, int n, int *index);
void flip_image(dn_image a);
dn_image float_to_image(int w, int h, int c, float *data);
void ghost_image(dn_image source, dn_image dest, int dx, int dy);
float network_accuracy(dn_network *net, dn_data d);
void random_distort_image(dn_image im, float hue, float saturation, float exposure);
void fill_image(dn_image m, float s);
dn_image grayscale_image(dn_image im);
void rotate_image_cw(dn_image im, int times);
double what_time_is_it_now();
dn_image rotate_image(dn_image m, float rad);
void visualize_network(dn_network *net);
float box_iou(dn_box a, dn_box b);
dn_data load_all_cifar10();
dn_box_label *read_boxes(const char *filename, int *n);
dn_box float_to_box(float *f, int stride);
void draw_detections(dn_image im, detection *dets, int num, float thresh, char **names, dn_image **alphabet, int classes);

dn_matrix network_predict_data(dn_network *net, dn_data test);
dn_image **load_alphabet();
dn_image get_network_image(dn_network *net);
float *network_predict(dn_network *net, float *input);

int network_width(dn_network *net);
int network_height(dn_network *net);
float *network_predict_image(dn_network *net, dn_image im);
void network_detect(dn_network *net, dn_image im, float thresh, float hier_thresh, float nms, detection *dets);
detection *get_network_boxes(dn_network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
void free_detections(detection *dets, int n);

void reset_network_state(dn_network *net, int b);

char **get_labels(const char *filename);
void do_nms_obj(detection *dets, int total, int classes, float thresh);
void do_nms_sort(detection *dets, int total, int classes, float thresh);

dn_matrix make_matrix(int rows, int cols);

#ifdef OPENCV
void *open_video_stream(const char *f, int c, int w, int h, int fps);
image get_image_from_stream(void *p);
void make_window(char *name, int w, int h, int fullscreen);
#endif

void free_image(dn_image m);
float train_network(dn_network *net, dn_data d);
pthread_t load_data_in_thread(dn_load_args args);
void load_data_blocking(dn_load_args args);
dn_list *get_paths(const char *filename);
void hierarchy_predictions(float *predictions, int n, dn_tree *hier, int only_leaves, int stride);
void change_leaves(dn_tree *t, char *leaf_list);

int find_int_arg(int argc, char **argv, char *arg, int def);
float find_float_arg(int argc, char **argv, char *arg, float def);
int find_arg(int argc, char* argv[], char *arg);
char *find_char_arg(int argc, char **argv, char *arg, char *def);
char *basecfg(char *cfgfile);
void find_replace(char *str, char *orig, char *rep, char *output);
void free_ptrs(void **ptrs, int n);
char *fgetl(FILE *fp);
void strip(char *s);
float sec(clock_t clocks);
void **list_to_array(dn_list *l);
void top_k(float *a, int n, int k, int *index);
int *read_map(const char *filename);
void error(const char *s);
int max_index(float *a, int n);
int max_int_index(int *a, int n);
int sample_array(float *a, int n);
int *random_index_order(int min, int max);
void free_list(dn_list *l);
float mse_array(float *a, int n);
float variance_array(float *a, int n);
float mag_array(float *a, int n);
void scale_array(float *a, int n, float s);
float mean_array(float *a, int n);
float sum_array(float *a, int n);
void normalize_array(float *a, int n);
int *read_intlist(char *s, int *n, int d);
size_t rand_size_t();
float rand_normal();
float rand_uniform(float min, float max);

#ifdef __cplusplus
}
#endif
#endif
