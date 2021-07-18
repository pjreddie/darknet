#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include "activation_layer.h"
#include "activations.h"
#include "assert.h"
#include "avgpool_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "connected_layer.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "crnn_layer.h"
#include "crop_layer.h"
#include "detection_layer.h"
#include "dropout_layer.h"
#include "gru_layer.h"
#include "list.h"
#include "local_layer.h"
#include "lstm_layer.h"
#include "conv_lstm_layer.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "reorg_layer.h"
#include "reorg_old_layer.h"
#include "rnn_layer.h"
#include "route_layer.h"
#include "shortcut_layer.h"
#include "scale_channels_layer.h"
#include "sam_layer.h"
#include "softmax_layer.h"
#include "utils.h"
#include "upsample_layer.h"
#include "version.h"
#include "yolo_layer.h"
#include "gaussian_yolo_layer.h"
#include "representation_layer.h"

void empty_func(dropout_layer l, network_state state) {
    //l.output_gpu = state.input;
}

typedef struct{
    char *type;
    list *options;
}section;

list *read_cfg(char *filename);

LAYER_TYPE string_to_layer_type(char * type)
{

    if (strcmp(type, "[shortcut]")==0) return SHORTCUT;
    if (strcmp(type, "[scale_channels]") == 0) return SCALE_CHANNELS;
    if (strcmp(type, "[sam]") == 0) return SAM;
    if (strcmp(type, "[crop]")==0) return CROP;
    if (strcmp(type, "[cost]")==0) return COST;
    if (strcmp(type, "[detection]")==0) return DETECTION;
    if (strcmp(type, "[region]")==0) return REGION;
    if (strcmp(type, "[yolo]") == 0) return YOLO;
    if (strcmp(type, "[Gaussian_yolo]") == 0) return GAUSSIAN_YOLO;
    if (strcmp(type, "[local]")==0) return LOCAL;
    if (strcmp(type, "[conv]")==0
            || strcmp(type, "[convolutional]")==0) return CONVOLUTIONAL;
    if (strcmp(type, "[activation]")==0) return ACTIVE;
    if (strcmp(type, "[net]")==0
            || strcmp(type, "[network]")==0) return NETWORK;
    if (strcmp(type, "[crnn]")==0) return CRNN;
    if (strcmp(type, "[gru]")==0) return GRU;
    if (strcmp(type, "[lstm]")==0) return LSTM;
    if (strcmp(type, "[conv_lstm]") == 0) return CONV_LSTM;
    if (strcmp(type, "[history]") == 0) return HISTORY;
    if (strcmp(type, "[rnn]")==0) return RNN;
    if (strcmp(type, "[conn]")==0
            || strcmp(type, "[connected]")==0) return CONNECTED;
    if (strcmp(type, "[max]")==0
            || strcmp(type, "[maxpool]")==0) return MAXPOOL;
    if (strcmp(type, "[local_avg]") == 0
        || strcmp(type, "[local_avgpool]") == 0) return LOCAL_AVGPOOL;
    if (strcmp(type, "[reorg3d]")==0) return REORG;
    if (strcmp(type, "[reorg]") == 0) return REORG_OLD;
    if (strcmp(type, "[avg]")==0
            || strcmp(type, "[avgpool]")==0) return AVGPOOL;
    if (strcmp(type, "[dropout]")==0) return DROPOUT;
    if (strcmp(type, "[lrn]")==0
            || strcmp(type, "[normalization]")==0) return NORMALIZATION;
    if (strcmp(type, "[batchnorm]")==0) return BATCHNORM;
    if (strcmp(type, "[soft]")==0
            || strcmp(type, "[softmax]")==0) return SOFTMAX;
    if (strcmp(type, "[contrastive]") == 0) return CONTRASTIVE;
    if (strcmp(type, "[route]")==0) return ROUTE;
    if (strcmp(type, "[upsample]") == 0) return UPSAMPLE;
    if (strcmp(type, "[empty]") == 0
        || strcmp(type, "[silence]") == 0) return EMPTY;
    if (strcmp(type, "[implicit]") == 0) return IMPLICIT;
    return BLANK;
}

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
    int index;
    int time_steps;
    int train;
    network net;
} size_params;

local_layer parse_local(list *options, size_params params)
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
    if(!(h && w && c)) error("Layer before local layer must output image.", DARKNET_LOC);

    local_layer layer = make_local_layer(batch,h,w,c,n,size,stride,pad,activation);

    return layer;
}

convolutional_layer parse_convolutional(list *options, size_params params)
{
    int n = option_find_int(options, "filters",1);
    int groups = option_find_int_quiet(options, "groups", 1);
    int size = option_find_int(options, "size",1);
    int stride = -1;
    //int stride = option_find_int(options, "stride",1);
    int stride_x = option_find_int_quiet(options, "stride_x", -1);
    int stride_y = option_find_int_quiet(options, "stride_y", -1);
    if (stride_x < 1 || stride_y < 1) {
        stride = option_find_int(options, "stride", 1);
        if (stride_x < 1) stride_x = stride;
        if (stride_y < 1) stride_y = stride;
    }
    else {
        stride = option_find_int_quiet(options, "stride", 1);
    }
    int dilation = option_find_int_quiet(options, "dilation", 1);
    int antialiasing = option_find_int_quiet(options, "antialiasing", 0);
    if (size == 1) dilation = 1;
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    if(pad) padding = size/2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int assisted_excitation = option_find_float_quiet(options, "assisted_excitation", 0);

    int share_index = option_find_int_quiet(options, "share_index", -1000000000);
    convolutional_layer *share_layer = NULL;
    if(share_index >= 0) share_layer = &params.net.layers[share_index];
    else if(share_index != -1000000000) share_layer = &params.net.layers[params.index + share_index];

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.", DARKNET_LOC);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int cbn = option_find_int_quiet(options, "cbn", 0);
    if (cbn) batch_normalize = 2;
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);
    int use_bin_output = option_find_int_quiet(options, "bin_output", 0);
    int sway = option_find_int_quiet(options, "sway", 0);
    int rotate = option_find_int_quiet(options, "rotate", 0);
    int stretch = option_find_int_quiet(options, "stretch", 0);
    int stretch_sway = option_find_int_quiet(options, "stretch_sway", 0);
    if ((sway + rotate + stretch + stretch_sway) > 1) {
        printf(" Error: should be used only 1 param: sway=1, rotate=1 or stretch=1 in the [convolutional] layer \n");
        exit(0);
    }
    int deform = sway || rotate || stretch || stretch_sway;
    if (deform && size == 1) {
        printf(" Error: params (sway=1, rotate=1 or stretch=1) should be used only with size >=3 in the [convolutional] layer \n");
        exit(0);
    }

    convolutional_layer layer = make_convolutional_layer(batch,1,h,w,c,n,groups,size,stride_x,stride_y,dilation,padding,activation, batch_normalize, binary, xnor, params.net.adam, use_bin_output, params.index, antialiasing, share_layer, assisted_excitation, deform, params.train);
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);
    layer.sway = sway;
    layer.rotate = rotate;
    layer.stretch = stretch;
    layer.stretch_sway = stretch_sway;
    layer.angle = option_find_float_quiet(options, "angle", 15);
    layer.grad_centr = option_find_int_quiet(options, "grad_centr", 0);
    layer.reverse = option_find_float_quiet(options, "reverse", 0);
    layer.coordconv = option_find_int_quiet(options, "coordconv", 0);

    layer.stream = option_find_int_quiet(options, "stream", -1);
    layer.wait_stream_id = option_find_int_quiet(options, "wait_stream", -1);

    if(params.net.adam){
        layer.B1 = params.net.B1;
        layer.B2 = params.net.B2;
        layer.eps = params.net.eps;
    }

    return layer;
}

layer parse_crnn(list *options, size_params params)
{
    int size = option_find_int_quiet(options, "size", 3);
    int stride = option_find_int_quiet(options, "stride", 1);
    int dilation = option_find_int_quiet(options, "dilation", 1);
    int pad = option_find_int_quiet(options, "pad", 0);
    int padding = option_find_int_quiet(options, "padding", 0);
    if (pad) padding = size / 2;

    int output_filters = option_find_int(options, "output",1);
    int hidden_filters = option_find_int(options, "hidden",1);
    int groups = option_find_int_quiet(options, "groups", 1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);

    layer l = make_crnn_layer(params.batch, params.h, params.w, params.c, hidden_filters, output_filters, groups, params.time_steps, size, stride, dilation, padding, activation, batch_normalize, xnor, params.train);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
}

layer parse_rnn(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    int hidden = option_find_int(options, "hidden",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int logistic = option_find_int_quiet(options, "logistic", 0);

    layer l = make_rnn_layer(params.batch, params.inputs, hidden, output, params.time_steps, activation, batch_normalize, logistic);

    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    return l;
}

layer parse_gru(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_gru_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize);

    return l;
}

layer parse_lstm(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    layer l = make_lstm_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize);

    return l;
}

layer parse_conv_lstm(list *options, size_params params)
{
    // a ConvLSTM with a larger transitional kernel should be able to capture faster motions
    int size = option_find_int_quiet(options, "size", 3);
    int stride = option_find_int_quiet(options, "stride", 1);
    int dilation = option_find_int_quiet(options, "dilation", 1);
    int pad = option_find_int_quiet(options, "pad", 0);
    int padding = option_find_int_quiet(options, "padding", 0);
    if (pad) padding = size / 2;

    int output_filters = option_find_int(options, "output", 1);
    int groups = option_find_int_quiet(options, "groups", 1);
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);
    int peephole = option_find_int_quiet(options, "peephole", 0);
    int bottleneck = option_find_int_quiet(options, "bottleneck", 0);

    layer l = make_conv_lstm_layer(params.batch, params.h, params.w, params.c, output_filters, groups, params.time_steps, size, stride, dilation, padding, activation, batch_normalize, peephole, xnor, bottleneck, params.train);

    l.state_constrain = option_find_int_quiet(options, "state_constrain", params.time_steps * 32);
    l.shortcut = option_find_int_quiet(options, "shortcut", 0);

    char *lstm_activation_s = option_find_str(options, "lstm_activation", "tanh");
    l.lstm_activation = get_activation(lstm_activation_s);
    l.time_normalizer = option_find_float_quiet(options, "time_normalizer", 1.0);

    return l;
}

layer parse_history(list *options, size_params params)
{
    int history_size = option_find_int(options, "history_size", 4);
    layer l = make_history_layer(params.batch, params.h, params.w, params.c, history_size, params.time_steps, params.train);
    return l;
}

connected_layer parse_connected(list *options, size_params params)
{
    int output = option_find_int(options, "output",1);
    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);

    connected_layer layer = make_connected_layer(params.batch, 1, params.inputs, output, activation, batch_normalize);

    return layer;
}

softmax_layer parse_softmax(list *options, size_params params)
{
	int groups = option_find_int_quiet(options, "groups", 1);
	softmax_layer layer = make_softmax_layer(params.batch, params.inputs, groups);
	layer.temperature = option_find_float_quiet(options, "temperature", 1);
	char *tree_file = option_find_str(options, "tree", 0);
	if (tree_file) layer.softmax_tree = read_tree(tree_file);
	layer.w = params.w;
	layer.h = params.h;
	layer.c = params.c;
	layer.spatial = option_find_float_quiet(options, "spatial", 0);
	layer.noloss = option_find_int_quiet(options, "noloss", 0);
	return layer;
}

contrastive_layer parse_contrastive(list *options, size_params params)
{
    int classes = option_find_int(options, "classes", 1000);
    layer *yolo_layer = NULL;
    int yolo_layer_id = option_find_int_quiet(options, "yolo_layer", 0);
    if (yolo_layer_id < 0) yolo_layer_id = params.index + yolo_layer_id;
    if(yolo_layer_id != 0) yolo_layer = params.net.layers + yolo_layer_id;
    if (yolo_layer->type != YOLO) {
        printf(" Error: [contrastive] layer should point to the [yolo] layer instead of %d layer! \n", yolo_layer_id);
        getchar();
        exit(0);
    }

    contrastive_layer layer = make_contrastive_layer(params.batch, params.w, params.h, params.c, classes, params.inputs, yolo_layer);
    layer.temperature = option_find_float_quiet(options, "temperature", 1);
    layer.steps = params.time_steps;
    layer.cls_normalizer = option_find_float_quiet(options, "cls_normalizer", 1);
    layer.max_delta = option_find_float_quiet(options, "max_delta", FLT_MAX);   // set 10
    layer.contrastive_neg_max = option_find_int_quiet(options, "contrastive_neg_max", 3);
    return layer;
}

int *parse_yolo_mask(char *a, int *num)
{
    int *mask = 0;
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == '#') break;
            if (a[i] == ',') ++n;
        }
        mask = (int*)xcalloc(n, sizeof(int));
        for (i = 0; i < n; ++i) {
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',') + 1;
        }
        *num = n;
    }
    return mask;
}

float *get_classes_multipliers(char *cpc, const int classes, const float max_delta)
{
    float *classes_multipliers = NULL;
    if (cpc) {
        int classes_counters = classes;
        int *counters_per_class = parse_yolo_mask(cpc, &classes_counters);
        if (classes_counters != classes) {
            printf(" number of values in counters_per_class = %d doesn't match with classes = %d \n", classes_counters, classes);
            exit(0);
        }
        float max_counter = 0;
        int i;
        for (i = 0; i < classes_counters; ++i) {
            if (counters_per_class[i] < 1) counters_per_class[i] = 1;
            if (max_counter < counters_per_class[i]) max_counter = counters_per_class[i];
        }
        classes_multipliers = (float *)calloc(classes_counters, sizeof(float));
        for (i = 0; i < classes_counters; ++i) {
            classes_multipliers[i] = max_counter / counters_per_class[i];
            if(classes_multipliers[i] > max_delta) classes_multipliers[i] = max_delta;
        }
        free(counters_per_class);
        printf(" classes_multipliers: ");
        for (i = 0; i < classes_counters; ++i) printf("%.1f, ", classes_multipliers[i]);
        printf("\n");
    }
    return classes_multipliers;
}

layer parse_yolo(list *options, size_params params)
{
    int classes = option_find_int(options, "classes", 20);
    int total = option_find_int(options, "num", 1);
    int num = total;
    char *a = option_find_str(options, "mask", 0);
    int *mask = parse_yolo_mask(a, &num);
    int max_boxes = option_find_int_quiet(options, "max", 200);
    layer l = make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes, max_boxes);
    if (l.outputs != params.inputs) {
        printf("Error: l.outputs == params.inputs \n");
        printf("filters= in the [convolutional]-layer doesn't correspond to classes= or mask= in [yolo]-layer \n");
        exit(EXIT_FAILURE);
    }
    //assert(l.outputs == params.inputs);

    l.show_details = option_find_int_quiet(options, "show_details", 1);
    l.max_delta = option_find_float_quiet(options, "max_delta", FLT_MAX);   // set 10
    char *cpc = option_find_str(options, "counters_per_class", 0);
    l.classes_multipliers = get_classes_multipliers(cpc, classes, l.max_delta);

    l.label_smooth_eps = option_find_float_quiet(options, "label_smooth_eps", 0.0f);
    l.scale_x_y = option_find_float_quiet(options, "scale_x_y", 1);
    l.objectness_smooth = option_find_int_quiet(options, "objectness_smooth", 0);
    l.new_coords = option_find_int_quiet(options, "new_coords", 0);
    l.iou_normalizer = option_find_float_quiet(options, "iou_normalizer", 0.75);
    l.obj_normalizer = option_find_float_quiet(options, "obj_normalizer", 1);
    l.cls_normalizer = option_find_float_quiet(options, "cls_normalizer", 1);
    l.delta_normalizer = option_find_float_quiet(options, "delta_normalizer", 1);
    char *iou_loss = option_find_str_quiet(options, "iou_loss", "mse");   //  "iou");

    if (strcmp(iou_loss, "mse") == 0) l.iou_loss = MSE;
    else if (strcmp(iou_loss, "giou") == 0) l.iou_loss = GIOU;
    else if (strcmp(iou_loss, "diou") == 0) l.iou_loss = DIOU;
    else if (strcmp(iou_loss, "ciou") == 0) l.iou_loss = CIOU;
    else l.iou_loss = IOU;
    fprintf(stderr, "[yolo] params: iou loss: %s (%d), iou_norm: %2.2f, obj_norm: %2.2f, cls_norm: %2.2f, delta_norm: %2.2f, scale_x_y: %2.2f\n",
        iou_loss, l.iou_loss, l.iou_normalizer, l.obj_normalizer, l.cls_normalizer, l.delta_normalizer, l.scale_x_y);

    char *iou_thresh_kind_str = option_find_str_quiet(options, "iou_thresh_kind", "iou");
    if (strcmp(iou_thresh_kind_str, "iou") == 0) l.iou_thresh_kind = IOU;
    else if (strcmp(iou_thresh_kind_str, "giou") == 0) l.iou_thresh_kind = GIOU;
    else if (strcmp(iou_thresh_kind_str, "diou") == 0) l.iou_thresh_kind = DIOU;
    else if (strcmp(iou_thresh_kind_str, "ciou") == 0) l.iou_thresh_kind = CIOU;
    else {
        fprintf(stderr, " Wrong iou_thresh_kind = %s \n", iou_thresh_kind_str);
        l.iou_thresh_kind = IOU;
    }

    l.beta_nms = option_find_float_quiet(options, "beta_nms", 0.6);
    char *nms_kind = option_find_str_quiet(options, "nms_kind", "default");
    if (strcmp(nms_kind, "default") == 0) l.nms_kind = DEFAULT_NMS;
    else {
        if (strcmp(nms_kind, "greedynms") == 0) l.nms_kind = GREEDY_NMS;
        else if (strcmp(nms_kind, "diounms") == 0) l.nms_kind = DIOU_NMS;
        else l.nms_kind = DEFAULT_NMS;
        printf("nms_kind: %s (%d), beta = %f \n", nms_kind, l.nms_kind, l.beta_nms);
    }

    l.jitter = option_find_float(options, "jitter", .2);
    l.resize = option_find_float_quiet(options, "resize", 1.0);
    l.focal_loss = option_find_int_quiet(options, "focal_loss", 0);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
    l.iou_thresh = option_find_float_quiet(options, "iou_thresh", 1); // recommended to use iou_thresh=0.213 in [yolo]
    l.random = option_find_float_quiet(options, "random", 0);

    l.track_history_size = option_find_int_quiet(options, "track_history_size", 5);
    l.sim_thresh = option_find_int_quiet(options, "sim_thresh", 0.8);
    l.dets_for_track = option_find_int_quiet(options, "dets_for_track", 1);
    l.dets_for_show = option_find_int_quiet(options, "dets_for_show", 1);
    l.track_ciou_norm = option_find_float_quiet(options, "track_ciou_norm", 0.01);
    int embedding_layer_id = option_find_int_quiet(options, "embedding_layer", 999999);
    if (embedding_layer_id < 0) embedding_layer_id = params.index + embedding_layer_id;
    if (embedding_layer_id != 999999) {
        printf(" embedding_layer_id = %d, ", embedding_layer_id);
        layer le = params.net.layers[embedding_layer_id];
        l.embedding_layer_id = embedding_layer_id;
        l.embedding_output = (float*)xcalloc(le.batch * le.outputs, sizeof(float));
        l.embedding_size = le.n / l.n;
        printf(" embedding_size = %d \n", l.embedding_size);
        if (le.n % l.n != 0) {
            printf(" Warning: filters=%d number in embedding_layer=%d isn't divisable by number of anchors %d \n", le.n, embedding_layer_id, l.n);
            getchar();
        }
    }

    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    a = option_find_str(options, "anchors", 0);
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == '#') break;
            if (a[i] == ',') ++n;
        }
        for (i = 0; i < n && i < total*2; ++i) {
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',') + 1;
        }
    }
    return l;
}


int *parse_gaussian_yolo_mask(char *a, int *num) // Gaussian_YOLOv3
{
    int *mask = 0;
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == '#') break;
            if (a[i] == ',') ++n;
        }
        mask = (int *)calloc(n, sizeof(int));
        for (i = 0; i < n; ++i) {
            int val = atoi(a);
            mask[i] = val;
            a = strchr(a, ',') + 1;
        }
        *num = n;
    }
    return mask;
}


layer parse_gaussian_yolo(list *options, size_params params) // Gaussian_YOLOv3
{
    int classes = option_find_int(options, "classes", 20);
    int max_boxes = option_find_int_quiet(options, "max", 200);
    int total = option_find_int(options, "num", 1);
    int num = total;

    char *a = option_find_str(options, "mask", 0);
    int *mask = parse_gaussian_yolo_mask(a, &num);
    layer l = make_gaussian_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes, max_boxes);
    if (l.outputs != params.inputs) {
        printf("Error: l.outputs == params.inputs \n");
        printf("filters= in the [convolutional]-layer doesn't correspond to classes= or mask= in [Gaussian_yolo]-layer \n");
        exit(EXIT_FAILURE);
    }
    //assert(l.outputs == params.inputs);
    l.max_delta = option_find_float_quiet(options, "max_delta", FLT_MAX);   // set 10
    char *cpc = option_find_str(options, "counters_per_class", 0);
    l.classes_multipliers = get_classes_multipliers(cpc, classes, l.max_delta);

    l.label_smooth_eps = option_find_float_quiet(options, "label_smooth_eps", 0.0f);
    l.scale_x_y = option_find_float_quiet(options, "scale_x_y", 1);
    l.objectness_smooth = option_find_int_quiet(options, "objectness_smooth", 0);
    l.uc_normalizer = option_find_float_quiet(options, "uc_normalizer", 1.0);
    l.iou_normalizer = option_find_float_quiet(options, "iou_normalizer", 0.75);
    l.obj_normalizer = option_find_float_quiet(options, "obj_normalizer", 1.0);
    l.cls_normalizer = option_find_float_quiet(options, "cls_normalizer", 1);
    l.delta_normalizer = option_find_float_quiet(options, "delta_normalizer", 1);
    char *iou_loss = option_find_str_quiet(options, "iou_loss", "mse");   //  "iou");

    if (strcmp(iou_loss, "mse") == 0) l.iou_loss = MSE;
    else if (strcmp(iou_loss, "giou") == 0) l.iou_loss = GIOU;
    else if (strcmp(iou_loss, "diou") == 0) l.iou_loss = DIOU;
    else if (strcmp(iou_loss, "ciou") == 0) l.iou_loss = CIOU;
    else l.iou_loss = IOU;

    char *iou_thresh_kind_str = option_find_str_quiet(options, "iou_thresh_kind", "iou");
    if (strcmp(iou_thresh_kind_str, "iou") == 0) l.iou_thresh_kind = IOU;
    else if (strcmp(iou_thresh_kind_str, "giou") == 0) l.iou_thresh_kind = GIOU;
    else if (strcmp(iou_thresh_kind_str, "diou") == 0) l.iou_thresh_kind = DIOU;
    else if (strcmp(iou_thresh_kind_str, "ciou") == 0) l.iou_thresh_kind = CIOU;
    else {
        fprintf(stderr, " Wrong iou_thresh_kind = %s \n", iou_thresh_kind_str);
        l.iou_thresh_kind = IOU;
    }

    l.beta_nms = option_find_float_quiet(options, "beta_nms", 0.6);
    char *nms_kind = option_find_str_quiet(options, "nms_kind", "default");
    if (strcmp(nms_kind, "default") == 0) l.nms_kind = DEFAULT_NMS;
    else {
        if (strcmp(nms_kind, "greedynms") == 0) l.nms_kind = GREEDY_NMS;
        else if (strcmp(nms_kind, "diounms") == 0) l.nms_kind = DIOU_NMS;
        else if (strcmp(nms_kind, "cornersnms") == 0) l.nms_kind = CORNERS_NMS;
        else l.nms_kind = DEFAULT_NMS;
        printf("nms_kind: %s (%d), beta = %f \n", nms_kind, l.nms_kind, l.beta_nms);
    }

    char *yolo_point = option_find_str_quiet(options, "yolo_point", "center");
    if (strcmp(yolo_point, "left_top") == 0) l.yolo_point = YOLO_LEFT_TOP;
    else if (strcmp(yolo_point, "right_bottom") == 0) l.yolo_point = YOLO_RIGHT_BOTTOM;
    else l.yolo_point = YOLO_CENTER;

    fprintf(stderr, "[Gaussian_yolo] iou loss: %s (%d), iou_norm: %2.2f, obj_norm: %2.2f, cls_norm: %2.2f, delta_norm: %2.2f, scale: %2.2f, point: %d\n",
        iou_loss, l.iou_loss, l.iou_normalizer, l.obj_normalizer, l.cls_normalizer, l.delta_normalizer, l.scale_x_y, l.yolo_point);

    l.jitter = option_find_float(options, "jitter", .2);
    l.resize = option_find_float_quiet(options, "resize", 1.0);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
    l.iou_thresh = option_find_float_quiet(options, "iou_thresh", 1); // recommended to use iou_thresh=0.213 in [yolo]
    l.random = option_find_float_quiet(options, "random", 0);

    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    a = option_find_str(options, "anchors", 0);
    if (a) {
        int len = strlen(a);
        int n = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (a[i] == ',') ++n;
        }
        for (i = 0; i < n; ++i) {
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',') + 1;
        }
    }
    return l;
}

layer parse_region(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 4);
    int classes = option_find_int(options, "classes", 20);
    int num = option_find_int(options, "num", 1);
    int max_boxes = option_find_int_quiet(options, "max", 200);

    layer l = make_region_layer(params.batch, params.w, params.h, num, classes, coords, max_boxes);
    if (l.outputs != params.inputs) {
        printf("Error: l.outputs == params.inputs \n");
        printf("filters= in the [convolutional]-layer doesn't correspond to classes= or num= in [region]-layer \n");
        exit(EXIT_FAILURE);
    }
    //assert(l.outputs == params.inputs);

    l.log = option_find_int_quiet(options, "log", 0);
    l.sqrt = option_find_int_quiet(options, "sqrt", 0);

    l.softmax = option_find_int(options, "softmax", 0);
    l.focal_loss = option_find_int_quiet(options, "focal_loss", 0);
    //l.max_boxes = option_find_int_quiet(options, "max",30);
    l.jitter = option_find_float(options, "jitter", .2);
    l.resize = option_find_float_quiet(options, "resize", 1.0);
    l.rescore = option_find_int_quiet(options, "rescore",0);

    l.thresh = option_find_float(options, "thresh", .5);
    l.classfix = option_find_int_quiet(options, "classfix", 0);
    l.absolute = option_find_int_quiet(options, "absolute", 0);
    l.random = option_find_float_quiet(options, "random", 0);

    l.coord_scale = option_find_float(options, "coord_scale", 1);
    l.object_scale = option_find_float(options, "object_scale", 1);
    l.noobject_scale = option_find_float(options, "noobject_scale", 1);
    l.mask_scale = option_find_float(options, "mask_scale", 1);
    l.class_scale = option_find_float(options, "class_scale", 1);
    l.bias_match = option_find_int_quiet(options, "bias_match",0);

    char *tree_file = option_find_str(options, "tree", 0);
    if (tree_file) l.softmax_tree = read_tree(tree_file);
    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    char *a = option_find_str(options, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n && i < num*2; ++i){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}
detection_layer parse_detection(list *options, size_params params)
{
    int coords = option_find_int(options, "coords", 1);
    int classes = option_find_int(options, "classes", 1);
    int rescore = option_find_int(options, "rescore", 0);
    int num = option_find_int(options, "num", 1);
    int side = option_find_int(options, "side", 7);
    detection_layer layer = make_detection_layer(params.batch, params.inputs, num, side, classes, coords, rescore);

    layer.softmax = option_find_int(options, "softmax", 0);
    layer.sqrt = option_find_int(options, "sqrt", 0);

    layer.max_boxes = option_find_int_quiet(options, "max",200);
    layer.coord_scale = option_find_float(options, "coord_scale", 1);
    layer.forced = option_find_int(options, "forced", 0);
    layer.object_scale = option_find_float(options, "object_scale", 1);
    layer.noobject_scale = option_find_float(options, "noobject_scale", 1);
    layer.class_scale = option_find_float(options, "class_scale", 1);
    layer.jitter = option_find_float(options, "jitter", .2);
    layer.resize = option_find_float_quiet(options, "resize", 1.0);
    layer.random = option_find_float_quiet(options, "random", 0);
    layer.reorg = option_find_int_quiet(options, "reorg", 0);
    return layer;
}

cost_layer parse_cost(list *options, size_params params)
{
    char *type_s = option_find_str(options, "type", "sse");
    COST_TYPE type = get_cost_type(type_s);
    float scale = option_find_float_quiet(options, "scale",1);
    cost_layer layer = make_cost_layer(params.batch, params.inputs, type, scale);
    layer.ratio =  option_find_float_quiet(options, "ratio",0);
    return layer;
}

crop_layer parse_crop(list *options, size_params params)
{
    int crop_height = option_find_int(options, "crop_height",1);
    int crop_width = option_find_int(options, "crop_width",1);
    int flip = option_find_int(options, "flip",0);
    float angle = option_find_float(options, "angle",0);
    float saturation = option_find_float(options, "saturation",1);
    float exposure = option_find_float(options, "exposure",1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before crop layer must output image.", DARKNET_LOC);

    int noadjust = option_find_int_quiet(options, "noadjust",0);

    crop_layer l = make_crop_layer(batch,h,w,c,crop_height,crop_width,flip, angle, saturation, exposure);
    l.shift = option_find_float(options, "shift", 0);
    l.noadjust = noadjust;
    return l;
}

layer parse_reorg(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int reverse = option_find_int_quiet(options, "reverse",0);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before reorg layer must output image.", DARKNET_LOC);

    layer layer = make_reorg_layer(batch,w,h,c,stride,reverse);
    return layer;
}

layer parse_reorg_old(list *options, size_params params)
{
    printf("\n reorg_old \n");
    int stride = option_find_int(options, "stride", 1);
    int reverse = option_find_int_quiet(options, "reverse", 0);

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c)) error("Layer before reorg layer must output image.", DARKNET_LOC);

    layer layer = make_reorg_old_layer(batch, w, h, c, stride, reverse);
    return layer;
}

maxpool_layer parse_local_avgpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride", 1);
    int stride_x = option_find_int_quiet(options, "stride_x", stride);
    int stride_y = option_find_int_quiet(options, "stride_y", stride);
    int size = option_find_int(options, "size", stride);
    int padding = option_find_int_quiet(options, "padding", size - 1);
    int maxpool_depth = 0;
    int out_channels = 1;
    int antialiasing = 0;
    const int avgpool = 1;

    int batch, h, w, c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch = params.batch;
    if (!(h && w && c)) error("Layer before [local_avgpool] layer must output image.", DARKNET_LOC);

    maxpool_layer layer = make_maxpool_layer(batch, h, w, c, size, stride_x, stride_y, padding, maxpool_depth, out_channels, antialiasing, avgpool, params.train);
    return layer;
}

maxpool_layer parse_maxpool(list *options, size_params params)
{
    int stride = option_find_int(options, "stride",1);
    int stride_x = option_find_int_quiet(options, "stride_x", stride);
    int stride_y = option_find_int_quiet(options, "stride_y", stride);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int_quiet(options, "padding", size-1);
    int maxpool_depth = option_find_int_quiet(options, "maxpool_depth", 0);
    int out_channels = option_find_int_quiet(options, "out_channels", 1);
    int antialiasing = option_find_int_quiet(options, "antialiasing", 0);
    const int avgpool = 0;

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before [maxpool] layer must output image.", DARKNET_LOC);

    maxpool_layer layer = make_maxpool_layer(batch, h, w, c, size, stride_x, stride_y, padding, maxpool_depth, out_channels, antialiasing, avgpool, params.train);
    layer.maxpool_zero_nonmax = option_find_int_quiet(options, "maxpool_zero_nonmax", 0);
    return layer;
}

avgpool_layer parse_avgpool(list *options, size_params params)
{
    int batch,w,h,c;
    w = params.w;
    h = params.h;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before avgpool layer must output image.", DARKNET_LOC);

    avgpool_layer layer = make_avgpool_layer(batch,w,h,c);
    return layer;
}

dropout_layer parse_dropout(list *options, size_params params)
{
    float probability = option_find_float(options, "probability", .2);
    int dropblock = option_find_int_quiet(options, "dropblock", 0);
    float dropblock_size_rel = option_find_float_quiet(options, "dropblock_size_rel", 0);
    int dropblock_size_abs = option_find_float_quiet(options, "dropblock_size_abs", 0);
    if (dropblock_size_abs > params.w || dropblock_size_abs > params.h) {
        printf(" [dropout] - dropblock_size_abs = %d that is bigger than layer size %d x %d \n", dropblock_size_abs, params.w, params.h);
        dropblock_size_abs = min_val_cmp(params.w, params.h);
    }
    if (dropblock && !dropblock_size_rel && !dropblock_size_abs) {
        printf(" [dropout] - None of the parameters (dropblock_size_rel or dropblock_size_abs) are set, will be used: dropblock_size_abs = 7 \n");
        dropblock_size_abs = 7;
    }
    if (dropblock_size_rel && dropblock_size_abs) {
        printf(" [dropout] - Both parameters are set, only the parameter will be used: dropblock_size_abs = %d \n", dropblock_size_abs);
        dropblock_size_rel = 0;
    }
    dropout_layer layer = make_dropout_layer(params.batch, params.inputs, probability, dropblock, dropblock_size_rel, dropblock_size_abs, params.w, params.h, params.c);
    layer.out_w = params.w;
    layer.out_h = params.h;
    layer.out_c = params.c;
    return layer;
}

layer parse_normalization(list *options, size_params params)
{
    float alpha = option_find_float(options, "alpha", .0001);
    float beta =  option_find_float(options, "beta" , .75);
    float kappa = option_find_float(options, "kappa", 1);
    int size = option_find_int(options, "size", 5);
    layer l = make_normalization_layer(params.batch, params.w, params.h, params.c, size, alpha, beta, kappa);
    return l;
}

layer parse_batchnorm(list *options, size_params params)
{
    layer l = make_batchnorm_layer(params.batch, params.w, params.h, params.c, params.train);
    return l;
}

layer parse_shortcut(list *options, size_params params, network net)
{
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);

    char *weights_type_str = option_find_str_quiet(options, "weights_type", "none");
    WEIGHTS_TYPE_T weights_type = NO_WEIGHTS;
    if(strcmp(weights_type_str, "per_feature") == 0 || strcmp(weights_type_str, "per_layer") == 0) weights_type = PER_FEATURE;
    else if (strcmp(weights_type_str, "per_channel") == 0) weights_type = PER_CHANNEL;
    else if (strcmp(weights_type_str, "none") != 0) {
        printf("Error: Incorrect weights_type = %s \n Use one of: none, per_feature, per_channel \n", weights_type_str);
        getchar();
        exit(0);
    }

    char *weights_normalization_str = option_find_str_quiet(options, "weights_normalization", "none");
    WEIGHTS_NORMALIZATION_T weights_normalization = NO_NORMALIZATION;
    if (strcmp(weights_normalization_str, "relu") == 0 || strcmp(weights_normalization_str, "avg_relu") == 0) weights_normalization = RELU_NORMALIZATION;
    else if (strcmp(weights_normalization_str, "softmax") == 0) weights_normalization = SOFTMAX_NORMALIZATION;
    else if (strcmp(weights_type_str, "none") != 0) {
        printf("Error: Incorrect weights_normalization = %s \n Use one of: none, relu, softmax \n", weights_normalization_str);
        getchar();
        exit(0);
    }

    char *l = option_find(options, "from");
    int len = strlen(l);
    if (!l) error("Route Layer must specify input layers: from = ...", DARKNET_LOC);
    int n = 1;
    int i;
    for (i = 0; i < len; ++i) {
        if (l[i] == ',') ++n;
    }

    int* layers = (int*)calloc(n, sizeof(int));
    int* sizes = (int*)calloc(n, sizeof(int));
    float **layers_output = (float **)calloc(n, sizeof(float *));
    float **layers_delta = (float **)calloc(n, sizeof(float *));
    float **layers_output_gpu = (float **)calloc(n, sizeof(float *));
    float **layers_delta_gpu = (float **)calloc(n, sizeof(float *));

    for (i = 0; i < n; ++i) {
        int index = atoi(l);
        l = strchr(l, ',') + 1;
        if (index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = params.net.layers[index].outputs;
        layers_output[i] = params.net.layers[index].output;
        layers_delta[i] = params.net.layers[index].delta;
    }

#ifdef GPU
    for (i = 0; i < n; ++i) {
        layers_output_gpu[i] = params.net.layers[layers[i]].output_gpu;
        layers_delta_gpu[i] = params.net.layers[layers[i]].delta_gpu;
    }
#endif// GPU

    layer s = make_shortcut_layer(params.batch, n, layers, sizes, params.w, params.h, params.c, layers_output, layers_delta,
        layers_output_gpu, layers_delta_gpu, weights_type, weights_normalization, activation, params.train);

    free(layers_output_gpu);
    free(layers_delta_gpu);

    for (i = 0; i < n; ++i) {
        int index = layers[i];
        assert(params.w == net.layers[index].out_w && params.h == net.layers[index].out_h);

        if (params.w != net.layers[index].out_w || params.h != net.layers[index].out_h || params.c != net.layers[index].out_c)
            fprintf(stderr, " (%4d x%4d x%4d) + (%4d x%4d x%4d) \n",
                params.w, params.h, params.c, net.layers[index].out_w, net.layers[index].out_h, params.net.layers[index].out_c);
    }

    return s;
}


layer parse_scale_channels(list *options, size_params params, network net)
{
    char *l = option_find(options, "from");
    int index = atoi(l);
    if (index < 0) index = params.index + index;
    int scale_wh = option_find_int_quiet(options, "scale_wh", 0);

    int batch = params.batch;
    layer from = net.layers[index];

    layer s = make_scale_channels_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c, scale_wh);

    char *activation_s = option_find_str_quiet(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    s.activation = activation;
    if (activation == SWISH || activation == MISH) {
        printf(" [scale_channels] layer doesn't support SWISH or MISH activations \n");
    }
    return s;
}

layer parse_sam(list *options, size_params params, network net)
{
    char *l = option_find(options, "from");
    int index = atoi(l);
    if (index < 0) index = params.index + index;

    int batch = params.batch;
    layer from = net.layers[index];

    layer s = make_sam_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);

    char *activation_s = option_find_str_quiet(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    s.activation = activation;
    if (activation == SWISH || activation == MISH) {
        printf(" [sam] layer doesn't support SWISH or MISH activations \n");
    }
    return s;
}

layer parse_implicit(list *options, size_params params, network net)
{
    float mean_init = option_find_float(options, "mean", 0.0);
    float std_init = option_find_float(options, "std", 0.2);
    int filters = option_find_int(options, "filters", 128);
    int atoms = option_find_int_quiet(options, "atoms", 1);

    layer s = make_implicit_layer(params.batch, params.index, mean_init, std_init, filters, atoms);

    return s;
}

layer parse_activation(list *options, size_params params)
{
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);

    layer l = make_activation_layer(params.batch, params.inputs, activation);

    l.out_h = params.h;
    l.out_w = params.w;
    l.out_c = params.c;
    l.h = params.h;
    l.w = params.w;
    l.c = params.c;

    return l;
}

layer parse_upsample(list *options, size_params params, network net)
{

    int stride = option_find_int(options, "stride", 2);
    layer l = make_upsample_layer(params.batch, params.w, params.h, params.c, stride);
    l.scale = option_find_float_quiet(options, "scale", 1);
    return l;
}

route_layer parse_route(list *options, size_params params)
{
    char *l = option_find(options, "layers");
    if(!l) error("Route Layer must specify input layers", DARKNET_LOC);
    int len = strlen(l);
    int n = 1;
    int i;
    for(i = 0; i < len; ++i){
        if (l[i] == ',') ++n;
    }

    int* layers = (int*)xcalloc(n, sizeof(int));
    int* sizes = (int*)xcalloc(n, sizeof(int));
    for(i = 0; i < n; ++i){
        int index = atoi(l);
        l = strchr(l, ',')+1;
        if(index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = params.net.layers[index].outputs;
    }
    int batch = params.batch;

    int groups = option_find_int_quiet(options, "groups", 1);
    int group_id = option_find_int_quiet(options, "group_id", 0);

    route_layer layer = make_route_layer(batch, n, layers, sizes, groups, group_id);

    convolutional_layer first = params.net.layers[layers[0]];
    layer.out_w = first.out_w;
    layer.out_h = first.out_h;
    layer.out_c = first.out_c;
    for(i = 1; i < n; ++i){
        int index = layers[i];
        convolutional_layer next = params.net.layers[index];
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            layer.out_c += next.out_c;
        }else{
            fprintf(stderr, " The width and height of the input layers are different. \n");
            layer.out_h = layer.out_w = layer.out_c = 0;
        }
    }
    layer.out_c = layer.out_c / layer.groups;

    layer.w = first.w;
    layer.h = first.h;
    layer.c = layer.out_c;

    layer.stream = option_find_int_quiet(options, "stream", -1);
    layer.wait_stream_id = option_find_int_quiet(options, "wait_stream", -1);

    if (n > 3) fprintf(stderr, " \t    ");
    else if (n > 1) fprintf(stderr, " \t            ");
    else fprintf(stderr, " \t\t            ");

    fprintf(stderr, "           ");
    if (layer.groups > 1) fprintf(stderr, "%d/%d", layer.group_id, layer.groups);
    else fprintf(stderr, "   ");
    fprintf(stderr, " -> %4d x%4d x%4d \n", layer.out_w, layer.out_h, layer.out_c);

    return layer;
}

learning_rate_policy get_policy(char *s)
{
    if (strcmp(s, "random")==0) return RANDOM;
    if (strcmp(s, "poly")==0) return POLY;
    if (strcmp(s, "constant")==0) return CONSTANT;
    if (strcmp(s, "step")==0) return STEP;
    if (strcmp(s, "exp")==0) return EXP;
    if (strcmp(s, "sigmoid")==0) return SIG;
    if (strcmp(s, "steps")==0) return STEPS;
    if (strcmp(s, "sgdr")==0) return SGDR;
    fprintf(stderr, "Couldn't find policy %s, going with constant\n", s);
    return CONSTANT;
}

void parse_net_options(list *options, network *net)
{
    net->max_batches = option_find_int(options, "max_batches", 0);
    net->batch = option_find_int(options, "batch",1);
    net->learning_rate = option_find_float(options, "learning_rate", .001);
    net->learning_rate_min = option_find_float_quiet(options, "learning_rate_min", .00001);
    net->batches_per_cycle = option_find_int_quiet(options, "sgdr_cycle", net->max_batches);
    net->batches_cycle_mult = option_find_int_quiet(options, "sgdr_mult", 2);
    net->momentum = option_find_float(options, "momentum", .9);
    net->decay = option_find_float(options, "decay", .0001);
    int subdivs = option_find_int(options, "subdivisions",1);
    net->time_steps = option_find_int_quiet(options, "time_steps",1);
    net->track = option_find_int_quiet(options, "track", 0);
    net->augment_speed = option_find_int_quiet(options, "augment_speed", 2);
    net->init_sequential_subdivisions = net->sequential_subdivisions = option_find_int_quiet(options, "sequential_subdivisions", subdivs);
    if (net->sequential_subdivisions > subdivs) net->init_sequential_subdivisions = net->sequential_subdivisions = subdivs;
    net->try_fix_nan = option_find_int_quiet(options, "try_fix_nan", 0);
    net->batch /= subdivs;          // mini_batch
    const int mini_batch = net->batch;
    net->batch *= net->time_steps;  // mini_batch * time_steps
    net->subdivisions = subdivs;    // number of mini_batches

    net->weights_reject_freq = option_find_int_quiet(options, "weights_reject_freq", 0);
    net->equidistant_point = option_find_int_quiet(options, "equidistant_point", 0);
    net->badlabels_rejection_percentage = option_find_float_quiet(options, "badlabels_rejection_percentage", 0);
    net->num_sigmas_reject_badlabels = option_find_float_quiet(options, "num_sigmas_reject_badlabels", 0);
    net->ema_alpha = option_find_float_quiet(options, "ema_alpha", 0);
    *net->badlabels_reject_threshold = 0;
    *net->delta_rolling_max = 0;
    *net->delta_rolling_avg = 0;
    *net->delta_rolling_std = 0;
    *net->seen = 0;
    *net->cur_iteration = 0;
    *net->cuda_graph_ready = 0;
    net->use_cuda_graph = option_find_int_quiet(options, "use_cuda_graph", 0);
    net->loss_scale = option_find_float_quiet(options, "loss_scale", 1);
    net->dynamic_minibatch = option_find_int_quiet(options, "dynamic_minibatch", 0);
    net->optimized_memory = option_find_int_quiet(options, "optimized_memory", 0);
    net->workspace_size_limit = (size_t)1024*1024 * option_find_float_quiet(options, "workspace_size_limit_MB", 1024);  // 1024 MB by default


    net->adam = option_find_int_quiet(options, "adam", 0);
    if(net->adam){
        net->B1 = option_find_float(options, "B1", .9);
        net->B2 = option_find_float(options, "B2", .999);
        net->eps = option_find_float(options, "eps", .000001);
    }

    net->h = option_find_int_quiet(options, "height",0);
    net->w = option_find_int_quiet(options, "width",0);
    net->c = option_find_int_quiet(options, "channels",0);
    net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    net->max_crop = option_find_int_quiet(options, "max_crop",net->w*2);
    net->min_crop = option_find_int_quiet(options, "min_crop",net->w);
    net->flip = option_find_int_quiet(options, "flip", 1);
    net->blur = option_find_int_quiet(options, "blur", 0);
    net->gaussian_noise = option_find_int_quiet(options, "gaussian_noise", 0);
    net->mixup = option_find_int_quiet(options, "mixup", 0);
    int cutmix = option_find_int_quiet(options, "cutmix", 0);
    int mosaic = option_find_int_quiet(options, "mosaic", 0);
    if (mosaic && cutmix) net->mixup = 4;
    else if (cutmix) net->mixup = 2;
    else if (mosaic) net->mixup = 3;
    net->letter_box = option_find_int_quiet(options, "letter_box", 0);
    net->mosaic_bound = option_find_int_quiet(options, "mosaic_bound", 0);
    net->contrastive = option_find_int_quiet(options, "contrastive", 0);
    net->contrastive_jit_flip = option_find_int_quiet(options, "contrastive_jit_flip", 0);
    net->contrastive_color = option_find_int_quiet(options, "contrastive_color", 0);
    net->unsupervised = option_find_int_quiet(options, "unsupervised", 0);
    if (net->contrastive && mini_batch < 2) {
        printf(" Error: mini_batch size (batch/subdivisions) should be higher than 1 for Contrastive loss \n");
        exit(0);
    }
    net->label_smooth_eps = option_find_float_quiet(options, "label_smooth_eps", 0.0f);
    net->resize_step = option_find_float_quiet(options, "resize_step", 32);
    net->attention = option_find_int_quiet(options, "attention", 0);
    net->adversarial_lr = option_find_float_quiet(options, "adversarial_lr", 0);
    net->max_chart_loss = option_find_float_quiet(options, "max_chart_loss", 20.0);

    net->angle = option_find_float_quiet(options, "angle", 0);
    net->aspect = option_find_float_quiet(options, "aspect", 1);
    net->saturation = option_find_float_quiet(options, "saturation", 1);
    net->exposure = option_find_float_quiet(options, "exposure", 1);
    net->hue = option_find_float_quiet(options, "hue", 0);
    net->power = option_find_float_quiet(options, "power", 4);

    if(!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied", DARKNET_LOC);

    char *policy_s = option_find_str(options, "policy", "constant");
    net->policy = get_policy(policy_s);
    net->burn_in = option_find_int_quiet(options, "burn_in", 0);
#ifdef GPU
    if (net->gpu_index >= 0) {
        char device_name[1024];
        int compute_capability = get_gpu_compute_capability(net->gpu_index, device_name);
#ifdef CUDNN_HALF
        if (compute_capability >= 700) net->cudnn_half = 1;
        else net->cudnn_half = 0;
#endif// CUDNN_HALF
        fprintf(stderr, " %d : compute_capability = %d, cudnn_half = %d, GPU: %s \n", net->gpu_index, compute_capability, net->cudnn_half, device_name);
    }
    else fprintf(stderr, " GPU isn't used \n");
#endif// GPU
    if(net->policy == STEP){
        net->step = option_find_int(options, "step", 1);
        net->scale = option_find_float(options, "scale", 1);
    } else if (net->policy == STEPS || net->policy == SGDR){
        char *l = option_find(options, "steps");
        char *p = option_find(options, "scales");
        char *s = option_find(options, "seq_scales");
        if(net->policy == STEPS && (!l || !p)) error("STEPS policy must have steps and scales in cfg file", DARKNET_LOC);

        if (l) {
            int len = strlen(l);
            int n = 1;
            int i;
            for (i = 0; i < len; ++i) {
                if (l[i] == '#') break;
                if (l[i] == ',') ++n;
            }
            int* steps = (int*)xcalloc(n, sizeof(int));
            float* scales = (float*)xcalloc(n, sizeof(float));
            float* seq_scales = (float*)xcalloc(n, sizeof(float));
            for (i = 0; i < n; ++i) {
                float scale = 1.0;
                if (p) {
                    scale = atof(p);
                    p = strchr(p, ',') + 1;
                }
                float sequence_scale = 1.0;
                if (s) {
                    sequence_scale = atof(s);
                    s = strchr(s, ',') + 1;
                }
                int step = atoi(l);
                l = strchr(l, ',') + 1;
                steps[i] = step;
                scales[i] = scale;
                seq_scales[i] = sequence_scale;
            }
            net->scales = scales;
            net->steps = steps;
            net->seq_scales = seq_scales;
            net->num_steps = n;
        }
    } else if (net->policy == EXP){
        net->gamma = option_find_float(options, "gamma", 1);
    } else if (net->policy == SIG){
        net->gamma = option_find_float(options, "gamma", 1);
        net->step = option_find_int(options, "step", 1);
    } else if (net->policy == POLY || net->policy == RANDOM){
        //net->power = option_find_float(options, "power", 1);
    }

}

int is_network(section *s)
{
    return (strcmp(s->type, "[net]")==0
            || strcmp(s->type, "[network]")==0);
}

void set_train_only_bn(network net)
{
    int train_only_bn = 0;
    int i;
    for (i = net.n - 1; i >= 0; --i) {
        if (net.layers[i].train_only_bn) train_only_bn = net.layers[i].train_only_bn;  // set l.train_only_bn for all previous layers
        if (train_only_bn) {
            net.layers[i].train_only_bn = train_only_bn;

            if (net.layers[i].type == CONV_LSTM) {
                net.layers[i].wf->train_only_bn = train_only_bn;
                net.layers[i].wi->train_only_bn = train_only_bn;
                net.layers[i].wg->train_only_bn = train_only_bn;
                net.layers[i].wo->train_only_bn = train_only_bn;
                net.layers[i].uf->train_only_bn = train_only_bn;
                net.layers[i].ui->train_only_bn = train_only_bn;
                net.layers[i].ug->train_only_bn = train_only_bn;
                net.layers[i].uo->train_only_bn = train_only_bn;
                if (net.layers[i].peephole) {
                    net.layers[i].vf->train_only_bn = train_only_bn;
                    net.layers[i].vi->train_only_bn = train_only_bn;
                    net.layers[i].vo->train_only_bn = train_only_bn;
                }
            }
            else if (net.layers[i].type == CRNN) {
                net.layers[i].input_layer->train_only_bn = train_only_bn;
                net.layers[i].self_layer->train_only_bn = train_only_bn;
                net.layers[i].output_layer->train_only_bn = train_only_bn;
            }
        }
    }
}

network parse_network_cfg(char *filename)
{
    return parse_network_cfg_custom(filename, 0, 0);
}

network parse_network_cfg_custom(char *filename, int batch, int time_steps)
{
    list *sections = read_cfg(filename);
    node *n = sections->front;
    if(!n) error("Config file has no sections", DARKNET_LOC);
    network net = make_network(sections->size - 1);
    net.gpu_index = gpu_index;
    size_params params;

    if (batch > 0) params.train = 0;    // allocates memory for Inference only
    else params.train = 1;              // allocates memory for Inference & Training

    section *s = (section *)n->val;
    list *options = s->options;
    if(!is_network(s)) error("First section must be [net] or [network]", DARKNET_LOC);
    parse_net_options(options, &net);

#ifdef GPU
    printf("net.optimized_memory = %d \n", net.optimized_memory);
    if (net.optimized_memory >= 2 && params.train) {
        pre_allocate_pinned_memory((size_t)1024 * 1024 * 1024 * 8);   // pre-allocate 8 GB CPU-RAM for pinned memory
    }
#endif  // GPU

    params.h = net.h;
    params.w = net.w;
    params.c = net.c;
    params.inputs = net.inputs;
    if (batch > 0) net.batch = batch;
    if (time_steps > 0) net.time_steps = time_steps;
    if (net.batch < 1) net.batch = 1;
    if (net.time_steps < 1) net.time_steps = 1;
    if (net.batch < net.time_steps) net.batch = net.time_steps;
    params.batch = net.batch;
    params.time_steps = net.time_steps;
    params.net = net;
    printf("mini_batch = %d, batch = %d, time_steps = %d, train = %d \n", net.batch, net.batch * net.subdivisions, net.time_steps, params.train);

    int last_stop_backward = -1;
    int avg_outputs = 0;
    int avg_counter = 0;
    float bflops = 0;
    size_t workspace_size = 0;
    size_t max_inputs = 0;
    size_t max_outputs = 0;
    int receptive_w = 1, receptive_h = 1;
    int receptive_w_scale = 1, receptive_h_scale = 1;
    const int show_receptive_field = option_find_float_quiet(options, "show_receptive_field", 0);

    n = n->next;
    int count = 0;
    free_section(s);

    // find l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
    node *n_tmp = n;
    int count_tmp = 0;
    if (params.train == 1) {
        while (n_tmp) {
            s = (section *)n_tmp->val;
            options = s->options;
            int stopbackward = option_find_int_quiet(options, "stopbackward", 0);
            if (stopbackward == 1) {
                last_stop_backward = count_tmp;
                printf("last_stop_backward = %d \n", last_stop_backward);
            }
            n_tmp = n_tmp->next;
            ++count_tmp;
        }
    }

    int old_params_train = params.train;

    fprintf(stderr, "   layer   filters  size/strd(dil)      input                output\n");
    while(n){

        params.train = old_params_train;
        if (count < last_stop_backward) params.train = 0;

        params.index = count;
        fprintf(stderr, "%4d ", count);
        s = (section *)n->val;
        options = s->options;
        layer l = { (LAYER_TYPE)0 };
        LAYER_TYPE lt = string_to_layer_type(s->type);
        if(lt == CONVOLUTIONAL){
            l = parse_convolutional(options, params);
        }else if(lt == LOCAL){
            l = parse_local(options, params);
        }else if(lt == ACTIVE){
            l = parse_activation(options, params);
        }else if(lt == RNN){
            l = parse_rnn(options, params);
        }else if(lt == GRU){
            l = parse_gru(options, params);
        }else if(lt == LSTM){
            l = parse_lstm(options, params);
        }else if (lt == CONV_LSTM) {
            l = parse_conv_lstm(options, params);
        }else if (lt == HISTORY) {
            l = parse_history(options, params);
        }else if(lt == CRNN){
            l = parse_crnn(options, params);
        }else if(lt == CONNECTED){
            l = parse_connected(options, params);
        }else if(lt == CROP){
            l = parse_crop(options, params);
        }else if(lt == COST){
            l = parse_cost(options, params);
            l.keep_delta_gpu = 1;
        }else if(lt == REGION){
            l = parse_region(options, params);
            l.keep_delta_gpu = 1;
        }else if (lt == YOLO) {
            l = parse_yolo(options, params);
            l.keep_delta_gpu = 1;
        }else if (lt == GAUSSIAN_YOLO) {
            l = parse_gaussian_yolo(options, params);
            l.keep_delta_gpu = 1;
        }else if(lt == DETECTION){
            l = parse_detection(options, params);
        }else if(lt == SOFTMAX){
            l = parse_softmax(options, params);
            net.hierarchy = l.softmax_tree;
            l.keep_delta_gpu = 1;
        }else if (lt == CONTRASTIVE) {
            l = parse_contrastive(options, params);
            l.keep_delta_gpu = 1;
        }else if(lt == NORMALIZATION){
            l = parse_normalization(options, params);
        }else if(lt == BATCHNORM){
            l = parse_batchnorm(options, params);
        }else if(lt == MAXPOOL){
            l = parse_maxpool(options, params);
        }else if (lt == LOCAL_AVGPOOL) {
            l = parse_local_avgpool(options, params);
        }else if(lt == REORG){
            l = parse_reorg(options, params);        }
        else if (lt == REORG_OLD) {
            l = parse_reorg_old(options, params);
        }else if(lt == AVGPOOL){
            l = parse_avgpool(options, params);
        }else if(lt == ROUTE){
            l = parse_route(options, params);
            int k;
            for (k = 0; k < l.n; ++k) {
                net.layers[l.input_layers[k]].use_bin_output = 0;
                if (count >= last_stop_backward)
                    net.layers[l.input_layers[k]].keep_delta_gpu = 1;
            }
        }else if (lt == UPSAMPLE) {
            l = parse_upsample(options, params, net);
        }else if(lt == SHORTCUT){
            l = parse_shortcut(options, params, net);
            net.layers[count - 1].use_bin_output = 0;
            net.layers[l.index].use_bin_output = 0;
            if (count >= last_stop_backward)
                net.layers[l.index].keep_delta_gpu = 1;
        }else if (lt == SCALE_CHANNELS) {
            l = parse_scale_channels(options, params, net);
            net.layers[count - 1].use_bin_output = 0;
            net.layers[l.index].use_bin_output = 0;
            net.layers[l.index].keep_delta_gpu = 1;
        }
        else if (lt == SAM) {
            l = parse_sam(options, params, net);
            net.layers[count - 1].use_bin_output = 0;
            net.layers[l.index].use_bin_output = 0;
            net.layers[l.index].keep_delta_gpu = 1;
        } else if (lt == IMPLICIT) {
            l = parse_implicit(options, params, net);
        }else if(lt == DROPOUT){
            l = parse_dropout(options, params);
            l.output = net.layers[count-1].output;
            l.delta = net.layers[count-1].delta;
#ifdef GPU
            l.output_gpu = net.layers[count-1].output_gpu;
            l.delta_gpu = net.layers[count-1].delta_gpu;
            l.keep_delta_gpu = 1;
#endif
        }
        else if (lt == EMPTY) {
            layer empty_layer = {(LAYER_TYPE)0};
            l = empty_layer;
            l.type = EMPTY;
            l.w = l.out_w = params.w;
            l.h = l.out_h = params.h;
            l.c = l.out_c = params.c;
            l.batch = params.batch;
            l.inputs = l.outputs = params.inputs;
            l.output = net.layers[count - 1].output;
            l.delta = net.layers[count - 1].delta;
            l.forward = empty_func;
            l.backward = empty_func;
#ifdef GPU
            l.output_gpu = net.layers[count - 1].output_gpu;
            l.delta_gpu = net.layers[count - 1].delta_gpu;
            l.keep_delta_gpu = 1;
            l.forward_gpu = empty_func;
            l.backward_gpu = empty_func;
#endif
            fprintf(stderr, "empty \n");
        }else{
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }

        // calculate receptive field
        if(show_receptive_field)
        {
            int dilation = max_val_cmp(1, l.dilation);
            int stride = max_val_cmp(1, l.stride);
            int size = max_val_cmp(1, l.size);

            if (l.type == UPSAMPLE || (l.type == REORG))
            {

                l.receptive_w = receptive_w;
                l.receptive_h = receptive_h;
                l.receptive_w_scale = receptive_w_scale = receptive_w_scale / stride;
                l.receptive_h_scale = receptive_h_scale = receptive_h_scale / stride;

            }
            else {
                if (l.type == ROUTE) {
                    receptive_w = receptive_h = receptive_w_scale = receptive_h_scale = 0;
                    int k;
                    for (k = 0; k < l.n; ++k) {
                        layer route_l = net.layers[l.input_layers[k]];
                        receptive_w = max_val_cmp(receptive_w, route_l.receptive_w);
                        receptive_h = max_val_cmp(receptive_h, route_l.receptive_h);
                        receptive_w_scale = max_val_cmp(receptive_w_scale, route_l.receptive_w_scale);
                        receptive_h_scale = max_val_cmp(receptive_h_scale, route_l.receptive_h_scale);
                    }
                }
                else
                {
                    int increase_receptive = size + (dilation - 1) * 2 - 1;// stride;
                    increase_receptive = max_val_cmp(0, increase_receptive);

                    receptive_w += increase_receptive * receptive_w_scale;
                    receptive_h += increase_receptive * receptive_h_scale;
                    receptive_w_scale *= stride;
                    receptive_h_scale *= stride;
                }

                l.receptive_w = receptive_w;
                l.receptive_h = receptive_h;
                l.receptive_w_scale = receptive_w_scale;
                l.receptive_h_scale = receptive_h_scale;
            }
            //printf(" size = %d, dilation = %d, stride = %d, receptive_w = %d, receptive_w_scale = %d - ", size, dilation, stride, receptive_w, receptive_w_scale);

            int cur_receptive_w = receptive_w;
            int cur_receptive_h = receptive_h;

            fprintf(stderr, "%4d - receptive field: %d x %d \n", count, cur_receptive_w, cur_receptive_h);
        }

#ifdef GPU
        // futher GPU-memory optimization: net.optimized_memory == 2
        l.optimized_memory = net.optimized_memory;
        if (net.optimized_memory == 1 && params.train && l.type != DROPOUT) {
            if (l.delta_gpu) {
                cuda_free(l.delta_gpu);
                l.delta_gpu = NULL;
            }
        } else if (net.optimized_memory >= 2 && params.train && l.type != DROPOUT)
        {
            if (l.output_gpu) {
                cuda_free(l.output_gpu);
                //l.output_gpu = cuda_make_array_pinned(l.output, l.batch*l.outputs); // l.steps
                l.output_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
            }
            if (l.activation_input_gpu) {
                cuda_free(l.activation_input_gpu);
                l.activation_input_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
            }

            if (l.x_gpu) {
                cuda_free(l.x_gpu);
                l.x_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
            }

            // maximum optimization
            if (net.optimized_memory >= 3 && l.type != DROPOUT) {
                if (l.delta_gpu) {
                    cuda_free(l.delta_gpu);
                    //l.delta_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
                    //printf("\n\n PINNED DELTA GPU = %d \n", l.batch*l.outputs);
                }
            }

            if (l.type == CONVOLUTIONAL) {
                set_specified_workspace_limit(&l, net.workspace_size_limit);   // workspace size limit 1 GB
            }
        }
#endif // GPU

        l.clip = option_find_float_quiet(options, "clip", 0);
        l.dynamic_minibatch = net.dynamic_minibatch;
        l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
        l.dont_update = option_find_int_quiet(options, "dont_update", 0);
        l.burnin_update = option_find_int_quiet(options, "burnin_update", 0);
        l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
        l.train_only_bn = option_find_int_quiet(options, "train_only_bn", 0);
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        option_unused(options);

        if (l.stopbackward == 1) printf(" ------- previous layers are frozen ------- \n");

        net.layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        if (l.inputs > max_inputs) max_inputs = l.inputs;
        if (l.outputs > max_outputs) max_outputs = l.outputs;
        free_section(s);
        n = n->next;
        ++count;
        if(n){
            if (l.antialiasing) {
                params.h = l.input_layer->out_h;
                params.w = l.input_layer->out_w;
                params.c = l.input_layer->out_c;
                params.inputs = l.input_layer->outputs;
            }
            else {
                params.h = l.out_h;
                params.w = l.out_w;
                params.c = l.out_c;
                params.inputs = l.outputs;
            }
        }
        if (l.bflops > 0) bflops += l.bflops;

        if (l.w > 1 && l.h > 1) {
            avg_outputs += l.outputs;
            avg_counter++;
        }
    }

    if (last_stop_backward > -1) {
        int k;
        for (k = 0; k < last_stop_backward; ++k) {
            layer l = net.layers[k];
            if (l.keep_delta_gpu) {
                if (!l.delta) {
                    net.layers[k].delta = (float*)xcalloc(l.outputs*l.batch, sizeof(float));
                }
#ifdef GPU
                if (!l.delta_gpu) {
                    net.layers[k].delta_gpu = (float *)cuda_make_array(NULL, l.outputs*l.batch);
                }
#endif
            }

            net.layers[k].onlyforward = 1;
            net.layers[k].train = 0;
        }
    }

    free_list(sections);

#ifdef GPU
    if (net.optimized_memory && params.train)
    {
        int k;
        for (k = 0; k < net.n; ++k) {
            layer l = net.layers[k];
            // delta GPU-memory optimization: net.optimized_memory == 1
            if (!l.keep_delta_gpu) {
                const size_t delta_size = l.outputs*l.batch; // l.steps
                if (net.max_delta_gpu_size < delta_size) {
                    net.max_delta_gpu_size = delta_size;
                    if (net.global_delta_gpu) cuda_free(net.global_delta_gpu);
                    if (net.state_delta_gpu) cuda_free(net.state_delta_gpu);
                    assert(net.max_delta_gpu_size > 0);
                    net.global_delta_gpu = (float *)cuda_make_array(NULL, net.max_delta_gpu_size);
                    net.state_delta_gpu = (float *)cuda_make_array(NULL, net.max_delta_gpu_size);
                }
                if (l.delta_gpu) {
                    if (net.optimized_memory >= 3) {}
                    else cuda_free(l.delta_gpu);
                }
                l.delta_gpu = net.global_delta_gpu;
            }
            else {
                if (!l.delta_gpu) l.delta_gpu = (float *)cuda_make_array(NULL, l.outputs*l.batch);
            }

            // maximum optimization
            if (net.optimized_memory >= 3 && l.type != DROPOUT) {
                if (l.delta_gpu && l.keep_delta_gpu) {
                    //cuda_free(l.delta_gpu);   // already called above
                    l.delta_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
                    //printf("\n\n PINNED DELTA GPU = %d \n", l.batch*l.outputs);
                }
            }

            net.layers[k] = l;
        }
    }
#endif

    set_train_only_bn(net); // set l.train_only_bn for all required layers

    net.outputs = get_network_output_size(net);
    net.output = get_network_output(net);
    avg_outputs = avg_outputs / avg_counter;
    fprintf(stderr, "Total BFLOPS %5.3f \n", bflops);
    fprintf(stderr, "avg_outputs = %d \n", avg_outputs);
#ifdef GPU
    get_cuda_stream();
    //get_cuda_memcpy_stream();
    if (gpu_index >= 0)
    {
        int size = get_network_input_size(net) * net.batch;
        net.input_state_gpu = cuda_make_array(0, size);
        if (cudaSuccess == cudaHostAlloc(&net.input_pinned_cpu, size * sizeof(float), cudaHostRegisterMapped)) net.input_pinned_cpu_flag = 1;
        else {
            cudaGetLastError(); // reset CUDA-error
            net.input_pinned_cpu = (float*)xcalloc(size, sizeof(float));
        }

        // pre-allocate memory for inference on Tensor Cores (fp16)
        *net.max_input16_size = 0;
        *net.max_output16_size = 0;
        if (net.cudnn_half) {
            *net.max_input16_size = max_inputs;
            CHECK_CUDA(cudaMalloc((void **)net.input16_gpu, *net.max_input16_size * sizeof(short))); //sizeof(half)
            *net.max_output16_size = max_outputs;
            CHECK_CUDA(cudaMalloc((void **)net.output16_gpu, *net.max_output16_size * sizeof(short))); //sizeof(half)
        }
        if (workspace_size) {
            fprintf(stderr, " Allocate additional workspace_size = %1.2f MB \n", (float)workspace_size/1000000);
            net.workspace = cuda_make_array(0, workspace_size / sizeof(float) + 1);
        }
        else {
            net.workspace = (float*)xcalloc(1, workspace_size);
        }
    }
#else
        if (workspace_size) {
            net.workspace = (float*)xcalloc(1, workspace_size);
        }
#endif

    LAYER_TYPE lt = net.layers[net.n - 1].type;
    if ((net.w % 32 != 0 || net.h % 32 != 0) && (lt == YOLO || lt == REGION || lt == DETECTION)) {
        printf("\n Warning: width=%d and height=%d in cfg-file must be divisible by 32 for default networks Yolo v1/v2/v3!!! \n\n",
            net.w, net.h);
    }
    return net;
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
                current = (section*)xmalloc(sizeof(section));
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

void save_convolutional_weights_binary(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    int size = (l.c/l.groups)*l.size*l.size;
    binarize_weights(l.weights, l.n, size, l.binary_weights);
    int i, j, k;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    for(i = 0; i < l.n; ++i){
        float mean = l.binary_weights[i*size];
        if(mean < 0) mean = -mean;
        fwrite(&mean, sizeof(float), 1, fp);
        for(j = 0; j < size/8; ++j){
            int index = i*size + j*8;
            unsigned char c = 0;
            for(k = 0; k < 8; ++k){
                if (j*8 + k >= size) break;
                if (l.binary_weights[index + k] > 0) c = (c | 1<<k);
            }
            fwrite(&c, sizeof(char), 1, fp);
        }
    }
}

void save_shortcut_weights(layer l, FILE *fp)
{
#ifdef GPU
    if (gpu_index >= 0) {
        pull_shortcut_layer(l);
        printf("\n pull_shortcut_layer \n");
    }
#endif
    int i;
    //if(l.weight_updates) for (i = 0; i < l.nweights; ++i) printf(" %f, ", l.weight_updates[i]);
    //printf(" l.nweights = %d - update \n", l.nweights);
    for (i = 0; i < l.nweights; ++i) printf(" %f, ", l.weights[i]);
    printf(" l.nweights = %d \n\n", l.nweights);

    int num = l.nweights;
    fwrite(l.weights, sizeof(float), num, fp);
}

void save_implicit_weights(layer l, FILE *fp)
{
#ifdef GPU
    if (gpu_index >= 0) {
        pull_implicit_layer(l);
        //printf("\n pull_implicit_layer \n");
    }
#endif
    int i;
    //if(l.weight_updates) for (i = 0; i < l.nweights; ++i) printf(" %f, ", l.weight_updates[i]);
    //printf(" l.nweights = %d - update \n", l.nweights);
    //for (i = 0; i < l.nweights; ++i) printf(" %f, ", l.weights[i]);
    //printf(" l.nweights = %d \n\n", l.nweights);

    int num = l.nweights;
    fwrite(l.weights, sizeof(float), num, fp);
}

void save_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //save_convolutional_weights_binary(l, fp);
        //return;
    }
#ifdef GPU
    if(gpu_index >= 0){
        pull_convolutional_layer(l);
    }
#endif
    int num = l.nweights;
    fwrite(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights, sizeof(float), num, fp);
    //if(l.adam){
    //    fwrite(l.m, sizeof(float), num, fp);
    //    fwrite(l.v, sizeof(float), num, fp);
    //}
}

void save_convolutional_weights_ema(layer l, FILE *fp)
{
    if (l.binary) {
        //save_convolutional_weights_binary(l, fp);
        //return;
    }
#ifdef GPU
    if (gpu_index >= 0) {
        pull_convolutional_layer(l);
    }
#endif
    int num = l.nweights;
    fwrite(l.biases_ema, sizeof(float), l.n, fp);
    if (l.batch_normalize) {
        fwrite(l.scales_ema, sizeof(float), l.n, fp);
        fwrite(l.rolling_mean, sizeof(float), l.n, fp);
        fwrite(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fwrite(l.weights_ema, sizeof(float), num, fp);
    //if(l.adam){
    //    fwrite(l.m, sizeof(float), num, fp);
    //    fwrite(l.v, sizeof(float), num, fp);
    //}
}

void save_batchnorm_weights(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_batchnorm_layer(l);
    }
#endif
    fwrite(l.biases, sizeof(float), l.c, fp);
    fwrite(l.scales, sizeof(float), l.c, fp);
    fwrite(l.rolling_mean, sizeof(float), l.c, fp);
    fwrite(l.rolling_variance, sizeof(float), l.c, fp);
}

void save_connected_weights(layer l, FILE *fp)
{
#ifdef GPU
    if(gpu_index >= 0){
        pull_connected_layer(l);
    }
#endif
    fwrite(l.biases, sizeof(float), l.outputs, fp);
    fwrite(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if (l.batch_normalize){
        fwrite(l.scales, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
        fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
    }
}

void save_weights_upto(network net, char *filename, int cutoff, int save_ema)
{
#ifdef GPU
    if(net.gpu_index >= 0){
        cuda_set_device(net.gpu_index);
    }
#endif
    fprintf(stderr, "Saving weights to %s\n", filename);
    FILE *fp = fopen(filename, "wb");
    if(!fp) file_error(filename);

    int major = MAJOR_VERSION;
    int minor = MINOR_VERSION;
    int revision = PATCH_VERSION;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    (*net.seen) = get_current_iteration(net) * net.batch * net.subdivisions; // remove this line, when you will save to weights-file both: seen & cur_iteration
    fwrite(net.seen, sizeof(uint64_t), 1, fp);

    int i;
    for(i = 0; i < net.n && i < cutoff; ++i){
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL && l.share_layer == NULL) {
            if (save_ema) {
                save_convolutional_weights_ema(l, fp);
            }
            else {
                save_convolutional_weights(l, fp);
            }
        } if (l.type == SHORTCUT && l.nweights > 0) {
            save_shortcut_weights(l, fp);
        } if (l.type == IMPLICIT) {
            save_implicit_weights(l, fp);
        } if(l.type == CONNECTED){
            save_connected_weights(l, fp);
        } if(l.type == BATCHNORM){
            save_batchnorm_weights(l, fp);
        } if(l.type == RNN){
            save_connected_weights(*(l.input_layer), fp);
            save_connected_weights(*(l.self_layer), fp);
            save_connected_weights(*(l.output_layer), fp);
        } if(l.type == GRU){
            save_connected_weights(*(l.input_z_layer), fp);
            save_connected_weights(*(l.input_r_layer), fp);
            save_connected_weights(*(l.input_h_layer), fp);
            save_connected_weights(*(l.state_z_layer), fp);
            save_connected_weights(*(l.state_r_layer), fp);
            save_connected_weights(*(l.state_h_layer), fp);
        } if(l.type == LSTM){
            save_connected_weights(*(l.wf), fp);
            save_connected_weights(*(l.wi), fp);
            save_connected_weights(*(l.wg), fp);
            save_connected_weights(*(l.wo), fp);
            save_connected_weights(*(l.uf), fp);
            save_connected_weights(*(l.ui), fp);
            save_connected_weights(*(l.ug), fp);
            save_connected_weights(*(l.uo), fp);
        } if (l.type == CONV_LSTM) {
            if (l.peephole) {
                save_convolutional_weights(*(l.vf), fp);
                save_convolutional_weights(*(l.vi), fp);
                save_convolutional_weights(*(l.vo), fp);
            }
            save_convolutional_weights(*(l.wf), fp);
            if (!l.bottleneck) {
                save_convolutional_weights(*(l.wi), fp);
                save_convolutional_weights(*(l.wg), fp);
                save_convolutional_weights(*(l.wo), fp);
            }
            save_convolutional_weights(*(l.uf), fp);
            save_convolutional_weights(*(l.ui), fp);
            save_convolutional_weights(*(l.ug), fp);
            save_convolutional_weights(*(l.uo), fp);
        } if(l.type == CRNN){
            save_convolutional_weights(*(l.input_layer), fp);
            save_convolutional_weights(*(l.self_layer), fp);
            save_convolutional_weights(*(l.output_layer), fp);
        } if(l.type == LOCAL){
#ifdef GPU
            if(gpu_index >= 0){
                pull_local_layer(l);
            }
#endif
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fwrite(l.biases, sizeof(float), l.outputs, fp);
            fwrite(l.weights, sizeof(float), size, fp);
        }
        fflush(fp);
    }
    fclose(fp);
}
void save_weights(network net, char *filename)
{
    save_weights_upto(net, filename, net.n, 0);
}

void transpose_matrix(float *a, int rows, int cols)
{
    float* transpose = (float*)xcalloc(rows * cols, sizeof(float));
    int x, y;
    for(x = 0; x < rows; ++x){
        for(y = 0; y < cols; ++y){
            transpose[y*rows + x] = a[x*cols + y];
        }
    }
    memcpy(a, transpose, rows*cols*sizeof(float));
    free(transpose);
}

void load_connected_weights(layer l, FILE *fp, int transpose)
{
    fread(l.biases, sizeof(float), l.outputs, fp);
    fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
    if(transpose){
        transpose_matrix(l.weights, l.inputs, l.outputs);
    }
    //printf("Biases: %f mean %f variance\n", mean_array(l.biases, l.outputs), variance_array(l.biases, l.outputs));
    //printf("Weights: %f mean %f variance\n", mean_array(l.weights, l.outputs*l.inputs), variance_array(l.weights, l.outputs*l.inputs));
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.outputs, fp);
        fread(l.rolling_mean, sizeof(float), l.outputs, fp);
        fread(l.rolling_variance, sizeof(float), l.outputs, fp);
        //printf("Scales: %f mean %f variance\n", mean_array(l.scales, l.outputs), variance_array(l.scales, l.outputs));
        //printf("rolling_mean: %f mean %f variance\n", mean_array(l.rolling_mean, l.outputs), variance_array(l.rolling_mean, l.outputs));
        //printf("rolling_variance: %f mean %f variance\n", mean_array(l.rolling_variance, l.outputs), variance_array(l.rolling_variance, l.outputs));
    }
#ifdef GPU
    if(gpu_index >= 0){
        push_connected_layer(l);
    }
#endif
}

void load_batchnorm_weights(layer l, FILE *fp)
{
    fread(l.biases, sizeof(float), l.c, fp);
    fread(l.scales, sizeof(float), l.c, fp);
    fread(l.rolling_mean, sizeof(float), l.c, fp);
    fread(l.rolling_variance, sizeof(float), l.c, fp);
#ifdef GPU
    if(gpu_index >= 0){
        push_batchnorm_layer(l);
    }
#endif
}

void load_convolutional_weights_binary(layer l, FILE *fp)
{
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    int size = (l.c / l.groups)*l.size*l.size;
    int i, j, k;
    for(i = 0; i < l.n; ++i){
        float mean = 0;
        fread(&mean, sizeof(float), 1, fp);
        for(j = 0; j < size/8; ++j){
            int index = i*size + j*8;
            unsigned char c = 0;
            fread(&c, sizeof(char), 1, fp);
            for(k = 0; k < 8; ++k){
                if (j*8 + k >= size) break;
                l.weights[index + k] = (c & 1<<k) ? mean : -mean;
            }
        }
    }
#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif
}

void load_convolutional_weights(layer l, FILE *fp)
{
    if(l.binary){
        //load_convolutional_weights_binary(l, fp);
        //return;
    }
    int num = l.nweights;
    int read_bytes;
    read_bytes = fread(l.biases, sizeof(float), l.n, fp);
    if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.biases - l.index = %d \n", l.index);
    //fread(l.weights, sizeof(float), num, fp); // as in connected layer
    if (l.batch_normalize && (!l.dontloadscales)){
        read_bytes = fread(l.scales, sizeof(float), l.n, fp);
        if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.scales - l.index = %d \n", l.index);
        read_bytes = fread(l.rolling_mean, sizeof(float), l.n, fp);
        if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.rolling_mean - l.index = %d \n", l.index);
        read_bytes = fread(l.rolling_variance, sizeof(float), l.n, fp);
        if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.rolling_variance - l.index = %d \n", l.index);
        if(0){
            int i;
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_mean[i]);
            }
            printf("\n");
            for(i = 0; i < l.n; ++i){
                printf("%g, ", l.rolling_variance[i]);
            }
            printf("\n");
        }
        if(0){
            fill_cpu(l.n, 0, l.rolling_mean, 1);
            fill_cpu(l.n, 0, l.rolling_variance, 1);
        }
    }
    read_bytes = fread(l.weights, sizeof(float), num, fp);
    if (read_bytes > 0 && read_bytes < l.n) printf("\n Warning: Unexpected end of wights-file! l.weights - l.index = %d \n", l.index);
    //if(l.adam){
    //    fread(l.m, sizeof(float), num, fp);
    //    fread(l.v, sizeof(float), num, fp);
    //}
    //if(l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
    if (l.flipped) {
        transpose_matrix(l.weights, (l.c/l.groups)*l.size*l.size, l.n);
    }
    //if (l.binary) binarize_weights(l.weights, l.n, (l.c/l.groups)*l.size*l.size, l.weights);
#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
#endif
}

void load_shortcut_weights(layer l, FILE *fp)
{
    int num = l.nweights;
    int read_bytes;
    read_bytes = fread(l.weights, sizeof(float), num, fp);
    if (read_bytes > 0 && read_bytes < num) printf("\n Warning: Unexpected end of wights-file! l.weights - l.index = %d \n", l.index);
    //for (int i = 0; i < l.nweights; ++i) printf(" %f, ", l.weights[i]);
    //printf(" read_bytes = %d \n\n", read_bytes);
#ifdef GPU
    if (gpu_index >= 0) {
        push_shortcut_layer(l);
    }
#endif
}

void load_implicit_weights(layer l, FILE *fp)
{
    int num = l.nweights;
    int read_bytes;
    read_bytes = fread(l.weights, sizeof(float), num, fp);
    if (read_bytes > 0 && read_bytes < num) printf("\n Warning: Unexpected end of wights-file! l.weights - l.index = %d \n", l.index);
    //for (int i = 0; i < l.nweights; ++i) printf(" %f, ", l.weights[i]);
    //printf(" read_bytes = %d \n\n", read_bytes);
#ifdef GPU
    if (gpu_index >= 0) {
        push_implicit_layer(l);
    }
#endif
}

void load_weights_upto(network *net, char *filename, int cutoff)
{
#ifdef GPU
    if(net->gpu_index >= 0){
        cuda_set_device(net->gpu_index);
    }
#endif
    fprintf(stderr, "Loading weights from %s...", filename);
    fflush(stdout);
    FILE *fp = fopen(filename, "rb");
    if(!fp) file_error(filename);

    int major;
    int minor;
    int revision;
    fread(&major, sizeof(int), 1, fp);
    fread(&minor, sizeof(int), 1, fp);
    fread(&revision, sizeof(int), 1, fp);
    if ((major * 10 + minor) >= 2) {
        printf("\n seen 64");
        uint64_t iseen = 0;
        fread(&iseen, sizeof(uint64_t), 1, fp);
        *net->seen = iseen;
    }
    else {
        printf("\n seen 32");
        uint32_t iseen = 0;
        fread(&iseen, sizeof(uint32_t), 1, fp);
        *net->seen = iseen;
    }
    *net->cur_iteration = get_current_batch(*net);
    printf(", trained: %.0f K-images (%.0f Kilo-batches_64) \n", (float)(*net->seen / 1000), (float)(*net->seen / 64000));
    int transpose = (major > 1000) || (minor > 1000);

    int i;
    for(i = 0; i < net->n && i < cutoff; ++i){
        layer l = net->layers[i];
        if (l.dontload) continue;
        if(l.type == CONVOLUTIONAL && l.share_layer == NULL){
            load_convolutional_weights(l, fp);
        }
        if (l.type == SHORTCUT && l.nweights > 0) {
            load_shortcut_weights(l, fp);
        }
        if (l.type == IMPLICIT) {
            load_implicit_weights(l, fp);
        }
        if(l.type == CONNECTED){
            load_connected_weights(l, fp, transpose);
        }
        if(l.type == BATCHNORM){
            load_batchnorm_weights(l, fp);
        }
        if(l.type == CRNN){
            load_convolutional_weights(*(l.input_layer), fp);
            load_convolutional_weights(*(l.self_layer), fp);
            load_convolutional_weights(*(l.output_layer), fp);
        }
        if(l.type == RNN){
            load_connected_weights(*(l.input_layer), fp, transpose);
            load_connected_weights(*(l.self_layer), fp, transpose);
            load_connected_weights(*(l.output_layer), fp, transpose);
        }
        if(l.type == GRU){
            load_connected_weights(*(l.input_z_layer), fp, transpose);
            load_connected_weights(*(l.input_r_layer), fp, transpose);
            load_connected_weights(*(l.input_h_layer), fp, transpose);
            load_connected_weights(*(l.state_z_layer), fp, transpose);
            load_connected_weights(*(l.state_r_layer), fp, transpose);
            load_connected_weights(*(l.state_h_layer), fp, transpose);
        }
        if(l.type == LSTM){
            load_connected_weights(*(l.wf), fp, transpose);
            load_connected_weights(*(l.wi), fp, transpose);
            load_connected_weights(*(l.wg), fp, transpose);
            load_connected_weights(*(l.wo), fp, transpose);
            load_connected_weights(*(l.uf), fp, transpose);
            load_connected_weights(*(l.ui), fp, transpose);
            load_connected_weights(*(l.ug), fp, transpose);
            load_connected_weights(*(l.uo), fp, transpose);
        }
        if (l.type == CONV_LSTM) {
            if (l.peephole) {
                load_convolutional_weights(*(l.vf), fp);
                load_convolutional_weights(*(l.vi), fp);
                load_convolutional_weights(*(l.vo), fp);
            }
            load_convolutional_weights(*(l.wf), fp);
            if (!l.bottleneck) {
                load_convolutional_weights(*(l.wi), fp);
                load_convolutional_weights(*(l.wg), fp);
                load_convolutional_weights(*(l.wo), fp);
            }
            load_convolutional_weights(*(l.uf), fp);
            load_convolutional_weights(*(l.ui), fp);
            load_convolutional_weights(*(l.ug), fp);
            load_convolutional_weights(*(l.uo), fp);
        }
        if(l.type == LOCAL){
            int locations = l.out_w*l.out_h;
            int size = l.size*l.size*l.c*l.n*locations;
            fread(l.biases, sizeof(float), l.outputs, fp);
            fread(l.weights, sizeof(float), size, fp);
#ifdef GPU
            if(gpu_index >= 0){
                push_local_layer(l);
            }
#endif
        }
        if (feof(fp)) break;
    }
    fprintf(stderr, "Done! Loaded %d layers from weights-file \n", i);
    fclose(fp);
}

void load_weights(network *net, char *filename)
{
    load_weights_upto(net, filename, net->n);
}

// load network & force - set batch size
network *load_network_custom(char *cfg, char *weights, int clear, int batch)
{
    printf(" Try to load cfg: %s, weights: %s, clear = %d \n", cfg, weights, clear);
    network* net = (network*)xcalloc(1, sizeof(network));
    *net = parse_network_cfg_custom(cfg, batch, 1);
    if (weights && weights[0] != 0) {
        printf(" Try to load weights: %s \n", weights);
        load_weights(net, weights);
    }
    fuse_conv_batchnorm(*net);
    if (clear) {
        (*net->seen) = 0;
        (*net->cur_iteration) = 0;
    }
    return net;
}

// load network & get batch size from cfg-file
network *load_network(char *cfg, char *weights, int clear)
{
    printf(" Try to load cfg: %s, clear = %d \n", cfg, clear);
    network* net = (network*)xcalloc(1, sizeof(network));
    *net = parse_network_cfg(cfg);
    if (weights && weights[0] != 0) {
        printf(" Try to load weights: %s \n", weights);
        load_weights(net, weights);
    }
    if (clear) {
        (*net->seen) = 0;
        (*net->cur_iteration) = 0;
    }
    return net;
}
