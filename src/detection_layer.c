int detection_out_height(detection_layer layer)
{
    return layer.size + layer.h*layer.stride;
}

int detection_out_width(detection_layer layer)
{
    return layer.size + layer.w*layer.stride;
}

detection_layer *make_detection_layer(int batch, int h, int w, int c, int n, int size, int stride, ACTIVATION activation)
{
    int i;
    size = 2*(size/2)+1; //HA! And you thought you'd use an even sized filter...
    detection_layer *layer = calloc(1, sizeof(detection_layer));
    layer->h = h;
    layer->w = w;
    layer->c = c;
    layer->n = n;
    layer->batch = batch;
    layer->stride = stride;
    layer->size = size;
    assert(c%n == 0);

    layer->filters = calloc(c*size*size, sizeof(float));
    layer->filter_updates = calloc(c*size*size, sizeof(float));
    layer->filter_momentum = calloc(c*size*size, sizeof(float));

    float scale = 1./(size*size*c);
    for(i = 0; i < c*n*size*size; ++i) layer->filters[i] = scale*(rand_uniform());

    int out_h = detection_out_height(*layer);
    int out_w = detection_out_width(*layer);

    layer->output = calloc(layer->batch * out_h * out_w * n, sizeof(float));
    layer->delta  = calloc(layer->batch * out_h * out_w * n, sizeof(float));

    layer->activation = activation;

    fprintf(stderr, "Convolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);
    srand(0);

    return layer;
}

void forward_detection_layer(const detection_layer layer, float *in)
{
    int out_h = detection_out_height(layer);
    int out_w = detection_out_width(layer);
    int i,j,fh, fw,c;
    memset(layer.output, 0, layer->batch*layer->n*out_h*out_w*sizeof(float));
    for(c = 0; c < layer.c; ++c){
        for(i = 0; i < layer.h; ++i){
            for(j = 0; j < layer.w; ++j){
                float val = layer->input[j+(i + c*layer.h)*layer.w];
                for(fh = 0; fh < layer.size; ++fh){
                    for(fw = 0; fw < layer.size; ++fw){
                        int h = i*layer.stride + fh;
                        int w = j*layer.stride + fw;
                        layer.output[w+(h+c/n*out_h)*out_w] += val*layer->filters[fw+(fh+c*layer.size)*layer.size];
                    }
                }
            }
        }
    }
}

void backward_detection_layer(const detection_layer layer, float *delta)
{
}


