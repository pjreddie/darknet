#include "xnor_layer.h"
#include "binary_convolution.h"
#include "convolutional_layer.h"

layer make_xnor_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation, int batch_normalize)
{
    int i;
    layer l = {0};
    l.type = XNOR;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = pad;
    l.batch_normalize = batch_normalize;

    l.filters = calloc(c*n*size*size, sizeof(float));
    l.biases = calloc(n, sizeof(float));

    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = calloc(l.batch*out_h * out_w * n, sizeof(float));

    if(batch_normalize){
        l.scales = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
    }

    l.activation = activation;

    fprintf(stderr, "XNOR Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);

    return l;
}

void forward_xnor_layer(const layer l, network_state state)
{
    int b = l.n;
    int c = l.c;
    int ix = l.w;
    int iy = l.h;
    int wx = l.size;
    int wy = l.size;
    int s = l.stride;
    int pad = l.pad * (l.size/2);

    // MANDATORY: Make the binary layer
    ai2_bin_conv_layer al = ai2_make_bin_conv_layer(b, c, ix, iy, wx, wy, s, pad);

    // OPTIONAL: You need to set the real-valued input like:
    ai2_setFltInput_unpadded(&al, state.input);
    // The above function will automatically binarize the input for the layer (channel wise).
    // If commented: using the default 0-valued input.

    ai2_setFltWeights(&al, l.filters);
    // The above function will automatically binarize the input for the layer (channel wise).
    // If commented: using the default 0-valued weights.

    // MANDATORY: Call forward
    ai2_bin_forward(&al);

    // OPTIONAL: Inspect outputs
    float *output = ai2_getFltOutput(&al);  // output is of size l.px * l.py where px and py are the padded outputs

    memcpy(l.output, output, l.outputs*sizeof(float));
    // MANDATORY: Free layer
    ai2_free_bin_conv_layer(&al);
}
