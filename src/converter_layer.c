#include "converter_layer.h"
#include "converter.h"

converter_layer make_converter_layer(int batch, int w, int h, int c,
                                        converter_params params)
{
    layer l = {0};

    l.type = CONVERTER;
    l.batch = batch;
    l.w = l.out_w = w;
    l.h = l.out_h = h;
    l.c = l.out_c = c;
    l.convert_params = params;
    l.outputs = l.out_w * l.out_h * l.out_c;

    /** Only fp32, int8 and uint8 outputs are currently supported
      * TODO: generalize using "void *" in layer and network
      * Note that only one of them is populated based on out_precision
      **/
    l.output = calloc(l.outputs * batch, sizeof(fp32));
    l.output_i8 = calloc(l.outputs * batch, sizeof(int8_t));
    l.output_u8 = calloc(l.outputs * batch, sizeof(uint8_t));

    l.forward = forward_converter_layer;
    l.backward = backward_converter_layer;

    return l;
}

void forward_converter_layer(const converter_layer l, network net)
{
    /* Make a call to precision converter */
    unsigned count = l.outputs * l.batch;
    converter_params params = l.convert_params;
    if (params.in_precision == FP32 && params.out_precision == UINT8) {
        fp32_to_uint8(net.input, l.output_u8, count, params);
    }
    else if (params.in_precision == FP32 && params.out_precision == INT8) {
        fp32_to_int8(net.input, l.output_i8, count, params);
    }
    else if (params.in_precision == UINT8 && params.out_precision == FP32) {
        uint8_to_fp32(net.input_u8, l.output, count, params);
    }
    else if (params.in_precision == INT8 && params.out_precision == FP32) {
        int8_to_fp32(net.input_i8, l.output, count, params);
    }
    else {
        fprintf(stderr, "Unsupported conversion from %s to %s\n",
                get_precision_str(params.in_precision),
                get_precision_str(params.out_precision));
    }
}

void backward_converter_layer(const converter_layer l, network net)
{
    fprintf(stderr, "Backward converter layer!!!\n");
}
