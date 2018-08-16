#include "converter_layer.h"
#include "converter.h"

#include "input_data.h"
#include "reshaped_data.h"

unsigned int roundup_and_align(unsigned int val, unsigned int round_to)
{
    unsigned int rounded;

    if (val % round_to != 0) {
        rounded = val + round_to - (val % round_to);
    } else {
        rounded = val;
    }

    return rounded;
}

converter_layer make_converter_layer(int batch, int w, int h, int c,
                                        converter_params params)
{
    layer l = {0};
    unsigned int num_out;

    l.type = CONVERTER;
    l.batch = batch;
    l.w = l.out_w = w;
    l.h = l.out_h = h;
    l.c = l.out_c = c;
    l.convert_params = params;
    l.outputs = l.out_w * l.out_h * l.out_c;
    num_out = l.out_w * l.out_h * roundup_and_align(l.out_c, 32);

    fprintf(stderr, "outputs %u num_out %u\n", l.outputs, num_out);

    if (params.out_precision == FP32)
        l.output = calloc(num_out * batch, sizeof(fp32));

    if (params.out_precision == INT8 || params.out_precision == UINT8)
        l.output_i8 = calloc(num_out * batch, sizeof(int8_t));

    l.forward = forward_converter_layer;
    l.backward = backward_converter_layer;

    return l;
}

void convert_nchw_to_nhwc(uint8_t *in, int w, int h, int c, uint8_t *out)
{
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < w; k++) {
                out[j*w*c + c*k + i] = in[w*h*i + w*j + k];
            }
        }
    }
}

void convert_fd_to_nchw(float *in, int w, int h, int c, float *out)
{
    unsigned int line_stride = w * 32;
    unsigned int surface_stride = line_stride * h;
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < w; k++) {
                int surface_index = i/32;
                out[w*h*i + w*j + k] = in[surface_stride*surface_index + line_stride*j + 32*k + i%32];
            }
        }
    }
}

static void odla_dump_image_data(uint8_t *data, int w, int h, int c)
{
    FILE *fp;

    fp = fopen("image_input.txt", "w");

    fprintf(fp, "blobs {\n");
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < w; k++) {
                fprintf(fp, "  double_data: %u\n", data[w*h*i + w*j + k]);
            }
        }
    }
    fprintf(fp, "  shape {\n");
    fprintf(fp, "    dim: 1\n");
    fprintf(fp, "    dim: %d\n", c);
    fprintf(fp, "    dim: %d\n", h);
    fprintf(fp, "    dim: %d\n", w);
    fprintf(fp, "  }\n");
    fprintf(fp, "}\n");

    fclose(fp);
}

void forward_converter_layer(const converter_layer l, network net)
{
    /* Make a call to precision converter */
    unsigned count = l.outputs * l.batch;
    unsigned int bufsize = l.out_w * l.out_h * roundup_and_align(l.out_c, 32);
    converter_params params = l.convert_params;

    if (params.in_precision == FP32 && params.out_precision == UINT8) {
#if 1
        uint8_t *temp = calloc(bufsize, sizeof(uint8_t));
        fp32_to_uint8(net.input, temp, count, params);
        odla_dump_image_data(temp, l.w, l.h, l.c);
        convert_nchw_to_nhwc(temp, l.w, l.h, l.c, (uint8_t*)l.output_i8);
        free(temp);
#endif
        //reference data for validation
//        memcpy((uint8_t *)l.output_i8, reshaped_data, 692224);
    } else if (params.in_precision == INT8 && params.out_precision == FP32) {
        float *temp = calloc(bufsize, sizeof(float));
        int8_to_fp32(net.input_i8, temp, count, params);
        convert_fd_to_nchw(temp, l.w, l.h, l.c, l.output);
        free(temp);
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
