#include "upsample_odla_layer.h"
#include "blas.h"

#include <stdio.h>

layer make_upsample_odla_layer(int batch, int w, int h, int c, int stride, int output_layer, int tensor)
{
    layer l = {0};
    l.type = UPSAMPLE_ODLA;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w*stride;
    l.out_h = h*stride;
    l.out_c = c;
    l.stride = stride;
    l.upsample_output_layer = output_layer;
    l.upsample_output_tensor = tensor;
    l.reverse = 0;

    l.forward = forward_upsample_odla_layer;

    if(l.reverse) fprintf(stderr, "downsample_dla     %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    else fprintf(stderr, "upsample_dla       %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void cubecpy(uint8_t *dst, const uint8_t *src, int dst_line_stride, int stride)
{
    for(int y = 0; y < stride; y++) {
        for(int x = 0; x < stride; x++) {
            memcpy(&dst[y*dst_line_stride+x*ATOMIC_CUBE], src, ATOMIC_CUBE);
        }
    }
}

void upsample_dla(int8_t *in, int w, int h, int c, int batch, int stride, int forward, int8_t *out)
{
    int i, j, k, b;
    int surf_num = (c+ATOMIC_CUBE-1)/ATOMIC_CUBE;
    int src_cube_size = surf_num*w*h*ATOMIC_CUBE, src_surf_stride = w*ATOMIC_CUBE*h, src_line_stride = w*ATOMIC_CUBE;
    int dst_cube_size = surf_num*w*h*ATOMIC_CUBE*stride*stride, dst_surf_stride = w*ATOMIC_CUBE*h*stride*stride, dst_line_stride = w*ATOMIC_CUBE*stride;

    fprintf(stderr, "upsample data, w %d h %d c %d\n", w, h, c);
    for(b = 0; b < batch; ++b){
        for(k = 0; k < surf_num; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int in_index = b*src_cube_size + k*src_surf_stride + j*src_line_stride + i*ATOMIC_CUBE;
                    int out_index = b*dst_cube_size + k*dst_surf_stride + j*stride*dst_line_stride + i*stride*ATOMIC_CUBE;
                    if(forward) cubecpy(out + out_index, in + in_index, dst_line_stride, stride);
                }
            }
        }
    }
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
}

static void odla_dump_us_data(const char *filename, int8_t *data, int w, int h, int c)
{
    FILE *fp;

    fp = fopen(filename, "w");

    unsigned int line_stride = w * 32;
    unsigned int surface_stride = line_stride * h;

    fprintf(fp, "blobs {\n");
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < w; k++) {
                int surface_index = i / 32;
                fprintf(fp, "  double_data: %d\n", data[surface_stride*surface_index + line_stride*j + 32*k + i%32]);
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

void forward_upsample_odla_layer(const layer l, network net)
{
    int8_t *output;
    layer *output_layer;

    fprintf(stderr, "forward_upsample_odla_layer output layer %d tensor %d\n", l.upsample_output_layer, l.upsample_output_tensor);

    output_layer = &net.layers[l.upsample_output_layer];
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
    output = output_layer->input_tensors[l.upsample_output_tensor].buffer;
    fprintf(stderr, "%s %d\n", __func__, __LINE__);

    upsample_dla(net.input_i8, l.w, l.h, l.c, l.batch, l.stride, 1, output);

    char filename[80];
    snprintf(filename, sizeof(filename), "upsample_output_%02d.dimg", l.layer_index);
    odla_dump_us_data(filename, output, l.out_w, l.out_h, l.out_c);
    fprintf(stderr, "%s %d\n", __func__, __LINE__);
}
