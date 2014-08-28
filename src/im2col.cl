
float im2col_get_pixel(__global float *im, int height, int width, int channels,
                       int batch, int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width) return 0;
    int index = col + width*(row + height*(channel+batch*channels));
    return im[index];
}

__kernel void im2col(__global float *data_im,  int batch,
     int channels,  int height,  int width,
     int ksize,  int stride,  int pad, __global float *data_col)
{
    int c,h,w,b;
    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    if (pad){
        height_col = 1 + (height-1) / stride;
        width_col = 1 + (width-1) / stride;
        pad = ksize/2;
    }
    int gid1 = get_global_id(0);
    b = gid1%batch;
    c = gid1/batch;

    int gid2 = get_global_id(1);
    h = gid2%height_col;
    w = gid2/height_col;


    int channels_col = channels * ksize * ksize;
    int col_size = height_col*width_col*channels_col;
    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;
    int im_row = h_offset + h * stride;
    int im_col = w_offset + w * stride;
    int col_index = (c * height_col + h) * width_col + w + b*col_size;
    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, b, im_row, im_col, c_im, pad);
}
