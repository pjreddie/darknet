
__kernel void im2col(__global float *data_im, const int im_offset,
    const int channels, const int height, const int width,
    const int ksize, const int stride, __global float *data_col, const int col_offset) 
{
    int b = get_global_id(0);
    int c = get_global_id(1);

    int h = get_local_id(0);
    int w = get_local_id(1);

    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    int channels_col = channels * ksize * ksize;

    int im_offset = height*width*channels*b;
    int col_offset = height_col*width_col*channels_col*b;

    int w_offset = c % ksize;
    int h_offset = (c / ksize) % ksize;
    int c_im = c / ksize / ksize;

    data_col[(c * height_col + h) * width_col + w + col_offset] =
        data_im[(c_im * height + h * stride + h_offset) * width
        + w * stride + w_offset + im_offset];
}
