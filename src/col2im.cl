__kernel void col2im(__global float *data_col, int batch,
        int channels, int height, int width,
        int ksize, int stride, int pad, __global float *data_im)
{

    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    if (pad){
        height_col = 1 + (height-1) / stride;
        width_col = 1 + (width-1) / stride;
        pad = ksize/2;
    }

    int id = get_global_id(0);
    int index = id;
    int w = id%width + pad;
    id /= width;
    int h = id%height + pad;
    id /= height;
    int c = id%channels;
    id /= channels;
    int b = id%batch;

    int w_start = (w<ksize)?0:(w-ksize)/stride + 1;
    int w_end = w/stride + 1;
    if(width_col < w_end) w_end = width_col;

    int h_start = (h<ksize)?0:(h-ksize)/stride+1;
    int h_end = h/stride + 1;
    if(height_col < h_end) h_end = height_col;

    int rows = channels * ksize * ksize;
    int cols = height_col*width_col;
    int offset = (c*ksize*ksize + h * ksize + w)*height_col*width_col;
    offset += b*cols*rows;
    int h_coeff = (1-stride*ksize*height_col)*width_col;
    int w_coeff = 1-stride*height_col*width_col;
    float val = 0;
    int h_col, w_col;
    for(h_col = h_start; h_col < h_end; ++h_col){
        for(w_col = w_start; w_col < w_end; ++w_col){
            val += data_col[offset +h_col*h_coeff + w_col*w_coeff];
        }
    }
    data_im[index] = val;
}
