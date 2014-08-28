int index(int row, int col)
{
    
}

__kernel void col2im(__global float *data_col,  int batch,
     int channels,  int height,  int width,
     int ksize,  int stride,  int pad, __global float *data_im)
{
    int id = get_global_id(0);
    int index = id;
    int w = id%width;
    id /= width;
    int h = id%height;
    id /= height;
    int c = id%channels;
    id /= channels;
    int b = id%batch;

    int height_col = (height - ksize) / stride + 1;
    int width_col = (width - ksize) / stride + 1;
    int rows = channels * ksize * ksize;
    if (pad){
        height_col = 1 + (height-1) / stride;
        width_col = 1 + (width-1) / stride;
        pad = ksize/2;
    }
    int cols = height_col*width_col;
    int batch_offset = b*cols*rows;
    int channel_offset = c*cols*ksize*ksize;
    data_col[index] = 0;
    int i,j;
    for(i = 0; i < ksize; ++i){
        row_offset = i*height_col*width_col;
        for(j = 0; j < ksize; ++j){
            col_offset = 
        }
    }

    data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, b, im_row, im_col, c_im, pad);
}
