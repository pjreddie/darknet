
__kernel void forward(int in_h, int in_w, int in_c, int stride, int size, __global float *input, __global float *output, __global int *indexes)
{
    int h = (in_h-1)/stride + 1;
    int w = (in_w-1)/stride + 1;
    int c = in_c;

    int id = get_global_id(0);
    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = (-size-1)/2 + 1;
    int h_offset = (-size-1)/2 + 1;

    int out_index = j + w*(i + h*(k + c*b));
    float max = -INFINITY;
    int max_i = -1;
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i*stride + l;
            int cur_w = w_offset + j*stride + m;
            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
            int valid = (cur_h >= 0 && cur_h < in_h &&
                    cur_w >= 0 && cur_w < in_w);
            float val = (valid != 0) ? input[index] : -INFINITY;
            max_i = (val > max) ? index : max_i;
            max   = (val > max) ? val   : max;
        }
    }
    output[out_index] = max;
    indexes[out_index] = max_i;
}

__kernel void backward(int in_h, int in_w, int in_c, int stride, int size, __global float *delta, __global float *prev_delta, __global int *indexes)
{
    int h = (in_h-1)/stride + 1;
    int w = (in_w-1)/stride + 1;
    int c = in_c;
    int area = (size-1)/stride;

    int id = get_global_id(0);
    int index = id;
    int j = id % in_w;
    id /= in_w;
    int i = id % in_h;
    id /= in_h;
    int k = id % in_c;
    id /= in_c;
    int b = id;

    int w_offset = (-size-1)/2 + 1;
    int h_offset = (-size-1)/2 + 1;

    float d = 0;
    int l, m;
    for(l = -area; l < area+1; ++l){
        for(m = -area; m < area+1; ++m){
            int out_w = (j-w_offset)/stride + m;
            int out_h = (i-h_offset)/stride + l;
            int out_index = out_w + w*(out_h + h*(k + c*b));
            int valid = (out_w >= 0 && out_w < w &&
                     out_h >= 0 && out_h < h);
            d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;
        }
    }
    prev_delta[index] = d;
}
