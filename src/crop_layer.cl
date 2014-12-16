__kernel void forward(__global float *input, int c, int h, int w, int crop_height, int crop_width, int dh, int dw, int flip, __global float *output)
{
    int id = get_global_id(0);
    int count = id;
    int j = id % crop_width;
    id /= crop_width;
    int i = id % crop_height;
    id /= crop_height;
    int k = id % c;
    id /= c;
    int b = id;
    int col = (flip) ? w - dw - j - 1 : j + dw;    
    int row = i + dh;
    int index = col+w*(row+h*(k + c*b)); 
    output[count] = input[index];
}
