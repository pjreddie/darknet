extern "C" {
#include "crop_layer.h"
#include "utils.h"
#include "cuda.h"
#include "image.h"
}

#define BLOCK 256

__device__ float get_pixel_kernel(float *image, int w, int h, int x, int y, int c)
{
    if(x < 0 || x >= w || y < 0 || y >= h) return 0;
    return image[x + w*(y + c*h)];
}

__device__ float billinear_interpolate_kernel(float *image, int w, int h, float x, float y, int c)
{
    int ix = (int) floorf(x);
    int iy = (int) floorf(y);

    float dx = x - ix;
    float dy = y - iy;

    float val = (1-dy) * (1-dx) * get_pixel_kernel(image, w, h, ix, iy, c) + 
                dy     * (1-dx) * get_pixel_kernel(image, w, h, ix, iy+1, c) + 
                (1-dy) *   dx   * get_pixel_kernel(image, w, h, ix+1, iy, c) +
                dy     *   dx   * get_pixel_kernel(image, w, h, ix+1, iy+1, c);
    return val;
}

__global__ void forward_crop_layer_kernel(float *input, int size, int c, int h, int w, int crop_height, int crop_width, int dh, int dw, int flip, float angle, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= size) return;

    float cx = w/2.;
    float cy = h/2.;

    int count = id;
    int j = id % crop_width;
    id /= crop_width;
    int i = id % crop_height;
    id /= crop_height;
    int k = id % c;
    id /= c;
    int b = id;

    input += w*h*c*b;

    int x = (flip) ? w - dw - j - 1 : j + dw;    
    int y = i + dh;

    float rx = cos(angle)*(x-cx) - sin(angle)*(y-cy) + cx;
    float ry = sin(angle)*(x-cx) + cos(angle)*(y-cy) + cy;

    output[count] = billinear_interpolate_kernel(input, w, h, rx, ry, k);
}

extern "C" void forward_crop_layer_gpu(crop_layer layer, network_state state)
{
    int flip = (layer.flip && rand()%2);
    int dh = rand()%(layer.h - layer.crop_height + 1);
    int dw = rand()%(layer.w - layer.crop_width + 1);
    float angle = rand_uniform() - .5;
    if(!state.train){
        angle = 0;
        flip = 0;
        dh = (layer.h - layer.crop_height)/2;
        dw = (layer.w - layer.crop_width)/2;
    }
    int size = layer.batch*layer.c*layer.crop_width*layer.crop_height;

    dim3 dimBlock(BLOCK, 1, 1);
    dim3 dimGrid((size-1)/BLOCK + 1, 1, 1);

    forward_crop_layer_kernel<<<cuda_gridsize(size), BLOCK>>>(state.input, size, layer.c, layer.h, layer.w,
                        layer.crop_height, layer.crop_width, dh, dw, flip, angle, layer.output_gpu);
    check_error(cudaPeekAtLastError());
}

