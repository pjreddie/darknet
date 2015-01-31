extern "C" {
#include "crop_layer.h"
#include "cuda.h"
}

#define BLOCK 256

__global__ void forward_crop_layer_kernel(float *input, int size, int c, int h, int w, int crop_height, int crop_width, int dh, int dw, int flip, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= size) return;

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

extern "C" void forward_crop_layer_gpu(crop_layer layer, int train, float *input)
{
    int flip = (layer.flip && rand()%2);
    int dh = rand()%(layer.h - layer.crop_height + 1);
    int dw = rand()%(layer.w - layer.crop_width + 1);
    if(!train){
        flip = 0;
        dh = (layer.h - layer.crop_height)/2;
        dw = (layer.w - layer.crop_width)/2;
    }
    int size = layer.batch*layer.c*layer.crop_width*layer.crop_height;

    dim3 dimBlock(BLOCK, 1, 1);
    dim3 dimGrid((size-1)/BLOCK + 1, 1, 1);

    forward_crop_layer_kernel<<<cuda_gridsize(size), BLOCK>>>(input, size, layer.c, layer.h, layer.w,
                        layer.crop_height, layer.crop_width, dh, dw, flip, layer.output_gpu);
    check_error(cudaPeekAtLastError());
}

