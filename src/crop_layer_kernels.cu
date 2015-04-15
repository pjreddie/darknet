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

__device__ float3 rgb_to_hsv_kernel(float3 rgb)
{
    float r = rgb.x;
    float g = rgb.y; 
    float b = rgb.z;

    float h, s, v;
    float max = (r > g) ? ( (r > b) ? r : b) : ( (g > b) ? g : b);
    float min = (r < g) ? ( (r < b) ? r : b) : ( (g < b) ? g : b);
    float delta = max - min;
    v = max;
    if(max == 0){
        s = 0;
        h = -1;
    }else{
        s = delta/max;
        if(r == max){
            h = (g - b) / delta;
        } else if (g == max) {
            h = 2 + (b - r) / delta;
        } else {
            h = 4 + (r - g) / delta;
        }
        if (h < 0) h += 6;
    }
    return make_float3(h, s, v);
}

__device__ float3 hsv_to_rgb_kernel(float3 hsv)
{
    float h = hsv.x;
    float s = hsv.y; 
    float v = hsv.z;

    float r, g, b;
    float f, p, q, t;

    if (s == 0) {
        r = g = b = v;
    } else {
        int index = (int) floorf(h);
        f = h - index;
        p = v*(1-s);
        q = v*(1-s*f);
        t = v*(1-s*(1-f));
        if(index == 0){
            r = v; g = t; b = p;
        } else if(index == 1){
            r = q; g = v; b = p;
        } else if(index == 2){
            r = p; g = v; b = t;
        } else if(index == 3){
            r = p; g = q; b = v;
        } else if(index == 4){
            r = t; g = p; b = v;
        } else {
            r = v; g = p; b = q;
        }
    }
    r = (r < 0) ? 0 : ((r > 1) ? 1 : r);
    g = (g < 0) ? 0 : ((g > 1) ? 1 : g);
    b = (b < 0) ? 0 : ((b > 1) ? 1 : b);
    return make_float3(r, g, b);
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

__global__ void levels_image_kernel(float *image, int batch, int w, int h, float saturation, float exposure, float translate, float scale)
{
    int size = batch * w * h;
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= size) return;
    int x = id % w;
    id /= w;
    int y = id % h;
    id /= h;
    size_t offset = id * h * w * 3;
    image += offset;
    float r = image[x + w*(y + h*2)];
    float g = image[x + w*(y + h*1)];
    float b = image[x + w*(y + h*0)];
    float3 rgb = make_float3(r,g,b);
    float3 hsv = rgb_to_hsv_kernel(rgb);
    hsv.y *= saturation;
    hsv.z *= exposure;
    rgb = hsv_to_rgb_kernel(hsv);
    image[x + w*(y + h*2)] = rgb.x*scale + translate;
    image[x + w*(y + h*1)] = rgb.y*scale + translate;
    image[x + w*(y + h*0)] = rgb.z*scale + translate;
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
    float radians = layer.angle*3.14159/180.;
    float angle = 2*radians*rand_uniform() - radians;

    float saturation = rand_uniform() + 1;
    if(rand_uniform() > .5) saturation = 1./saturation;
    float exposure = rand_uniform() + 1;
    if(rand_uniform() > .5) exposure = 1./exposure;

    float scale = 2;
    float translate = -1;

    if(!state.train){
        angle = 0;
        flip = 0;
        dh = (layer.h - layer.crop_height)/2;
        dw = (layer.w - layer.crop_width)/2;
        saturation = 1;
        exposure = 1;
    }

    int size = layer.batch * layer.w * layer.h;

    levels_image_kernel<<<cuda_gridsize(size), BLOCK>>>(state.input, layer.batch, layer.w, layer.h, saturation, exposure, translate, scale);
    check_error(cudaPeekAtLastError());
    
    size = layer.batch*layer.c*layer.crop_width*layer.crop_height;

    forward_crop_layer_kernel<<<cuda_gridsize(size), BLOCK>>>(state.input, size, layer.c, layer.h, layer.w,
            layer.crop_height, layer.crop_width, dh, dw, flip, angle, layer.output_gpu);
    check_error(cudaPeekAtLastError());

/*
       cuda_pull_array(layer.output_gpu, layer.output, size);
       image im = float_to_image(layer.crop_width, layer.crop_height, layer.c, layer.output + 0*(size/layer.batch));
       image im2 = float_to_image(layer.crop_width, layer.crop_height, layer.c, layer.output + 1*(size/layer.batch));
       image im3 = float_to_image(layer.crop_width, layer.crop_height, layer.c, layer.output + 2*(size/layer.batch));
       show_image(im, "cropped");
       show_image(im2, "cropped2");
       show_image(im3, "cropped3");
       cvWaitKey(0);
       */
}

