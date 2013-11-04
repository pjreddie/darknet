#include "image.h"
#include <stdio.h>

int windows = 0;

void subtract_image(image a, image b)
{
    int i;
    for(i = 0; i < a.h*a.w*a.c; ++i) a.data[i] -= b.data[i];
}

void normalize_image(image p)
{
    double *min = calloc(p.c, sizeof(double));
    double *max = calloc(p.c, sizeof(double));
    int i,j;
    for(i = 0; i < p.c; ++i) min[i] = max[i] = p.data[i*p.h*p.w];

    for(j = 0; j < p.c; ++j){
        for(i = 0; i < p.h*p.w; ++i){
            double v = p.data[i+j*p.h*p.w];
            if(v < min[j]) min[j] = v;
            if(v > max[j]) max[j] = v;
        }
    }
    for(i = 0; i < p.c; ++i){
        if(max[i] - min[i] < .00001){
            min[i] = 0;
            max[i] = 1;
        }
    }
    for(j = 0; j < p.c; ++j){
        for(i = 0; i < p.w*p.h; ++i){
            p.data[i+j*p.h*p.w] = (p.data[i+j*p.h*p.w] - min[j])/(max[j]-min[j]);
        }
    }
}

void threshold_image(image p, double t)
{
    int i;
    for(i = 0; i < p.w*p.h*p.c; ++i){
        if(p.data[i] < t) p.data[i] = 0;
    }
}

image copy_image(image p)
{
    image copy = p;
    copy.data = calloc(p.h*p.w*p.c, sizeof(double));
    memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(double));
    return copy;
}

void show_image(image p, char *name)
{
    int i,j,k;
    image copy = copy_image(p);
    normalize_image(copy);

    char buff[256];
    sprintf(buff, "%s (%d)", name, windows);

    IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
    int step = disp->widthStep;
    cvNamedWindow(buff, CV_WINDOW_AUTOSIZE); 
    cvMoveWindow(buff, 100*(windows%10) + 200*(windows/10), 100*(windows%10));
    ++windows;
    for(i = 0; i < p.h; ++i){
        for(j = 0; j < p.w; ++j){
            for(k= 0; k < p.c; ++k){
                disp->imageData[i*step + j*p.c + k] = (unsigned char)(get_pixel(copy,i,j,k)*255);
            }
        }
    }
    if(disp->height < 100 || disp->width < 100){
        IplImage *buffer = disp;
        disp = cvCreateImage(cvSize(100,100*p.h/p.w), buffer->depth, buffer->nChannels);
        cvResize(buffer, disp, CV_INTER_NN);
        cvReleaseImage(&buffer);
    }
    cvShowImage(buff, disp);
    cvReleaseImage(&disp);
}

void show_image_layers(image p, char *name)
{
    int i;
    char buff[256];
    for(i = 0; i < p.c; ++i){
        sprintf(buff, "%s - Layer %d", name, i);
        image layer = get_image_layer(p, i);
        show_image(layer, buff);
        free_image(layer);
    }
}

image make_image(int h, int w, int c)
{
    image out;
    out.h = h;
    out.w = w;
    out.c = c;
    out.data = calloc(h*w*c, sizeof(double));
    return out;
}

void zero_image(image m)
{
    memset(m.data, 0, m.h*m.w*m.c*sizeof(double));
}

void zero_channel(image m, int c)
{
    memset(&(m.data[c*m.h*m.w]), 0, m.h*m.w*sizeof(double));
}

void rotate_image(image m)
{
    int i,j;
    for(j = 0; j < m.c; ++j){
        for(i = 0; i < m.h*m.w/2; ++i){
            double swap = m.data[j*m.h*m.w + i];
            m.data[j*m.h*m.w + i] = m.data[j*m.h*m.w + (m.h*m.w-1 - i)];
            m.data[j*m.h*m.w + (m.h*m.w-1 - i)] = swap;
        }
    }
}

image make_random_image(int h, int w, int c)
{
    image out = make_image(h,w,c);
    int i;
    for(i = 0; i < h*w*c; ++i){
        out.data[i] = (double)rand()/RAND_MAX;
    }
    return out;
}

image make_random_kernel(int size, int c)
{
    int pad;
    if((pad=(size%2==0))) ++size;
    image out = make_random_image(size,size,c);
    int i,k;
    if(pad){
        for(k = 0; k < out.c; ++k){
            for(i = 0; i < size; ++i) {
                set_pixel(out, i, 0, k, 0);
                set_pixel(out, 0, i, k, 0);
            }
        }
    }
    return out;
}


image load_image(char *filename)
{
    IplImage* src = 0;
    if( (src = cvLoadImage(filename,-1)) == 0 )
    {
        printf("Cannot load file image %s\n", filename);
        exit(0);
    }
    unsigned char *data = (unsigned char *)src->imageData;
    int c = src->nChannels;
    int h = src->height;
    int w = src->width;
    int step = src->widthStep;
    image out = make_image(h,w,c);
    int i, j, k, count=0;;

    for(k= 0; k < c; ++k){
        for(i = 0; i < h; ++i){
            for(j = 0; j < w; ++j){
                out.data[count++] = data[i*step + j*c + k];
            }
        }
    }
    cvReleaseImage(&src);
    return out;
}

image get_image_layer(image m, int l)
{
    image out = make_image(m.h, m.w, 1);
    int i;
    for(i = 0; i < m.h*m.w; ++i){
        out.data[i] = m.data[i+l*m.h*m.w];
    }
    return out;
}

double get_pixel(image m, int x, int y, int c)
{
    assert(x < m.h && y < m.w && c < m.c);
    return m.data[c*m.h*m.w + x*m.w + y];
}
double get_pixel_extend(image m, int x, int y, int c)
{
    if(x < 0 || x >= m.h || y < 0 || y >= m.w || c < 0 || c >= m.c) return 0;
    return get_pixel(m, x, y, c);
}
void set_pixel(image m, int x, int y, int c, double val)
{
    assert(x < m.h && y < m.w && c < m.c);
    m.data[c*m.h*m.w + x*m.w + y] = val;
}
void set_pixel_extend(image m, int x, int y, int c, double val)
{
    if(x < 0 || x >= m.h || y < 0 || y >= m.w || c < 0 || c >= m.c) return;
    set_pixel(m, x, y, c, val);
}

void add_pixel(image m, int x, int y, int c, double val)
{
    assert(x < m.h && y < m.w && c < m.c);
    m.data[c*m.h*m.w + x*m.w + y] += val;
}

void add_pixel_extend(image m, int x, int y, int c, double val)
{
    if(x < 0 || x >= m.h || y < 0 || y >= m.w || c < 0 || c >= m.c) return;
    add_pixel(m, x, y, c, val);
}

void two_d_convolve(image m, int mc, image kernel, int kc, int stride, image out, int oc)
{
    int x,y,i,j;
    for(x = 0; x < m.h; x += stride){
        for(y = 0; y < m.w; y += stride){
            double sum = 0;
            for(i = 0; i < kernel.h; ++i){
                for(j = 0; j < kernel.w; ++j){
                    sum += get_pixel(kernel, i, j, kc)*get_pixel_extend(m, x+i-kernel.h/2, y+j-kernel.w/2, mc);
                }
            }
            add_pixel(out, x/stride, y/stride, oc, sum);
        }
    }
}

double single_convolve(image m, image kernel, int x, int y)
{
    double sum = 0;
    int i, j, k;
    for(i = 0; i < kernel.h; ++i){
        for(j = 0; j < kernel.w; ++j){
            for(k = 0; k < kernel.c; ++k){
                sum += get_pixel(kernel, i, j, k)*get_pixel_extend(m, x+i-kernel.h/2, y+j-kernel.w/2, k);
            }
        }
    }
    return sum;
}

void convolve(image m, image kernel, int stride, int channel, image out)
{
    assert(m.c == kernel.c);
    int i;
    zero_channel(out, channel);
    for(i = 0; i < m.c; ++i){
        two_d_convolve(m, i, kernel, i, stride, out, channel);
    }
    /*
    int j;
    for(i = 0; i < m.h; i += stride){
        for(j = 0; j < m.w; j += stride){
            double val = single_convolve(m, kernel, i, j);
            set_pixel(out, i/stride, j/stride, channel, val);
        }
    }
    */
}

void upsample_image(image m, int stride, image out)
{
    int i,j,k;
    zero_image(out);
    for(k = 0; k < m.c; ++k){
        for(i = 0; i < m.h; ++i){
            for(j = 0; j< m.w; ++j){
                double val = get_pixel(m, i, j, k);
                set_pixel(out, i*stride, j*stride, k, val);
            }
        }
    }
}

void single_update(image m, image update, int x, int y, double error)
{
    int i, j, k;
    for(i = 0; i < update.h; ++i){
        for(j = 0; j < update.w; ++j){
            for(k = 0; k < update.c; ++k){
                double val = get_pixel_extend(m, x+i-update.h/2, y+j-update.w/2, k);
                add_pixel(update, i, j, k, val*error);
            }
        }
    }
}

void kernel_update(image m, image update, int stride, int channel, image out)
{
    assert(m.c == update.c);
    zero_image(update);
    int i, j;
    for(i = 0; i < m.h; i += stride){
        for(j = 0; j < m.w; j += stride){
            double error = get_pixel(out, i/stride, j/stride, channel);
            single_update(m, update, i, j, error);
        }
    }
    for(i = 0; i < update.h*update.w*update.c; ++i){
        update.data[i] /= (m.h/stride)*(m.w/stride);
    }
}

void single_back_convolve(image m, image kernel, int x, int y, double val)
{
    int i, j, k;
    for(i = 0; i < kernel.h; ++i){
        for(j = 0; j < kernel.w; ++j){
            for(k = 0; k < kernel.c; ++k){
                double pval = get_pixel(kernel, i, j, k) * val;
                add_pixel_extend(m, x+i-kernel.h/2, y+j-kernel.w/2, k, pval);
            }
        }
    }
}

void back_convolve(image m, image kernel, int stride, int channel, image out)
{
    assert(m.c == kernel.c);
    int i, j;
    for(i = 0; i < m.h; i += stride){
        for(j = 0; j < m.w; j += stride){
            double val = get_pixel(out, i/stride, j/stride, channel);
            single_back_convolve(m, kernel, i, j, val);
        }
    }
}

void free_image(image m)
{
    free(m.data);
}
