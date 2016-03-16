#include "image.h"
#include "utils.h"
#include "blas.h"
#include <stdio.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int windows = 0;

float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

float get_color(int c, int x, int max)
{
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    //printf("%f\n", r);
    return r;
}

void draw_label(image a, int r, int c, image label, const float *rgb)
{
    float ratio = (float) label.w / label.h;
    int h = label.h;
    int w = ratio * h;
    image rl = resize_image(label, w, h);
    if (r - h >= 0) r = r - h;

    int i, j, k;
    for(j = 0; j < h && j + r < a.h; ++j){
        for(i = 0; i < w && i + c < a.w; ++i){
            for(k = 0; k < label.c; ++k){
                float val = get_pixel(rl, i, j, k);
                set_pixel(a, i+c, j+r, k, rgb[k] * val);
            }
        }
    }
    free_image(rl);
}

void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
    //normalize_image(a);
    int i;
    if(x1 < 0) x1 = 0;
    if(x1 >= a.w) x1 = a.w-1;
    if(x2 < 0) x2 = 0;
    if(x2 >= a.w) x2 = a.w-1;

    if(y1 < 0) y1 = 0;
    if(y1 >= a.h) y1 = a.h-1;
    if(y2 < 0) y2 = 0;
    if(y2 >= a.h) y2 = a.h-1;

    for(i = x1; i <= x2; ++i){
        a.data[i + y1*a.w + 0*a.w*a.h] = r;
        a.data[i + y2*a.w + 0*a.w*a.h] = r;

        a.data[i + y1*a.w + 1*a.w*a.h] = g;
        a.data[i + y2*a.w + 1*a.w*a.h] = g;

        a.data[i + y1*a.w + 2*a.w*a.h] = b;
        a.data[i + y2*a.w + 2*a.w*a.h] = b;
    }
    for(i = y1; i <= y2; ++i){
        a.data[x1 + i*a.w + 0*a.w*a.h] = r;
        a.data[x2 + i*a.w + 0*a.w*a.h] = r;

        a.data[x1 + i*a.w + 1*a.w*a.h] = g;
        a.data[x2 + i*a.w + 1*a.w*a.h] = g;

        a.data[x1 + i*a.w + 2*a.w*a.h] = b;
        a.data[x2 + i*a.w + 2*a.w*a.h] = b;
    }
}

void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
    int i;
    for(i = 0; i < w; ++i){
        draw_box(a, x1+i, y1+i, x2-i, y2-i, r, g, b);
    }
}

void draw_bbox(image a, box bbox, int w, float r, float g, float b)
{
    int left  = (bbox.x-bbox.w/2)*a.w;
    int right = (bbox.x+bbox.w/2)*a.w;
    int top   = (bbox.y-bbox.h/2)*a.h;
    int bot   = (bbox.y+bbox.h/2)*a.h;

    int i;
    for(i = 0; i < w; ++i){
        draw_box(a, left+i, top+i, right-i, bot-i, r, g, b);
    }
}

void draw_detections(image im, int num, float thresh, box *boxes, float **probs, char **names, image *labels, int classes)
{
    int i;

    for(i = 0; i < num; ++i){
        int class = max_index(probs[i], classes);
        float prob = probs[i][class];
        if(prob > thresh){
            int width = pow(prob, 1./2.)*10+1;
            printf("%s: %.2f\n", names[class], prob);
            int offset = class*17 % classes;
            float red = get_color(0,offset,classes);
            float green = get_color(1,offset,classes);
            float blue = get_color(2,offset,classes);
            float rgb[3];
            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b = boxes[i];

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

            draw_box_width(im, left, top, right, bot, width, red, green, blue);
            if (labels) draw_label(im, top + width, left, labels[class], rgb);
        }
    }
}

void transpose_image(image im)
{
    assert(im.w == im.h);
    int n, m;
    int c;
    for(c = 0; c < im.c; ++c){
        for(n = 0; n < im.w-1; ++n){
            for(m = n + 1; m < im.w; ++m){
                float swap = im.data[m + im.w*(n + im.h*c)];
                im.data[m + im.w*(n + im.h*c)] = im.data[n + im.w*(m + im.h*c)];
                im.data[n + im.w*(m + im.h*c)] = swap;
            }
        }
    }
}

void rotate_image_cw(image im, int times)
{
    assert(im.w == im.h);
    times = (times + 400) % 4;
    int i, x, y, c;
    int n = im.w;
    for(i = 0; i < times; ++i){
        for(c = 0; c < im.c; ++c){
            for(x = 0; x < n/2; ++x){
                for(y = 0; y < (n-1)/2 + 1; ++y){
                    float temp = im.data[y + im.w*(x + im.h*c)];
                    im.data[y + im.w*(x + im.h*c)] = im.data[n-1-x + im.w*(y + im.h*c)];
                    im.data[n-1-x + im.w*(y + im.h*c)] = im.data[n-1-y + im.w*(n-1-x + im.h*c)];
                    im.data[n-1-y + im.w*(n-1-x + im.h*c)] = im.data[x + im.w*(n-1-y + im.h*c)];
                    im.data[x + im.w*(n-1-y + im.h*c)] = temp;
                }
            }
        }
    }
}

void flip_image(image a)
{
    int i,j,k;
    for(k = 0; k < a.c; ++k){
        for(i = 0; i < a.h; ++i){
            for(j = 0; j < a.w/2; ++j){
                int index = j + a.w*(i + a.h*(k));
                int flip = (a.w - j - 1) + a.w*(i + a.h*(k));
                float swap = a.data[flip];
                a.data[flip] = a.data[index];
                a.data[index] = swap;
            }
        }
    }
}

image image_distance(image a, image b)
{
    int i,j;
    image dist = make_image(a.w, a.h, 1);
    for(i = 0; i < a.c; ++i){
        for(j = 0; j < a.h*a.w; ++j){
            dist.data[j] += pow(a.data[i*a.h*a.w+j]-b.data[i*a.h*a.w+j],2);
        }
    }
    for(j = 0; j < a.h*a.w; ++j){
        dist.data[j] = sqrt(dist.data[j]);
    }
    return dist;
}

void embed_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float val = get_pixel(source, x,y,k);
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}

image collapse_image_layers(image source, int border)
{
    int h = source.h;
    h = (h+border)*source.c - border;
    image dest = make_image(source.w, h, 1);
    int i;
    for(i = 0; i < source.c; ++i){
        image layer = get_image_layer(source, i);
        int h_offset = i*(source.h+border);
        embed_image(layer, dest, 0, h_offset);
        free_image(layer);
    }
    return dest;
}

void constrain_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h*im.c; ++i){
        if(im.data[i] < 0) im.data[i] = 0;
        if(im.data[i] > 1) im.data[i] = 1;
    }
}

void normalize_image(image p)
{
    float *min = calloc(p.c, sizeof(float));
    float *max = calloc(p.c, sizeof(float));
    int i,j;
    for(i = 0; i < p.c; ++i) min[i] = max[i] = p.data[i*p.h*p.w];

    for(j = 0; j < p.c; ++j){
        for(i = 0; i < p.h*p.w; ++i){
            float v = p.data[i+j*p.h*p.w];
            if(v < min[j]) min[j] = v;
            if(v > max[j]) max[j] = v;
        }
    }
    for(i = 0; i < p.c; ++i){
        if(max[i] - min[i] < .000000001){
            min[i] = 0;
            max[i] = 1;
        }
    }
    for(j = 0; j < p.c; ++j){
        for(i = 0; i < p.w*p.h; ++i){
            p.data[i+j*p.h*p.w] = (p.data[i+j*p.h*p.w] - min[j])/(max[j]-min[j]);
        }
    }
    free(min);
    free(max);
}

image copy_image(image p)
{
    image copy = p;
    copy.data = calloc(p.h*p.w*p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));
    return copy;
}

void rgbgr_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}

#ifdef OPENCV
void show_image_cv(image p, const char *name)
{
    int x,y,k;
    image copy = copy_image(p);
    constrain_image(copy);
    if(p.c == 3) rgbgr_image(copy);
    //normalize_image(copy);

    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    sprintf(buff, "%s", name);

    IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
    int step = disp->widthStep;
    cvNamedWindow(buff, CV_WINDOW_NORMAL); 
    //cvMoveWindow(buff, 100*(windows%10) + 200*(windows/10), 100*(windows%10));
    ++windows;
    for(y = 0; y < p.h; ++y){
        for(x = 0; x < p.w; ++x){
            for(k= 0; k < p.c; ++k){
                disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(copy,x,y,k)*255);
            }
        }
    }
    free_image(copy);
    if(0){
        //if(disp->height < 448 || disp->width < 448 || disp->height > 1000){
        int w = 448;
        int h = w*p.h/p.w;
        if(h > 1000){
            h = 1000;
            w = h*p.w/p.h;
        }
        IplImage *buffer = disp;
        disp = cvCreateImage(cvSize(w, h), buffer->depth, buffer->nChannels);
        cvResize(buffer, disp, CV_INTER_LINEAR);
        cvReleaseImage(&buffer);
    }
    cvShowImage(buff, disp);
    cvReleaseImage(&disp);
    }
#endif

    void show_image(image p, const char *name)
    {
#ifdef OPENCV
        show_image_cv(p, name);
#else
        fprintf(stderr, "Not compiled with OpenCV, saving to %s.png instead\n", name);
        save_image(p, name);
#endif
    }

    void save_image(image im, const char *name)
    {
        char buff[256];
        //sprintf(buff, "%s (%d)", name, windows);
        sprintf(buff, "%s.png", name);
        unsigned char *data = calloc(im.w*im.h*im.c, sizeof(char));
        int i,k;
        for(k = 0; k < im.c; ++k){
            for(i = 0; i < im.w*im.h; ++i){
                data[i*im.c+k] = (unsigned char) (255*im.data[i + k*im.w*im.h]);
            }
        }
        int success = stbi_write_png(buff, im.w, im.h, im.c, data, im.w*im.c);
        free(data);
        if(!success) fprintf(stderr, "Failed to write image %s\n", buff);
    }

#ifdef OPENCV
    image get_image_from_stream(CvCapture *cap)
    {
        IplImage* src = cvQueryFrame(cap);
        image im = ipl_to_image(src);
        rgbgr_image(im);
        return im;
    }
#endif

#ifdef OPENCV
    void save_image_jpg(image p, char *name)
    {
        image copy = copy_image(p);
        rgbgr_image(copy);
        int x,y,k;

        char buff[256];
        sprintf(buff, "%s.jpg", name);

        IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
        int step = disp->widthStep;
        for(y = 0; y < p.h; ++y){
            for(x = 0; x < p.w; ++x){
                for(k= 0; k < p.c; ++k){
                    disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(copy,x,y,k)*255);
                }
            }
        }
        cvSaveImage(buff, disp,0);
        cvReleaseImage(&disp);
        free_image(copy);
    }
#endif

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

    void show_image_collapsed(image p, char *name)
    {
        image c = collapse_image_layers(p, 1);
        show_image(c, name);
        free_image(c);
    }

    image make_empty_image(int w, int h, int c)
    {
        image out;
        out.data = 0;
        out.h = h;
        out.w = w;
        out.c = c;
        return out;
    }

    image make_image(int w, int h, int c)
    {
        image out = make_empty_image(w,h,c);
        out.data = calloc(h*w*c, sizeof(float));
        return out;
    }

    image make_random_image(int w, int h, int c)
    {
        image out = make_empty_image(w,h,c);
        out.data = calloc(h*w*c, sizeof(float));
        int i;
        for(i = 0; i < w*h*c; ++i){
            out.data[i] = (rand_normal() * .25) + .5;
        }
        return out;
    }

    image float_to_image(int w, int h, int c, float *data)
    {
        image out = make_empty_image(w,h,c);
        out.data = data;
        return out;
    }

    image rotate_image(image im, float rad)
    {
        int x, y, c;
        float cx = im.w/2.;
        float cy = im.h/2.;
        image rot = make_image(im.w, im.h, im.c);
        for(c = 0; c < im.c; ++c){
            for(y = 0; y < im.h; ++y){
                for(x = 0; x < im.w; ++x){
                    float rx = cos(rad)*(x-cx) - sin(rad)*(y-cy) + cx;
                    float ry = sin(rad)*(x-cx) + cos(rad)*(y-cy) + cy;
                    float val = bilinear_interpolate(im, rx, ry, c);
                    set_pixel(rot, x, y, c, val);
                }
            }
        }
        return rot;
    }

    void translate_image(image m, float s)
    {
        int i;
        for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] += s;
    }

    void scale_image(image m, float s)
    {
        int i;
        for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] *= s;
    }

    image crop_image(image im, int dx, int dy, int w, int h)
    {
        image cropped = make_image(w, h, im.c);
        int i, j, k;
        for(k = 0; k < im.c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int r = j + dy;
                    int c = i + dx;
                    float val = 0;
                    if (r >= 0 && r < im.h && c >= 0 && c < im.w) {
                        val = get_pixel(im, c, r, k);
                    }
                    set_pixel(cropped, i, j, k, val);
                }
            }
        }
        return cropped;
    }

    image resize_min(image im, int min)
    {
        int w = im.w;
        int h = im.h;
        if(w < h){
            h = (h * min) / w;
            w = min;
        } else {
            w = (w * min) / h;
            h = min;
        }
        image resized = resize_image(im, w, h);
        return resized;
    }

    image random_crop_image(image im, int low, int high, int size)
    {
        int r = rand_int(low, high);
        image resized = resize_min(im, r);
        int dx = rand_int(0, resized.w - size);
        int dy = rand_int(0, resized.h - size);
        image crop = crop_image(resized, dx, dy, size, size);

        /*
           show_image(im, "orig");
           show_image(crop, "cropped");
           cvWaitKey(0);
         */

        free_image(resized);
        return crop;
    }

    float three_way_max(float a, float b, float c)
    {
        return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
    }

    float three_way_min(float a, float b, float c)
    {
        return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
    }

    // http://www.cs.rit.edu/~ncs/color/t_convert.html
    void rgb_to_hsv(image im)
    {
        assert(im.c == 3);
        int i, j;
        float r, g, b;
        float h, s, v;
        for(j = 0; j < im.h; ++j){
            for(i = 0; i < im.w; ++i){
                r = get_pixel(im, i , j, 0);
                g = get_pixel(im, i , j, 1);
                b = get_pixel(im, i , j, 2);
                float max = three_way_max(r,g,b);
                float min = three_way_min(r,g,b);
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
                set_pixel(im, i, j, 0, h);
                set_pixel(im, i, j, 1, s);
                set_pixel(im, i, j, 2, v);
            }
        }
    }

    void hsv_to_rgb(image im)
    {
        assert(im.c == 3);
        int i, j;
        float r, g, b;
        float h, s, v;
        float f, p, q, t;
        for(j = 0; j < im.h; ++j){
            for(i = 0; i < im.w; ++i){
                h = get_pixel(im, i , j, 0);
                s = get_pixel(im, i , j, 1);
                v = get_pixel(im, i , j, 2);
                if (s == 0) {
                    r = g = b = v;
                } else {
                    int index = floor(h);
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
                set_pixel(im, i, j, 0, r);
                set_pixel(im, i, j, 1, g);
                set_pixel(im, i, j, 2, b);
            }
        }
    }

    image grayscale_image(image im)
    {
        assert(im.c == 3);
        int i, j, k;
        image gray = make_image(im.w, im.h, 1);
        float scale[] = {0.587, 0.299, 0.114};
        for(k = 0; k < im.c; ++k){
            for(j = 0; j < im.h; ++j){
                for(i = 0; i < im.w; ++i){
                    gray.data[i+im.w*j] += scale[k]*get_pixel(im, i, j, k);
                }
            }
        }
        return gray;
    }

    image threshold_image(image im, float thresh)
    {
        int i;
        image t = make_image(im.w, im.h, im.c);
        for(i = 0; i < im.w*im.h*im.c; ++i){
            t.data[i] = im.data[i]>thresh ? 1 : 0;
        }
        return t;
    }

    image blend_image(image fore, image back, float alpha)
    {
        assert(fore.w == back.w && fore.h == back.h && fore.c == back.c);
        image blend = make_image(fore.w, fore.h, fore.c);
        int i, j, k;
        for(k = 0; k < fore.c; ++k){
            for(j = 0; j < fore.h; ++j){
                for(i = 0; i < fore.w; ++i){
                    float val = alpha * get_pixel(fore, i, j, k) + 
                        (1 - alpha)* get_pixel(back, i, j, k);
                    set_pixel(blend, i, j, k, val);
                }
            }
        }
        return blend;
    }

    void scale_image_channel(image im, int c, float v)
    {
        int i, j;
        for(j = 0; j < im.h; ++j){
            for(i = 0; i < im.w; ++i){
                float pix = get_pixel(im, i, j, c);
                pix = pix*v;
                set_pixel(im, i, j, c, pix);
            }
        }
    }

    image binarize_image(image im)
    {
        image c = copy_image(im);
        int i;
        for(i = 0; i < im.w * im.h * im.c; ++i){
            if(c.data[i] > .5) c.data[i] = 1;
            else c.data[i] = 0;
        }
        return c;
    }

    void saturate_image(image im, float sat)
    {
        rgb_to_hsv(im);
        scale_image_channel(im, 1, sat);
        hsv_to_rgb(im);
        constrain_image(im);
    }

    void exposure_image(image im, float sat)
    {
        rgb_to_hsv(im);
        scale_image_channel(im, 2, sat);
        hsv_to_rgb(im);
        constrain_image(im);
    }

    void saturate_exposure_image(image im, float sat, float exposure)
    {
        rgb_to_hsv(im);
        scale_image_channel(im, 1, sat);
        scale_image_channel(im, 2, exposure);
        hsv_to_rgb(im);
        constrain_image(im);
    }

    /*
       image saturate_image(image im, float sat)
       {
       image gray = grayscale_image(im);
       image blend = blend_image(im, gray, sat);
       free_image(gray);
       constrain_image(blend);
       return blend;
       }

       image brightness_image(image im, float b)
       {
       image bright = make_image(im.w, im.h, im.c);
       return bright;
       }
     */

    float bilinear_interpolate(image im, float x, float y, int c)
    {
        int ix = (int) floorf(x);
        int iy = (int) floorf(y);

        float dx = x - ix;
        float dy = y - iy;

        float val = (1-dy) * (1-dx) * get_pixel_extend(im, ix, iy, c) + 
            dy     * (1-dx) * get_pixel_extend(im, ix, iy+1, c) + 
            (1-dy) *   dx   * get_pixel_extend(im, ix+1, iy, c) +
            dy     *   dx   * get_pixel_extend(im, ix+1, iy+1, c);
        return val;
    }

    image resize_image(image im, int w, int h)
    {
        image resized = make_image(w, h, im.c);   
        image part = make_image(w, im.h, im.c);
        int r, c, k;
        float w_scale = (float)(im.w - 1) / (w - 1);
        float h_scale = (float)(im.h - 1) / (h - 1);
        for(k = 0; k < im.c; ++k){
            for(r = 0; r < im.h; ++r){
                for(c = 0; c < w; ++c){
                    float val = 0;
                    if(c == w-1 || im.w == 1){
                        val = get_pixel(im, im.w-1, r, k);
                    } else {
                        float sx = c*w_scale;
                        int ix = (int) sx;
                        float dx = sx - ix;
                        val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                    }
                    set_pixel(part, c, r, k, val);
                }
            }
        }
        for(k = 0; k < im.c; ++k){
            for(r = 0; r < h; ++r){
                float sy = r*h_scale;
                int iy = (int) sy;
                float dy = sy - iy;
                for(c = 0; c < w; ++c){
                    float val = (1-dy) * get_pixel(part, c, iy, k);
                    set_pixel(resized, c, r, k, val);
                }
                if(r == h-1 || im.h == 1) continue;
                for(c = 0; c < w; ++c){
                    float val = dy * get_pixel(part, c, iy+1, k);
                    add_pixel(resized, c, r, k, val);
                }
            }
        }

        free_image(part);
        return resized;
    }

#include "cuda.h"

    void test_resize(char *filename)
    {
        image im = load_image(filename, 0,0, 3);
        float mag = mag_array(im.data, im.w*im.h*im.c);
        printf("L2 Norm: %f\n", mag);
        image gray = grayscale_image(im);

        image sat2 = copy_image(im);
        saturate_image(sat2, 2);

        image sat5 = copy_image(im);
        saturate_image(sat5, .5);

        image exp2 = copy_image(im);
        exposure_image(exp2, 2);

        image exp5 = copy_image(im);
        exposure_image(exp5, .5);

        image bin = binarize_image(im);

#ifdef GPU
        image r = resize_image(im, im.w, im.h);
        image black = make_image(im.w*2 + 3, im.h*2 + 3, 9);
        image black2 = make_image(im.w, im.h, 3);

        float *r_gpu = cuda_make_array(r.data, r.w*r.h*r.c);
        float *black_gpu = cuda_make_array(black.data, black.w*black.h*black.c);
        float *black2_gpu = cuda_make_array(black2.data, black2.w*black2.h*black2.c);
        shortcut_gpu(3, r.w, r.h, 1, r_gpu, black.w, black.h, 3, black_gpu);
        //flip_image(r);
        //shortcut_gpu(3, r.w, r.h, 1, r.data, black.w, black.h, 3, black.data);

        shortcut_gpu(3, black.w, black.h, 3, black_gpu, black2.w, black2.h, 1, black2_gpu);
        cuda_pull_array(black_gpu, black.data, black.w*black.h*black.c);
        cuda_pull_array(black2_gpu, black2.data, black2.w*black2.h*black2.c);
        show_image_layers(black, "Black");
        show_image(black2, "Recreate");
#endif

        show_image(im,   "Original");
        show_image(bin,  "Binary");
        show_image(gray, "Gray");
        show_image(sat2, "Saturation-2");
        show_image(sat5, "Saturation-.5");
        show_image(exp2, "Exposure-2");
        show_image(exp5, "Exposure-.5");
#ifdef OPENCV
        cvWaitKey(0);
#endif
    }

#ifdef OPENCV
    image ipl_to_image(IplImage* src)
    {
        unsigned char *data = (unsigned char *)src->imageData;
        int h = src->height;
        int w = src->width;
        int c = src->nChannels;
        int step = src->widthStep;
        image out = make_image(w, h, c);
        int i, j, k, count=0;;

        for(k= 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    out.data[count++] = data[i*step + j*c + k]/255.;
                }
            }
        }
        return out;
    }

    image load_image_cv(char *filename, int channels)
    {
        IplImage* src = 0;
        int flag = -1;
        if (channels == 0) flag = -1;
        else if (channels == 1) flag = 0;
        else if (channels == 3) flag = 1;
        else {
            fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
        }

        if( (src = cvLoadImage(filename, flag)) == 0 )
        {
            fprintf(stderr, "Cannot load image \"%s\"\n", filename);
            char buff[256];
            sprintf(buff, "echo %s >> bad.list", filename);
            system(buff);
            return make_image(10,10,3);
            //exit(0);
        }
        image out = ipl_to_image(src);
        cvReleaseImage(&src);
        rgbgr_image(out);
        return out;
    }

#endif


    image load_image_stb(char *filename, int channels)
    {
        int w, h, c;
        unsigned char *data = stbi_load(filename, &w, &h, &c, channels);
        if (!data) {
            fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
            exit(0);
        }
        if(channels) c = channels;
        int i,j,k;
        image im = make_image(w, h, c);
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int dst_index = i + w*j + w*h*k;
                    int src_index = k + c*i + c*w*j;
                    im.data[dst_index] = (float)data[src_index]/255.;
                }
            }
        }
        free(data);
        return im;
    }

    image load_image(char *filename, int w, int h, int c)
    {
#ifdef OPENCV
        image out = load_image_cv(filename, c);
#else
        image out = load_image_stb(filename, c);
#endif

        if((h && w) && (h != out.h || w != out.w)){
            image resized = resize_image(out, w, h);
            free_image(out);
            out = resized;
        }
        return out;
    }

    image load_image_color(char *filename, int w, int h)
    {
        return load_image(filename, w, h, 3);
    }

    image get_image_layer(image m, int l)
    {
        image out = make_image(m.w, m.h, 1);
        int i;
        for(i = 0; i < m.h*m.w; ++i){
            out.data[i] = m.data[i+l*m.h*m.w];
        }
        return out;
    }

    float get_pixel(image m, int x, int y, int c)
    {
        assert(x < m.w && y < m.h && c < m.c);
        return m.data[c*m.h*m.w + y*m.w + x];
    }
    float get_pixel_extend(image m, int x, int y, int c)
    {
        if(x < 0 || x >= m.w || y < 0 || y >= m.h || c < 0 || c >= m.c) return 0;
        return get_pixel(m, x, y, c);
    }
    void set_pixel(image m, int x, int y, int c, float val)
    {
        assert(x < m.w && y < m.h && c < m.c);
        m.data[c*m.h*m.w + y*m.w + x] = val;
    }
    void add_pixel(image m, int x, int y, int c, float val)
    {
        assert(x < m.w && y < m.h && c < m.c);
        m.data[c*m.h*m.w + y*m.w + x] += val;
    }

    void print_image(image m)
    {
        int i, j, k;
        for(i =0 ; i < m.c; ++i){
            for(j =0 ; j < m.h; ++j){
                for(k = 0; k < m.w; ++k){
                    printf("%.2lf, ", m.data[i*m.h*m.w + j*m.w + k]);
                    if(k > 30) break;
                }
                printf("\n");
                if(j > 30) break;
            }
            printf("\n");
        }
        printf("\n");
    }

    image collapse_images_vert(image *ims, int n)
    {
        int color = 1;
        int border = 1;
        int h,w,c;
        w = ims[0].w;
        h = (ims[0].h + border) * n - border;
        c = ims[0].c;
        if(c != 3 || !color){
            w = (w+border)*c - border;
            c = 1;
        }

        image filters = make_image(w, h, c);
        int i,j;
        for(i = 0; i < n; ++i){
            int h_offset = i*(ims[0].h+border);
            image copy = copy_image(ims[i]);
            //normalize_image(copy);
            if(c == 3 && color){
                embed_image(copy, filters, 0, h_offset);
            }
            else{
                for(j = 0; j < copy.c; ++j){
                    int w_offset = j*(ims[0].w+border);
                    image layer = get_image_layer(copy, j);
                    embed_image(layer, filters, w_offset, h_offset);
                    free_image(layer);
                }
            }
            free_image(copy);
        }
        return filters;
    } 

    image collapse_images_horz(image *ims, int n)
    {
        int color = 1;
        int border = 1;
        int h,w,c;
        int size = ims[0].h;
        h = size;
        w = (ims[0].w + border) * n - border;
        c = ims[0].c;
        if(c != 3 || !color){
            h = (h+border)*c - border;
            c = 1;
        }

        image filters = make_image(w, h, c);
        int i,j;
        for(i = 0; i < n; ++i){
            int w_offset = i*(size+border);
            image copy = copy_image(ims[i]);
            //normalize_image(copy);
            if(c == 3 && color){
                embed_image(copy, filters, w_offset, 0);
            }
            else{
                for(j = 0; j < copy.c; ++j){
                    int h_offset = j*(size+border);
                    image layer = get_image_layer(copy, j);
                    embed_image(layer, filters, w_offset, h_offset);
                    free_image(layer);
                }
            }
            free_image(copy);
        }
        return filters;
    } 

    void show_image_normalized(image im, const char *name)
    {
        image c = copy_image(im);
        normalize_image(c);
        show_image(c, name);
        free_image(c);
    }

    void show_images(image *ims, int n, char *window)
    {
        image m = collapse_images_vert(ims, n);
        /*
           int w = 448;
           int h = ((float)m.h/m.w) * 448;
           if(h > 896){
           h = 896;
           w = ((float)m.w/m.h) * 896;
           }
           image sized = resize_image(m, w, h);
         */
        normalize_image(m);
        image sized = resize_image(m, m.w, m.h);
        save_image(sized, window);
        show_image(sized, window);
        free_image(sized);
        free_image(m);
    }

    void free_image(image m)
    {
        free(m.data);
    }
