#include "image.h"
#include "utils.h"
#include <stdio.h>

int windows = 0;

void draw_box(image a, int x1, int y1, int x2, int y2)
{
    int i, c;
    for(c = 0; c < a.c; ++c){
        for(i = x1; i < x2; ++i){
            a.data[i + y1*a.w + c*a.w*a.h] = (c==0)?1:-1;
            a.data[i + y2*a.w + c*a.w*a.h] = (c==0)?1:-1;
        }
    }
    for(c = 0; c < a.c; ++c){
        for(i = y1; i < y2; ++i){
            a.data[x1 + i*a.w + c*a.w*a.h] = (c==0)?1:-1;
            a.data[x2 + i*a.w + c*a.w*a.h] = (c==0)?1:-1;
        }
    }
}

image image_distance(image a, image b)
{
    int i,j;
    image dist = make_image(a.h, a.w, 1);
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

void subtract_image(image a, image b)
{
    int i;
    for(i = 0; i < a.h*a.w*a.c; ++i) a.data[i] -= b.data[i];
}

void embed_image(image source, image dest, int h, int w)
{
    int i,j,k;
    for(k = 0; k < source.c; ++k){
        for(i = 0; i < source.h; ++i){
            for(j = 0; j < source.w; ++j){
                float val = get_pixel(source, i,j,k);
                set_pixel(dest, h+i, w+j, k, val);
            }
        }
    }
}

image collapse_image_layers(image source, int border)
{
    int h = source.h;
    h = (h+border)*source.c - border;
    image dest = make_image(h, source.w, 1);
    int i;
    for(i = 0; i < source.c; ++i){
        image layer = get_image_layer(source, i);
        int h_offset = i*(source.h+border);
        embed_image(layer, dest, h_offset, 0);
        free_image(layer);
    }
    return dest;
}

void z_normalize_image(image p)
{
    normalize_array(p.data, p.h*p.w*p.c);
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

float avg_image_layer(image m, int l)
{
    int i;
    float sum = 0;
    for(i = 0; i < m.h*m.w; ++i){
        sum += m.data[l*m.h*m.w + i];
    }
    return sum/(m.h*m.w);
}

void threshold_image(image p, float t)
{
    int i;
    for(i = 0; i < p.w*p.h*p.c; ++i){
        if(p.data[i] < t) p.data[i] = 0;
    }
}

image copy_image(image p)
{
    image copy = p;
    copy.data = calloc(p.h*p.w*p.c, sizeof(float));
    memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));
    return copy;
}


void show_image(image p, char *name)
{
    int i,j,k;
    image copy = copy_image(p);
    normalize_image(copy);

    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    sprintf(buff, "%s", name);

    IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
    int step = disp->widthStep;
    cvNamedWindow(buff, CV_WINDOW_AUTOSIZE); 
    //cvMoveWindow(buff, 100*(windows%10) + 200*(windows/10), 100*(windows%10));
    ++windows;
    for(i = 0; i < p.h; ++i){
        for(j = 0; j < p.w; ++j){
            for(k= 0; k < p.c; ++k){
                disp->imageData[i*step + j*p.c + k] = (unsigned char)(get_pixel(copy,i,j,k)*255);
            }
        }
    }
    free_image(copy);
    if(disp->height < 500 || disp->width < 500 || disp->height > 1000){
        int w = 500;
        int h = w*p.h/p.w;
        if(h > 1000){
            h = 1000;
            w = h*p.w/p.h;
        }
        IplImage *buffer = disp;
        disp = cvCreateImage(cvSize(w, h), buffer->depth, buffer->nChannels);
        cvResize(buffer, disp, CV_INTER_NN);
        cvReleaseImage(&buffer);
    }
    cvShowImage(buff, disp);
    cvReleaseImage(&disp);
}

void save_image(image p, char *name)
{
    int i,j,k;
    image copy = copy_image(p);
    normalize_image(copy);

    char buff[256];
    //sprintf(buff, "%s (%d)", name, windows);
    sprintf(buff, "%s.png", name);

    IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
    int step = disp->widthStep;
    for(i = 0; i < p.h; ++i){
        for(j = 0; j < p.w; ++j){
            for(k= 0; k < p.c; ++k){
                disp->imageData[i*step + j*p.c + k] = (unsigned char)(get_pixel(copy,i,j,k)*255);
            }
        }
    }
    free_image(copy);
    cvSaveImage(buff, disp,0);
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

void show_image_collapsed(image p, char *name)
{
    image c = collapse_image_layers(p, 1);
    show_image(c, name);
    free_image(c);
}

image make_empty_image(int h, int w, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int h, int w, int c)
{
    image out = make_empty_image(h,w,c);
    out.data = calloc(h*w*c, sizeof(float));
    return out;
}

image float_to_image(int h, int w, int c, float *data)
{
    image out = make_empty_image(h,w,c);
    out.data = data;
    return out;
}

void zero_image(image m)
{
    memset(m.data, 0, m.h*m.w*m.c*sizeof(float));
}

void zero_channel(image m, int c)
{
    memset(&(m.data[c*m.h*m.w]), 0, m.h*m.w*sizeof(float));
}

void rotate_image(image m)
{
    int i,j;
    for(j = 0; j < m.c; ++j){
        for(i = 0; i < m.h*m.w/2; ++i){
            float swap = m.data[j*m.h*m.w + i];
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
        out.data[i] = rand_normal();
        //out.data[i] = rand()%3;
    }
    return out;
}

void add_into_image(image src, image dest, int h, int w)
{
    int i,j,k;
    for(k = 0; k < src.c; ++k){
        for(i = 0; i < src.h; ++i){
            for(j = 0; j < src.w; ++j){
                add_pixel(dest, h+i, w+j, k, get_pixel(src, i, j, k));
            }
        }
    }
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

image make_random_kernel(int size, int c, float scale)
{
    int pad;
    if((pad=(size%2==0))) ++size;
    image out = make_random_image(size,size,c);
    scale_image(out, scale);
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

// Returns a new image that is a cropped version (rectangular cut-out)
// of the original image.
IplImage* cropImage(const IplImage *img, const CvRect region)
{
    IplImage *imageCropped;
    CvSize size;

    if (img->width <= 0 || img->height <= 0
            || region.width <= 0 || region.height <= 0) {
        //cerr << "ERROR in cropImage(): invalid dimensions." << endl;
        exit(1);
    }

    if (img->depth != IPL_DEPTH_8U) {
        //cerr << "ERROR in cropImage(): image depth is not 8." << endl;
        exit(1);
    }

    // Set the desired region of interest.
    cvSetImageROI((IplImage*)img, region);
    // Copy region of interest into a new iplImage and return it.
    size.width = region.width;
    size.height = region.height;
    imageCropped = cvCreateImage(size, IPL_DEPTH_8U, img->nChannels);
    cvCopy(img, imageCropped,NULL);  // Copy just the region.

    return imageCropped;
}

// Creates a new image copy that is of a desired size. The aspect ratio will
// be kept constant if 'keepAspectRatio' is true, by cropping undesired parts
// so that only pixels of the original image are shown, instead of adding
// extra blank space.
// Remember to free the new image later.
IplImage* resizeImage(const IplImage *origImg, int newHeight, int newWidth,
        int keepAspectRatio)
{
    IplImage *outImg = 0;
    int origWidth = 0;
    int origHeight = 0;
    if (origImg) {
        origWidth = origImg->width;
        origHeight = origImg->height;
    }
    if (newWidth <= 0 || newHeight <= 0 || origImg == 0
            || origWidth <= 0 || origHeight <= 0) {
        //cerr << "ERROR: Bad desired image size of " << newWidth
        //  << "x" << newHeight << " in resizeImage().\n";
        exit(1);
    }

    if (keepAspectRatio) {
        // Resize the image without changing its aspect ratio,
        // by cropping off the edges and enlarging the middle section.
        CvRect r;
        // input aspect ratio
        float origAspect = (origWidth / (float)origHeight);
        // output aspect ratio
        float newAspect = (newWidth / (float)newHeight);
        // crop width to be origHeight * newAspect
        if (origAspect > newAspect) {
            int tw = (origHeight * newWidth) / newHeight;
            r = cvRect((origWidth - tw)/2, 0, tw, origHeight);
        }
        else {  // crop height to be origWidth / newAspect
            int th = (origWidth * newHeight) / newWidth;
            r = cvRect(0, (origHeight - th)/2, origWidth, th);
        }
        IplImage *croppedImg = cropImage(origImg, r);

        // Call this function again, with the new aspect ratio image.
        // Will do a scaled image resize with the correct aspect ratio.
        outImg = resizeImage(croppedImg, newHeight, newWidth, 0);
        cvReleaseImage( &croppedImg );
    }
    else {

        // Scale the image to the new dimensions,
        // even if the aspect ratio will be changed.
        outImg = cvCreateImage(cvSize(newWidth, newHeight),
                origImg->depth, origImg->nChannels);
        if (newWidth > origImg->width && newHeight > origImg->height) {
            // Make the image larger
            cvResetImageROI((IplImage*)origImg);
            // CV_INTER_LINEAR: good at enlarging.
            // CV_INTER_CUBIC: good at enlarging.           
            cvResize(origImg, outImg, CV_INTER_LINEAR);
        }
        else {
            // Make the image smaller
            cvResetImageROI((IplImage*)origImg);
            // CV_INTER_AREA: good at shrinking (decimation) only.
            cvResize(origImg, outImg, CV_INTER_AREA);
        }

    }
    return outImg;
}

image ipl_to_image(IplImage* src)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
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
    return out;
}

image load_image_color(char *filename, int h, int w)
{
    IplImage* src = 0;
    if( (src = cvLoadImage(filename, 1)) == 0 )
    {
        printf("Cannot load file image %s\n", filename);
        exit(0);
    }
    if(h && w && (src->height != h || src->width != w)){
        //printf("Resized!\n");
        IplImage *resized = resizeImage(src, h, w, 0);
        cvReleaseImage(&src);
        src = resized;
    }
    image out = ipl_to_image(src);
    cvReleaseImage(&src);
    return out;
}

image load_image(char *filename, int h, int w)
{
    IplImage* src = 0;
    if( (src = cvLoadImage(filename,-1)) == 0 )
    {
        printf("Cannot load file image %s\n", filename);
        exit(0);
    }
    if(h && w ){
        IplImage *resized = resizeImage(src, h, w, 1);
        cvReleaseImage(&src);
        src = resized;
    }
    image out = ipl_to_image(src);
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
image get_sub_image(image m, int h, int w, int dh, int dw)
{
    image out = make_image(dh, dw, m.c);
    int i,j,k;
    for(k = 0; k < out.c; ++k){
        for(i = 0; i < dh; ++i){
            for(j = 0; j < dw; ++j){
                float val = get_pixel(m, h+i, w+j, k);
                set_pixel(out, i, j, k, val);
            }
        }
    }
    return out;
}

float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.h && y < m.w && c < m.c);
    return m.data[c*m.h*m.w + x*m.w + y];
}
float get_pixel_extend(image m, int x, int y, int c)
{
    if(x < 0 || x >= m.h || y < 0 || y >= m.w || c < 0 || c >= m.c) return 0;
    return get_pixel(m, x, y, c);
}
void set_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.h && y < m.w && c < m.c);
    m.data[c*m.h*m.w + x*m.w + y] = val;
}
void set_pixel_extend(image m, int x, int y, int c, float val)
{
    if(x < 0 || x >= m.h || y < 0 || y >= m.w || c < 0 || c >= m.c) return;
    set_pixel(m, x, y, c, val);
}

void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.h && y < m.w && c < m.c);
    m.data[c*m.h*m.w + x*m.w + y] += val;
}

void add_pixel_extend(image m, int x, int y, int c, float val)
{
    if(x < 0 || x >= m.h || y < 0 || y >= m.w || c < 0 || c >= m.c) return;
    add_pixel(m, x, y, c, val);
}

void two_d_convolve(image m, int mc, image kernel, int kc, int stride, image out, int oc, int edge)
{
    int x,y,i,j;
    int xstart, xend, ystart, yend;
    if(edge){
        xstart = ystart = 0;
        xend = m.h;
        yend = m.w;
    }else{
        xstart = kernel.h/2;
        ystart = kernel.w/2;
        xend = m.h-kernel.h/2;
        yend = m.w - kernel.w/2;
    }
    for(x = xstart; x < xend; x += stride){
        for(y = ystart; y < yend; y += stride){
            float sum = 0;
            for(i = 0; i < kernel.h; ++i){
                for(j = 0; j < kernel.w; ++j){
                    sum += get_pixel(kernel, i, j, kc)*get_pixel_extend(m, x+i-kernel.h/2, y+j-kernel.w/2, mc);
                }
            }
            add_pixel(out, (x-xstart)/stride, (y-ystart)/stride, oc, sum);
        }
    }
}

float single_convolve(image m, image kernel, int x, int y)
{
    float sum = 0;
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

void convolve(image m, image kernel, int stride, int channel, image out, int edge)
{
    assert(m.c == kernel.c);
    int i;
    zero_channel(out, channel);
    for(i = 0; i < m.c; ++i){
        two_d_convolve(m, i, kernel, i, stride, out, channel, edge);
    }
    /*
       int j;
       for(i = 0; i < m.h; i += stride){
       for(j = 0; j < m.w; j += stride){
       float val = single_convolve(m, kernel, i, j);
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
                float val = get_pixel(m, i, j, k);
                set_pixel(out, i*stride, j*stride, k, val);
            }
        }
    }
}

void single_update(image m, image update, int x, int y, float error)
{
    int i, j, k;
    for(i = 0; i < update.h; ++i){
        for(j = 0; j < update.w; ++j){
            for(k = 0; k < update.c; ++k){
                float val = get_pixel_extend(m, x+i-update.h/2, y+j-update.w/2, k);
                add_pixel(update, i, j, k, val*error);
            }
        }
    }
}

void kernel_update(image m, image update, int stride, int channel, image out, int edge)
{
    assert(m.c == update.c);
    zero_image(update);
    int i, j, istart, jstart, iend, jend;
    if(edge){
        istart = jstart = 0;
        iend = m.h;
        jend = m.w;
    }else{
        istart = update.h/2;
        jstart = update.w/2;
        iend = m.h-update.h/2;
        jend = m.w - update.w/2;
    }
    for(i = istart; i < iend; i += stride){
        for(j = jstart; j < jend; j += stride){
            float error = get_pixel(out, (i-istart)/stride, (j-jstart)/stride, channel);
            single_update(m, update, i, j, error);
        }
    }
    /*
       for(i = 0; i < update.h*update.w*update.c; ++i){
       update.data[i] /= (m.h/stride)*(m.w/stride);
       }
     */
}

void single_back_convolve(image m, image kernel, int x, int y, float val)
{
    int i, j, k;
    for(i = 0; i < kernel.h; ++i){
        for(j = 0; j < kernel.w; ++j){
            for(k = 0; k < kernel.c; ++k){
                float pval = get_pixel(kernel, i, j, k) * val;
                add_pixel_extend(m, x+i-kernel.h/2, y+j-kernel.w/2, k, pval);
            }
        }
    }
}

void back_convolve(image m, image kernel, int stride, int channel, image out, int edge)
{
    assert(m.c == kernel.c);
    int i, j, istart, jstart, iend, jend;
    if(edge){
        istart = jstart = 0;
        iend = m.h;
        jend = m.w;
    }else{
        istart = kernel.h/2;
        jstart = kernel.w/2;
        iend = m.h-kernel.h/2;
        jend = m.w - kernel.w/2;
    }
    for(i = istart; i < iend; i += stride){
        for(j = jstart; j < jend; j += stride){
            float val = get_pixel(out, (i-istart)/stride, (j-jstart)/stride, channel);
            single_back_convolve(m, kernel, i, j, val);
        }
    }
}

void print_image(image m)
{
    int i;
    for(i =0 ; i < m.h*m.w*m.c; ++i) printf("%lf, ", m.data[i]);
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

    image filters = make_image(h,w,c);
    int i,j;
    for(i = 0; i < n; ++i){
        int h_offset = i*(ims[0].h+border);
        image copy = copy_image(ims[i]);
        //normalize_image(copy);
        if(c == 3 && color){
            embed_image(copy, filters, h_offset, 0);
        }
        else{
            for(j = 0; j < copy.c; ++j){
                int w_offset = j*(ims[0].w+border);
                image layer = get_image_layer(copy, j);
                embed_image(layer, filters, h_offset, w_offset);
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

    image filters = make_image(h,w,c);
    int i,j;
    for(i = 0; i < n; ++i){
        int w_offset = i*(size+border);
        image copy = copy_image(ims[i]);
        //normalize_image(copy);
        if(c == 3 && color){
            embed_image(copy, filters, 0, w_offset);
        }
        else{
            for(j = 0; j < copy.c; ++j){
                int h_offset = j*(size+border);
                image layer = get_image_layer(copy, j);
                embed_image(layer, filters, h_offset, w_offset);
                free_image(layer);
            }
        }
        free_image(copy);
    }
    return filters;
} 

void show_images(image *ims, int n, char *window)
{
    image m = collapse_images_vert(ims, n);
    save_image(m, window);
    show_image(m, window);
    free_image(m);
}

image grid_images(image **ims, int h, int w)
{
    int i;
    image *rows = calloc(h, sizeof(image));
    for(i = 0; i < h; ++i){
        rows[i] = collapse_images_horz(ims[i], w);
    }
    image out = collapse_images_vert(rows, h);
    for(i = 0; i < h; ++i){
        free_image(rows[i]);
    }
    free(rows);
    return out;
}

void test_grid()
{
    int i,j;
    int num = 3;
    int topk = 3;
    image **vizs = calloc(num, sizeof(image*));
    for(i = 0; i < num; ++i){
        vizs[i] = calloc(topk, sizeof(image));
        for(j = 0; j < topk; ++j) vizs[i][j] = make_image(3,3,3);
    }
    image grid = grid_images(vizs, num, topk);
    save_image(grid, "Test Grid");
    free_image(grid);
}

void show_images_grid(image **ims, int h, int w, char *window)
{
    image out = grid_images(ims, h, w);
    show_image(out, window);
    free_image(out);
}

void free_image(image m)
{
    free(m.data);
}
