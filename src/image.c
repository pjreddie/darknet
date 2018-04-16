#include "image.h"
#include "utils.h"
#include "blas.h"
#include "cuda.h"
#include <stdio.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/version.hpp"
#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio_c.h"
#include "opencv2/imgcodecs/imgcodecs_c.h"
#include "http_stream.h"
#endif
#include "http_stream.h"
#endif

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

void composite_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float val = get_pixel(source, x, y, k);
                float val2 = get_pixel_extend(dest, dx+x, dy+y, k);
                set_pixel(dest, dx+x, dy+y, k, val * val2);
            }
        }
    }
}

image border_image(image a, int border)
{
    image b = make_image(a.w + 2*border, a.h + 2*border, a.c);
    int x,y,k;
    for(k = 0; k < b.c; ++k){
        for(y = 0; y < b.h; ++y){
            for(x = 0; x < b.w; ++x){
                float val = get_pixel_extend(a, x - border, y - border, k);
                if(x - border < 0 || x - border >= a.w || y - border < 0 || y - border >= a.h) val = 1;
                set_pixel(b, x, y, k, val);
            }
        }
    }
    return b;
}

image tile_images(image a, image b, int dx)
{
    if(a.w == 0) return copy_image(b);
    image c = make_image(a.w + b.w + dx, (a.h > b.h) ? a.h : b.h, (a.c > b.c) ? a.c : b.c);
    fill_cpu(c.w*c.h*c.c, 1, c.data, 1);
    embed_image(a, c, 0, 0); 
    composite_image(b, c, a.w + dx, 0);
    return c;
}

image get_label(image **characters, char *string, int size)
{
    if(size > 7) size = 7;
    image label = make_empty_image(0,0,0);
    while(*string){
        image l = characters[size][(int)*string];
        image n = tile_images(label, l, -size - 1 + (size+1)/2);
        free_image(label);
        label = n;
        ++string;
    }
    image b = border_image(label, label.h*.25);
    free_image(label);
    return b;
}

image get_label_v3(image **characters, char *string, int size)
{
	size = size / 10;
	if (size > 7) size = 7;
	image label = make_empty_image(0, 0, 0);
	while (*string) {
		image l = characters[size][(int)*string];
		image n = tile_images(label, l, -size - 1 + (size + 1) / 2);
		free_image(label);
		label = n;
		++string;
	}
	image b = border_image(label, label.h*.25);
	free_image(label);
	return b;
}

void draw_label(image a, int r, int c, image label, const float *rgb)
{
    int w = label.w;
    int h = label.h;
    if (r - h >= 0) r = r - h;

    int i, j, k;
    for(j = 0; j < h && j + r < a.h; ++j){
        for(i = 0; i < w && i + c < a.w; ++i){
            for(k = 0; k < label.c; ++k){
                float val = get_pixel(label, i, j, k);
                set_pixel(a, i+c, j+r, k, rgb[k] * val);
            }
        }
    }
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

image **load_alphabet()
{
    int i, j;
    const int nsize = 8;
    image **alphabets = calloc(nsize, sizeof(image));
    for(j = 0; j < nsize; ++j){
        alphabets[j] = calloc(128, sizeof(image));
        for(i = 32; i < 127; ++i){
            char buff[256];
            sprintf(buff, "data/labels/%d_%d.png", i, j);
            alphabets[j][i] = load_image_color(buff, 0, 0);
        }
    }
    return alphabets;
}

void draw_detections_v3(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes)
{
	int i, j;

	for (i = 0; i < num; ++i) {
		char labelstr[4096] = { 0 };
		int class_id = -1;
		for (j = 0; j < classes; ++j) {
			if (dets[i].prob[j] > thresh) {
				if (class_id < 0) {
					strcat(labelstr, names[j]);
					class_id = j;
				}
				else {
					strcat(labelstr, ", ");
					strcat(labelstr, names[j]);
				}
				printf("%s: %.0f%%\n", names[j], dets[i].prob[j] * 100);
			}
		}
		if (class_id >= 0) {
			int width = im.h * .006;

			/*
			if(0){
			width = pow(prob, 1./2.)*10+1;
			alphabet = 0;
			}
			*/

			//printf("%d %s: %.0f%%\n", i, names[class_id], prob*100);
			int offset = class_id * 123457 % classes;
			float red = get_color(2, offset, classes);
			float green = get_color(1, offset, classes);
			float blue = get_color(0, offset, classes);
			float rgb[3];

			//width = prob*20+2;

			rgb[0] = red;
			rgb[1] = green;
			rgb[2] = blue;
			box b = dets[i].bbox;
			//printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

			int left = (b.x - b.w / 2.)*im.w;
			int right = (b.x + b.w / 2.)*im.w;
			int top = (b.y - b.h / 2.)*im.h;
			int bot = (b.y + b.h / 2.)*im.h;

			if (left < 0) left = 0;
			if (right > im.w - 1) right = im.w - 1;
			if (top < 0) top = 0;
			if (bot > im.h - 1) bot = im.h - 1;

			//int b_x_center = (left + right) / 2;
			//int b_y_center = (top + bot) / 2;
			//int b_width = right - left;
			//int b_height = bot - top;
			//sprintf(labelstr, "%d x %d - w: %d, h: %d", b_x_center, b_y_center, b_width, b_height);

			draw_box_width(im, left, top, right, bot, width, red, green, blue);
			if (alphabet) {
				image label = get_label_v3(alphabet, labelstr, (im.h*.03));
				draw_label(im, top + width, left, label, rgb);
				free_image(label);
			}
			if (dets[i].mask) {
				image mask = float_to_image(14, 14, 1, dets[i].mask);
				image resized_mask = resize_image(mask, b.w*im.w, b.h*im.h);
				image tmask = threshold_image(resized_mask, .5);
				embed_image(tmask, im, left, top);
				free_image(mask);
				free_image(resized_mask);
				free_image(tmask);
			}
		}
	}
}

void draw_detections(image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes)
{
    int i;

    for(i = 0; i < num; ++i){
        int class_id = max_index(probs[i], classes);
        float prob = probs[i][class_id];
        if(prob > thresh){

			//// for comparison with OpenCV version of DNN Darknet Yolo v2
			//printf("\n %f, %f, %f, %f, ", boxes[i].x, boxes[i].y, boxes[i].w, boxes[i].h);
			// int k;
			//for (k = 0; k < classes; ++k) {
			//	printf("%f, ", probs[i][k]);
			//}
			//printf("\n");

            int width = im.h * .012;

            if(0){
                width = pow(prob, 1./2.)*10+1;
                alphabet = 0;
            }

            int offset = class_id*123457 % classes;
            float red = get_color(2,offset,classes);
            float green = get_color(1,offset,classes);
            float blue = get_color(0,offset,classes);
            float rgb[3];

            //width = prob*20+2;

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
			printf("%s: %.0f%%", names[class_id], prob * 100);
			
			//printf(" - id: %d, x_center: %d, y_center: %d, width: %d, height: %d",
			//	class_id, (right + left) / 2, (bot - top) / 2, right - left, bot - top);

			printf("\n");
            draw_box_width(im, left, top, right, bot, width, red, green, blue);
            if (alphabet) {
                image label = get_label(alphabet, names[class_id], (im.h*.03)/10);
                draw_label(im, top + width, left, label, rgb);
            }
        }
    }
}

#ifdef OPENCV

void draw_detections_cv_v3(IplImage* show_img, detection *dets, int num, float thresh, char **names, image **alphabet, int classes)
{
	int i, j;
	if (!show_img) return;

	for (i = 0; i < num; ++i) {
		char labelstr[4096] = { 0 };
		int class_id = -1;
		for (j = 0; j < classes; ++j) {
			if (dets[i].prob[j] > thresh) {
				if (class_id < 0) {
					strcat(labelstr, names[j]);
					class_id = j;
				}
				else {
					strcat(labelstr, ", ");
					strcat(labelstr, names[j]);
				}
				printf("%s: %.0f%%\n", names[j], dets[i].prob[j] * 100);
			}
		}
		if (class_id >= 0) {
			int width = show_img->height * .006;

			/*
			if(0){
			width = pow(prob, 1./2.)*10+1;
			alphabet = 0;
			}
			*/

			//printf("%d %s: %.0f%%\n", i, names[class_id], prob*100);
			int offset = class_id * 123457 % classes;
			float red = get_color(2, offset, classes);
			float green = get_color(1, offset, classes);
			float blue = get_color(0, offset, classes);
			float rgb[3];

			//width = prob*20+2;

			rgb[0] = red;
			rgb[1] = green;
			rgb[2] = blue;
			box b = dets[i].bbox;
			//printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

			int left = (b.x - b.w / 2.)*show_img->width;
			int right = (b.x + b.w / 2.)*show_img->width;
			int top = (b.y - b.h / 2.)*show_img->height;
			int bot = (b.y + b.h / 2.)*show_img->height;

			if (left < 0) left = 0;
			if (right > show_img->width - 1) right = show_img->width - 1;
			if (top < 0) top = 0;
			if (bot > show_img->height - 1) bot = show_img->height - 1;

			//int b_x_center = (left + right) / 2;
			//int b_y_center = (top + bot) / 2;
			//int b_width = right - left;
			//int b_height = bot - top;
			//sprintf(labelstr, "%d x %d - w: %d, h: %d", b_x_center, b_y_center, b_width, b_height);

			float const font_size = show_img->height / 1000.F;
			CvPoint pt1, pt2, pt_text, pt_text_bg1, pt_text_bg2;
			pt1.x = left;
			pt1.y = top;
			pt2.x = right;
			pt2.y = bot;
			pt_text.x = left;
			pt_text.y = top - 12;
			pt_text_bg1.x = left;
			pt_text_bg1.y = top - (10 + 25 * font_size);
			pt_text_bg2.x = right;
			pt_text_bg2.y = top;
			CvScalar color;
			color.val[0] = red * 256;
			color.val[1] = green * 256;
			color.val[2] = blue * 256;

			cvRectangle(show_img, pt1, pt2, color, width, 8, 0);
			//printf("left=%d, right=%d, top=%d, bottom=%d, obj_id=%d, obj=%s \n", left, right, top, bot, class_id, names[class_id]);
			cvRectangle(show_img, pt_text_bg1, pt_text_bg2, color, width, 8, 0);
			cvRectangle(show_img, pt_text_bg1, pt_text_bg2, color, CV_FILLED, 8, 0);	// filled
			CvScalar black_color;
			black_color.val[0] = 0;
			CvFont font;
			cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, font_size, font_size, 0, font_size * 3, 8);
			cvPutText(show_img, labelstr, pt_text, &font, black_color);
		}
	}
}

void draw_detections_cv(IplImage* show_img, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes)
{
	int i;

	for (i = 0; i < num; ++i) {
		int class_id = max_index(probs[i], classes);
		float prob = probs[i][class_id];
		if (prob > thresh) {

			int width = show_img->height * .012;

			if (0) {
				width = pow(prob, 1. / 2.) * 10 + 1;
				alphabet = 0;
			}

			printf("%s: %.0f%%\n", names[class_id], prob * 100);
			int offset = class_id * 123457 % classes;
			float red = get_color(2, offset, classes);
			float green = get_color(1, offset, classes);
			float blue = get_color(0, offset, classes);
			float rgb[3];

			//width = prob*20+2;

			rgb[0] = red;
			rgb[1] = green;
			rgb[2] = blue;
			box b = boxes[i];

			int left = (b.x - b.w / 2.)*show_img->width;
			int right = (b.x + b.w / 2.)*show_img->width;
			int top = (b.y - b.h / 2.)*show_img->height;
			int bot = (b.y + b.h / 2.)*show_img->height;

			if (left < 0) left = 0;
			if (right > show_img->width - 1) right = show_img->width - 1;
			if (top < 0) top = 0;
			if (bot > show_img->height - 1) bot = show_img->height - 1;

			float const font_size = show_img->height / 1000.F;
			CvPoint pt1, pt2, pt_text, pt_text_bg1, pt_text_bg2;
			pt1.x = left;
			pt1.y = top;
			pt2.x = right;
			pt2.y = bot;
			pt_text.x = left;
			pt_text.y = top - 12;
			pt_text_bg1.x = left;
			pt_text_bg1.y = top - (10+25*font_size);
			pt_text_bg2.x = right;
			pt_text_bg2.y = top;
			CvScalar color;
			color.val[0] = red * 256;
			color.val[1] = green * 256;
			color.val[2] = blue * 256;

			cvRectangle(show_img, pt1, pt2, color, width, 8, 0);
			//printf("left=%d, right=%d, top=%d, bottom=%d, obj_id=%d, obj=%s \n", left, right, top, bot, class_id, names[class_id]);
			cvRectangle(show_img, pt_text_bg1, pt_text_bg2, color, width, 8, 0);
			cvRectangle(show_img, pt_text_bg1, pt_text_bg2, color, CV_FILLED, 8, 0);	// filled
			CvScalar black_color;
			black_color.val[0] = 0;
			CvFont font;
			cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, font_size, font_size, 0, font_size * 3, 8);	
			cvPutText(show_img, names[class_id], pt_text, &font, black_color);
		}
	}
}

IplImage* draw_train_chart(float max_img_loss, int max_batches, int number_of_lines, int img_size)
{
	int img_offset = 50;
	int draw_size = img_size - img_offset;
	IplImage* img = cvCreateImage(cvSize(img_size, img_size), 8, 3);
	cvSet(img, CV_RGB(255, 255, 255), 0);
	CvPoint pt1, pt2, pt_text;
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX_SMALL, 0.7, 0.7, 0, 1, CV_AA);
	char char_buff[100];
	int i;
	// vertical lines
	pt1.x = img_offset; pt2.x = img_size, pt_text.x = 10;
	for (i = 1; i <= number_of_lines; ++i) {
		pt1.y = pt2.y = (float)i * draw_size / number_of_lines;
		cvLine(img, pt1, pt2, CV_RGB(224, 224, 224), 1, 8, 0);
		if (i % 10 == 0) {
			sprintf(char_buff, "%2.1f", max_img_loss*(number_of_lines - i) / number_of_lines);
			pt_text.y = pt1.y + 5;
			cvPutText(img, char_buff, pt_text, &font, CV_RGB(0, 0, 0));
			cvLine(img, pt1, pt2, CV_RGB(128, 128, 128), 1, 8, 0);
		}
	}
	// horizontal lines
	pt1.y = draw_size; pt2.y = 0, pt_text.y = draw_size + 15;
	for (i = 0; i <= number_of_lines; ++i) {
		pt1.x = pt2.x = img_offset + (float)i * draw_size / number_of_lines;
		cvLine(img, pt1, pt2, CV_RGB(224, 224, 224), 1, 8, 0);
		if (i % 10 == 0) {
			sprintf(char_buff, "%d", max_batches * i / number_of_lines);
			pt_text.x = pt1.x - 20;
			cvPutText(img, char_buff, pt_text, &font, CV_RGB(0, 0, 0));
			cvLine(img, pt1, pt2, CV_RGB(128, 128, 128), 1, 8, 0);
		}
	}
	cvPutText(img, "Iteration number", cvPoint(draw_size / 2, img_size - 10), &font, CV_RGB(0, 0, 0));
	cvPutText(img, "Press 's' to save: chart.jpg", cvPoint(5, img_size - 10), &font, CV_RGB(0, 0, 0));
	printf(" If error occurs - run training with flag: -dont_show \n");
	cvNamedWindow("average loss", CV_WINDOW_NORMAL);
	cvMoveWindow("average loss", 0, 0);
	cvResizeWindow("average loss", img_size, img_size);
	cvShowImage("average loss", img);
	cvWaitKey(20);
	return img;
}

void draw_train_loss(IplImage* img, int img_size, float avg_loss, float max_img_loss, int current_batch, int max_batches)
{
	int img_offset = 50;
	int draw_size = img_size - img_offset;
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_COMPLEX_SMALL, 0.7, 0.7, 0, 1, CV_AA);
	char char_buff[100];
	CvPoint pt1, pt2;
	pt1.x = img_offset + draw_size * (float)current_batch / max_batches;
	pt1.y = draw_size * (1 - avg_loss / max_img_loss);
	if (pt1.y < 0) pt1.y = 1;
	cvCircle(img, pt1, 1, CV_RGB(0, 0, 255), CV_FILLED, 8, 0);

	sprintf(char_buff, "current avg loss = %2.4f", avg_loss);
	pt1.x = img_size / 2, pt1.y = 30;
	pt2.x = pt1.x + 250, pt2.y = pt1.y + 20;
	cvRectangle(img, pt1, pt2, CV_RGB(255, 255, 255), CV_FILLED, 8, 0);
	pt1.y += 15;
	cvPutText(img, char_buff, pt1, &font, CV_RGB(0, 0, 0));
	cvShowImage("average loss", img);
	int k = cvWaitKey(20);
	if (k == 's' || current_batch == (max_batches-1)) cvSaveImage("chart.jpg", img, 0);
}
#endif	// OPENCV

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
    int i;
    float min = 9999999;
    float max = -999999;

    for(i = 0; i < p.h*p.w*p.c; ++i){
        float v = p.data[i];
        if(v < min) min = v;
        if(v > max) max = v;
    }
    if(max - min < .000000001){
        min = 0;
        max = 1;
    }
    for(i = 0; i < p.c*p.w*p.h; ++i){
        p.data[i] = (p.data[i] - min)/(max-min);
    }
}

void normalize_image2(image p)
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


void show_image_cv_ipl(IplImage *disp, const char *name)
{
	if (disp == NULL) return;
	char buff[256];
	//sprintf(buff, "%s (%d)", name, windows);
	sprintf(buff, "%s", name);
	cvNamedWindow(buff, CV_WINDOW_NORMAL);
	//cvMoveWindow(buff, 100*(windows%10) + 200*(windows/10), 100*(windows%10));
	++windows;
	cvShowImage(buff, disp);
	//cvReleaseImage(&disp);
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

image get_image_from_stream(CvCapture *cap)
{
    IplImage* src = cvQueryFrame(cap);
    if (!src) return make_empty_image(0,0,0);
    image im = ipl_to_image(src);
    rgbgr_image(im);
    return im;
}

image get_image_from_stream_resize(CvCapture *cap, int w, int h, IplImage** in_img, int use_webcam)
{
	IplImage* src;
	if (use_webcam) src = get_webcam_frame(cap);
	else src = cvQueryFrame(cap);

	if (!src) return make_empty_image(0, 0, 0);
	IplImage* new_img = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 3);
	*in_img = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 3);
	cvResize(src, *in_img, CV_INTER_LINEAR);
	cvResize(src, new_img, CV_INTER_LINEAR);
	image im = ipl_to_image(new_img);
	cvReleaseImage(&new_img);
	rgbgr_image(im);
	return im;
}

void save_image_jpg(image p, const char *name)
{
    image copy = copy_image(p);
    if(p.c == 3) rgbgr_image(copy);
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

void save_image_png(image im, const char *name)
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

void save_image(image im, const char *name)
{
#ifdef OPENCV
    save_image_jpg(im, name);
#else
    save_image_png(im, name);
#endif
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


image rotate_crop_image(image im, float rad, float s, int w, int h, float dx, float dy, float aspect)
{
    int x, y, c;
    float cx = im.w/2.;
    float cy = im.h/2.;
    image rot = make_image(w, h, im.c);
    for(c = 0; c < im.c; ++c){
        for(y = 0; y < h; ++y){
            for(x = 0; x < w; ++x){
                float rx = cos(rad)*((x - w/2.)/s*aspect + dx/s*aspect) - sin(rad)*((y - h/2.)/s + dy/s) + cx;
                float ry = sin(rad)*((x - w/2.)/s*aspect + dx/s*aspect) + cos(rad)*((y - h/2.)/s + dy/s) + cy;
                float val = bilinear_interpolate(im, rx, ry, c);
                set_pixel(rot, x, y, c, val);
            }
        }
    }
    return rot;
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
                r = constrain_int(r, 0, im.h-1);
                c = constrain_int(c, 0, im.w-1);
                if (r >= 0 && r < im.h && c >= 0 && c < im.w) {
                    val = get_pixel(im, c, r, k);
                }
                set_pixel(cropped, i, j, k, val);
            }
        }
    }
    return cropped;
}

int best_3d_shift_r(image a, image b, int min, int max)
{
    if(min == max) return min;
    int mid = floor((min + max) / 2.);
    image c1 = crop_image(b, 0, mid, b.w, b.h);
    image c2 = crop_image(b, 0, mid+1, b.w, b.h);
    float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 10);
    float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 10);
    free_image(c1);
    free_image(c2);
    if(d1 < d2) return best_3d_shift_r(a, b, min, mid);
    else return best_3d_shift_r(a, b, mid+1, max);
}

int best_3d_shift(image a, image b, int min, int max)
{
    int i;
    int best = 0;
    float best_distance = FLT_MAX;
    for(i = min; i <= max; i += 2){
        image c = crop_image(b, 0, i, b.w, b.h);
        float d = dist_array(c.data, a.data, a.w*a.h*a.c, 100);
        if(d < best_distance){
            best_distance = d;
            best = i;
        }
        printf("%d %f\n", i, d);
        free_image(c);
    }
    return best;
}

void composite_3d(char *f1, char *f2, char *out, int delta)
{
    if(!out) out = "out";
    image a = load_image(f1, 0,0,0);
    image b = load_image(f2, 0,0,0);
    int shift = best_3d_shift_r(a, b, -a.h/100, a.h/100);

    image c1 = crop_image(b, 10, shift, b.w, b.h);
    float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 100);
    image c2 = crop_image(b, -10, shift, b.w, b.h);
    float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 100);

    if(d2 < d1 && 0){
        image swap = a;
        a = b;
        b = swap;
        shift = -shift;
        printf("swapped, %d\n", shift);
    }
    else{
        printf("%d\n", shift);
    }

    image c = crop_image(b, delta, shift, a.w, a.h);
    int i;
    for(i = 0; i < c.w*c.h; ++i){
        c.data[i] = a.data[i];
    }
#ifdef OPENCV
    save_image_jpg(c, out);
#else
    save_image(c, out);
#endif
}

void fill_image(image m, float s)
{
	int i;
	for (i = 0; i < m.h*m.w*m.c; ++i) m.data[i] = s;
}

void letterbox_image_into(image im, int w, int h, image boxed)
{
	int new_w = im.w;
	int new_h = im.h;
	if (((float)w / im.w) < ((float)h / im.h)) {
		new_w = w;
		new_h = (im.h * w) / im.w;
	}
	else {
		new_h = h;
		new_w = (im.w * h) / im.h;
	}
	image resized = resize_image(im, new_w, new_h);
	embed_image(resized, boxed, (w - new_w) / 2, (h - new_h) / 2);
	free_image(resized);
}

image letterbox_image(image im, int w, int h)
{
	int new_w = im.w;
	int new_h = im.h;
	if (((float)w / im.w) < ((float)h / im.h)) {
		new_w = w;
		new_h = (im.h * w) / im.w;
	}
	else {
		new_h = h;
		new_w = (im.w * h) / im.h;
	}
	image resized = resize_image(im, new_w, new_h);
	image boxed = make_image(w, h, im.c);
	fill_image(boxed, .5);
	//int i;
	//for(i = 0; i < boxed.w*boxed.h*boxed.c; ++i) boxed.data[i] = 0;
	embed_image(resized, boxed, (w - new_w) / 2, (h - new_h) / 2);
	free_image(resized);
	return boxed;
}

image resize_max(image im, int max)
{
    int w = im.w;
    int h = im.h;
    if(w > h){
        h = (h * max) / w;
        w = max;
    } else {
        w = (w * max) / h;
        h = max;
    }
    if(w == im.w && h == im.h) return im;
    image resized = resize_image(im, w, h);
    return resized;
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
    if(w == im.w && h == im.h) return im;
    image resized = resize_image(im, w, h);
    return resized;
}

image random_crop_image(image im, int w, int h)
{
    int dx = rand_int(0, im.w - w);
    int dy = rand_int(0, im.h - h);
    image crop = crop_image(im, dx, dy, w, h);
    return crop;
}

image random_augment_image(image im, float angle, float aspect, int low, int high, int size)
{
    aspect = rand_scale(aspect);
    int r = rand_int(low, high);
    int min = (im.h < im.w*aspect) ? im.h : im.w*aspect;
    float scale = (float)r / min;

    float rad = rand_uniform(-angle, angle) * TWO_PI / 360.;

    float dx = (im.w*scale/aspect - size) / 2.;
    float dy = (im.h*scale - size) / 2.;
    if(dx < 0) dx = 0;
    if(dy < 0) dy = 0;
    dx = rand_uniform(-dx, dx);
    dy = rand_uniform(-dy, dy);

    image crop = rotate_crop_image(im, rad, scale, size, size, dx, dy, aspect);

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
                h = 0;
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
                h = h/6.;
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
            h = 6 * get_pixel(im, i , j, 0);
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

void translate_image_channel(image im, int c, float v)
{
    int i, j;
    for(j = 0; j < im.h; ++j){
        for(i = 0; i < im.w; ++i){
            float pix = get_pixel(im, i, j, c);
            pix = pix+v;
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

void hue_image(image im, float hue)
{
    rgb_to_hsv(im);
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        im.data[i] = im.data[i] + hue;
        if (im.data[i] > 1) im.data[i] -= 1;
        if (im.data[i] < 0) im.data[i] += 1;
    }
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

void distort_image(image im, float hue, float sat, float val)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    scale_image_channel(im, 2, val);
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        im.data[i] = im.data[i] + hue;
        if (im.data[i] > 1) im.data[i] -= 1;
        if (im.data[i] < 0) im.data[i] += 1;
    }
    hsv_to_rgb(im);
    constrain_image(im);
}

void random_distort_image(image im, float hue, float saturation, float exposure)
{
    float dhue = rand_uniform_strong(-hue, hue);
    float dsat = rand_scale(saturation);
    float dexp = rand_scale(exposure);
    distort_image(im, dhue, dsat, dexp);
}

void saturate_exposure_image(image im, float sat, float exposure)
{
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    scale_image_channel(im, 2, exposure);
    hsv_to_rgb(im);
    constrain_image(im);
}

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


void test_resize(char *filename)
{
    image im = load_image(filename, 0,0, 3);
    float mag = mag_array(im.data, im.w*im.h*im.c);
    printf("L2 Norm: %f\n", mag);
    image gray = grayscale_image(im);

    image c1 = copy_image(im);
    image c2 = copy_image(im);
    image c3 = copy_image(im);
    image c4 = copy_image(im);
    distort_image(c1, .1, 1.5, 1.5);
    distort_image(c2, -.1, .66666, .66666);
    distort_image(c3, .1, 1.5, .66666);
    distort_image(c4, .1, .66666, 1.5);


    show_image(im,   "Original");
    show_image(gray, "Gray");
    show_image(c1, "C1");
    show_image(c2, "C2");
    show_image(c3, "C3");
    show_image(c4, "C4");
#ifdef OPENCV
    while(1){
        image aug = random_augment_image(im, 0, .75, 320, 448, 320);
        show_image(aug, "aug");
        free_image(aug);


        float exposure = 1.15;
        float saturation = 1.15;
        float hue = .05;

        image c = copy_image(im);

        float dexp = rand_scale(exposure);
        float dsat = rand_scale(saturation);
        float dhue = rand_uniform(-hue, hue);

        distort_image(c, dhue, dsat, dexp);
        show_image(c, "rand");
        printf("%f %f %f\n", dhue, dsat, dexp);
        free_image(c);
        cvWaitKey(0);
    }
#endif
}


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

#ifndef CV_VERSION_EPOCH
	//image out = load_image_stb(filename, c);	// OpenCV 3.x
	image out = load_image_cv(filename, c);
#else
	image out = load_image_cv(filename, c);		// OpenCV 2.4.x
#endif

#else
    image out = load_image_stb(filename, c);	// without OpenCV
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
    if(x < 0) x = 0;
    if(x >= m.w) x = m.w-1;
    if(y < 0) y = 0;
    if(y >= m.h) y = m.h-1;
    if(c < 0 || c >= m.c) return 0;
    return get_pixel(m, x, y, c);
}
void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
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
    save_image(m, window);
    show_image(m, window);
    free_image(m);
}

void free_image(image m)
{
    if(m.data){
        free(m.data);
    }
}
