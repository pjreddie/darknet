#pragma once
#include <memory>
#include <vector>

#ifdef OPENCV
#include <opencv2/opencv.hpp>			// C++
#include "opencv2/highgui/highgui_c.h"	// C
#include "opencv2/imgproc/imgproc_c.h"	// C
#endif	// OPENCV

#ifdef YOLODLL_EXPORTS
#define YOLODLL_API __declspec(dllexport) 
#else
#define YOLODLL_API __declspec(dllimport) 
#endif

struct bbox_t {
	float x, y, w, h;
	float prob;
	unsigned int obj_id;
};

typedef struct {
	int h;
	int w;
	int c;
	float *data;
} image_t;


class Detector {
	std::shared_ptr<void> detector_gpu_ptr;
public:

	YOLODLL_API Detector(std::string cfg_filename, std::string weight_filename, int gpu_id = 0);
	YOLODLL_API ~Detector();

	YOLODLL_API std::vector<bbox_t> detect(std::string image_filename, float thresh = 0.2);

	YOLODLL_API std::vector<bbox_t> detect(image_t img, float thresh = 0.2);


#ifdef OPENCV
	std::vector<bbox_t> detect(cv::Mat mat, float thresh = 0.2) {
		std::shared_ptr<image_t> image_ptr(new image_t, [](image_t *img) { free_image(*img); } );
		*image_ptr = mat_to_image(mat);
		return detect(*image_ptr, thresh);
	}

private:
	static image_t mat_to_image(cv::Mat img)
	{
		std::shared_ptr<IplImage> ipl_small = std::make_shared<IplImage>(img);
		image_t im_small = ipl_to_image(ipl_small.get());
		rgbgr_image(im_small);
		return im_small;
	}

	static image_t ipl_to_image(IplImage* src)
	{
		unsigned char *data = (unsigned char *)src->imageData;
		int h = src->height;
		int w = src->width;
		int c = src->nChannels;
		int step = src->widthStep;
		image_t out = make_image_custom(w, h, c);
		int i, j, k, count = 0;;

		for (k = 0; k < c; ++k) {
			for (i = 0; i < h; ++i) {
				for (j = 0; j < w; ++j) {
					out.data[count++] = data[i*step + j*c + k] / 255.;
				}
			}
		}
		return out;
	}

	static image_t make_empty_image(int w, int h, int c)
	{
		image_t out;
		out.data = 0;
		out.h = h;
		out.w = w;
		out.c = c;
		return out;
	}

	static image_t make_image_custom(int w, int h, int c)
	{
		image_t out = make_empty_image(w, h, c);
		out.data = (float *)calloc(h*w*c, sizeof(float));
		return out;
	}

	static void rgbgr_image(image_t im)
	{
		int i;
		for (i = 0; i < im.w*im.h; ++i) {
			float swap = im.data[i];
			im.data[i] = im.data[i + im.w*im.h * 2];
			im.data[i + im.w*im.h * 2] = swap;
		}
	}

	static void free_image(image_t m)
	{
		if (m.data) {
			free(m.data);
		}
	}
#endif	// OPENCV
};



