#pragma once
#include <memory>
#include <vector>
#include <deque>
#include <algorithm>

#ifdef OPENCV
#include <opencv2/opencv.hpp>			// C++
#include "opencv2/highgui/highgui_c.h"	// C
#include "opencv2/imgproc/imgproc_c.h"	// C
#endif	// OPENCV

#ifdef YOLODLL_EXPORTS
#if defined(_MSC_VER)
#define YOLODLL_API __declspec(dllexport) 
#else
#define YOLODLL_API __attribute__((visibility("default")))
#endif
#else
#if defined(_MSC_VER)
#define YOLODLL_API __declspec(dllimport) 
#else
#define YOLODLL_API
#endif
#endif

struct bbox_t {
	unsigned int x, y, w, h;	// (x,y) - top-left corner, (w, h) - width & height of bounded box
	float prob;					// confidence - probability that the object was found correctly
	unsigned int obj_id;		// class of object - from range [0, classes-1]
	unsigned int track_id;		// tracking id for video (0 - untracked, 1 - inf - tracked object)
};

struct image_t {
	int h;						// height
	int w;						// width
	int c;						// number of chanels (3 - for RGB)
	float *data;				// pointer to the image data
};


class Detector {
	std::shared_ptr<void> detector_gpu_ptr;
public:
	float nms = .4;

	YOLODLL_API Detector(std::string cfg_filename, std::string weight_filename, int gpu_id = 0);
	YOLODLL_API ~Detector();

	YOLODLL_API std::vector<bbox_t> detect(std::string image_filename, float thresh = 0.2, bool use_mean = false);
	YOLODLL_API std::vector<bbox_t> detect(image_t img, float thresh = 0.2, bool use_mean = false);
	static YOLODLL_API image_t load_image(std::string image_filename);
	static YOLODLL_API void free_image(image_t m);
	YOLODLL_API int get_net_width() const;
	YOLODLL_API int get_net_height() const;

	YOLODLL_API std::vector<bbox_t> tracking(std::vector<bbox_t> cur_bbox_vec, int const frames_story = 6);

#ifdef OPENCV
	std::vector<bbox_t> detect(cv::Mat mat, float thresh = 0.2, bool use_mean = false)
	{
		if(mat.data == NULL)
			throw std::runtime_error("Image is empty");
		auto image_ptr = mat_to_image_resize(mat);
		return detect_resized(*image_ptr, mat.size(), thresh, use_mean);
	}

	std::vector<bbox_t> detect_resized(image_t img, cv::Size init_size, float thresh = 0.2, bool use_mean = false)
	{
		if (img.data == NULL)
			throw std::runtime_error("Image is empty");
		auto detection_boxes = detect(img, thresh, use_mean);
		float wk = (float)init_size.width / img.w, hk = (float)init_size.height / img.h;
		for (auto &i : detection_boxes) i.x *= wk, i.w *= wk, i.y *= hk, i.h *= hk;
		return detection_boxes;
	}

	std::shared_ptr<image_t> mat_to_image_resize(cv::Mat mat) const
	{
		if (mat.data == NULL) return std::shared_ptr<image_t>(NULL);
		cv::Mat det_mat;
		cv::resize(mat, det_mat, cv::Size(get_net_width(), get_net_height()));
		return mat_to_image(det_mat);
	}

	static std::shared_ptr<image_t> mat_to_image(cv::Mat img)
	{
		std::shared_ptr<image_t> image_ptr(new image_t, [](image_t *img) { free_image(*img); delete img; });
		std::shared_ptr<IplImage> ipl_small = std::make_shared<IplImage>(img);
		*image_ptr = ipl_to_image(ipl_small.get());
		rgbgr_image(*image_ptr);
		return image_ptr;
	}

private:
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

#endif	// OPENCV

	std::deque<std::vector<bbox_t>> prev_bbox_vec_deque;
};



