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
#define YOLODLL_API __declspec(dllexport) 
#else
#define YOLODLL_API __declspec(dllimport) 
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

	YOLODLL_API std::vector<bbox_t> detect(std::string image_filename, float thresh = 0.2);
	YOLODLL_API std::vector<bbox_t> detect(image_t img, float thresh = 0.2);
	static YOLODLL_API image_t load_image(std::string image_filename);
	static YOLODLL_API void free_image(image_t m);

#ifdef OPENCV
	std::vector<bbox_t> detect(cv::Mat mat, float thresh = 0.2) 
	{
		if(mat.data == NULL)
			throw std::runtime_error("file not found");
		std::shared_ptr<image_t> image_ptr(new image_t, [](image_t *img) { free_image(*img); delete img; });
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

#endif	// OPENCV

	std::deque<std::vector<bbox_t>> prev_bbox_vec_deque;

public:
	std::vector<bbox_t> tracking(std::vector<bbox_t> cur_bbox_vec, int const frames_story = 4)
	{
		bool prev_track_id_present = false;
		for (auto &i : prev_bbox_vec_deque)
			if (i.size() > 0) prev_track_id_present = true;

		static unsigned int track_id = 1;

		if(!prev_track_id_present) {
			//track_id = 1;
			for (size_t i = 0; i < cur_bbox_vec.size(); ++i)
				cur_bbox_vec[i].track_id = track_id++;
			prev_bbox_vec_deque.push_front(cur_bbox_vec);
			if (prev_bbox_vec_deque.size() > frames_story) prev_bbox_vec_deque.pop_back();
			return cur_bbox_vec;
		}

		std::vector<unsigned int> dist_vec(cur_bbox_vec.size(), std::numeric_limits<unsigned int>::max());

		for (auto &prev_bbox_vec : prev_bbox_vec_deque) {
			for (auto &i : prev_bbox_vec) {
				int cur_index = -1;
				for (size_t m = 0; m < cur_bbox_vec.size(); ++m) {
					bbox_t const& k = cur_bbox_vec[m];
					if (i.obj_id == k.obj_id) {
						unsigned int cur_dist = sqrt(((float)i.x - k.x)*((float)i.x - k.x) + ((float)i.y - k.y)*((float)i.y - k.y));
						if (cur_dist < 100 && (k.track_id == 0 || dist_vec[m] > cur_dist)) {
							dist_vec[m] = cur_dist;
							cur_index = m;
						}
					}
				}

				bool track_id_absent = !std::any_of(cur_bbox_vec.begin(), cur_bbox_vec.end(), [&](bbox_t const& b) { return b.track_id == i.track_id; });

				if (cur_index >= 0 && track_id_absent)
					cur_bbox_vec[cur_index].track_id = i.track_id;
			}
		}

		for (size_t i = 0; i < cur_bbox_vec.size(); ++i)
			if (cur_bbox_vec[i].track_id == 0)
				cur_bbox_vec[i].track_id = track_id++;

		prev_bbox_vec_deque.push_front(cur_bbox_vec);
		if (prev_bbox_vec_deque.size() > frames_story) prev_bbox_vec_deque.pop_back();

		return cur_bbox_vec;
	}
};



