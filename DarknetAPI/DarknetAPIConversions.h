//
// Created by frivas on 26/07/16.
//

#ifndef DARKNET_DARKNETAPICONVERSIONS_H
#define DARKNET_DARKNETAPICONVERSIONS_H

#include <image.h>
#include <opencv2/core/core.hpp>
#include "DarknetAPI.h"

inline image cv_to_image(const cv::Mat_<cv::Vec3b>& src)
{
    unsigned char *data = (unsigned char *)src.data;
    int h = src.rows;
    int w = src.cols;
    int c = src.channels();
    int step = src.step;
    image out = c_make_image(w, h, c);
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

inline cv::Mat_<cv::Vec3b> image_to_cv(image p)
{
    image copy = c_copy_image(p);
    c_rgbgr_image(copy);
    int x,y,k;

    cv::Mat_<cv::Vec3b> disp(p.h, p.w);
    //IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
    int step = disp.step;
    for(y = 0; y < p.h; ++y){
        for(x = 0; x < p.w; ++x){
            for(k= 0; k < p.c; ++k){
                disp.data[y*step + x*p.c + k] = (unsigned char)(c_get_pixel(copy,x,y,k)*255);
            }
        }
    }

    c_free_image(copy);

    return disp;
}

inline std::vector< std::pair<int, cv::Rect> > box_to_cv(const std::vector< std::pair<int, box> >& boxes){

    std::vector< std::pair<int, cv::Rect> > cvboxes;

    for(size_t i = 0; i < boxes.size(); i++){

        box b = boxes[i].second;
        //std::cout<<b.x<<" "<<b.y<<" "<<b.w<<" "<<b.h<<std::endl;
        cv::Rect cvbox(b.x, b.y, b.w, b.h);
        cvboxes.push_back(std::make_pair(boxes[i].first, cvbox));

    }

    return cvboxes;

}

#endif //DARKNET_DARKNETAPICONVERSIONS_H
