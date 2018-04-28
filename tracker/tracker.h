#pragma once

#include <opencv2/opencv.hpp>
#include <string>

class Tracker
{
public:
    Tracker()  {}
    virtual  ~Tracker() { }

    virtual void init(const cv::Rect &roi, cv::Mat image) = 0;
    virtual cv::Rect  update( cv::Mat image)=0;


protected:
    cv::Rect_<float> _roi;
};



