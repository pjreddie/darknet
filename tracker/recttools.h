
#pragma once

//#include <cv.h>
#include <math.h>

#ifndef _OPENCV_RECTTOOLS_HPP_
#define _OPENCV_RECTTOOLS_HPP_
#endif

namespace RectTools
{

template <typename t>
inline cv::Vec<t, 2 > center(const cv::Rect_<t> &rect)
{
    return cv::Vec<t, 2 > (rect.x + rect.width / (t) 2, rect.y + rect.height / (t) 2);
}

template <typename t>
inline t x2(const cv::Rect_<t> &rect)
{
    return rect.x + rect.width;
}

template <typename t>
inline t y2(const cv::Rect_<t> &rect)
{
    return rect.y + rect.height;
}

template <typename t>
inline void resize(cv::Rect_<t> &rect, float scalex, float scaley = 0)
{
    if (!scaley)scaley = scalex;
    rect.x -= rect.width * (scalex - 1.f) / 2.f;
    rect.width *= scalex;

    rect.y -= rect.height * (scaley - 1.f) / 2.f;
    rect.height *= scaley;

}

template <typename t>
inline void limit(cv::Rect_<t> &rect, cv::Rect_<t> limit)
{
    if (rect.x + rect.width > limit.x + limit.width)rect.width = (limit.x + limit.width - rect.x);
    if (rect.y + rect.height > limit.y + limit.height)rect.height = (limit.y + limit.height - rect.y);
    if (rect.x < limit.x)
    {
        rect.width -= (limit.x - rect.x);
        rect.x = limit.x;
    }
    if (rect.y < limit.y)
    {
        rect.height -= (limit.y - rect.y);
        rect.y = limit.y;
    }
    if(rect.width<0)rect.width=0;
    if(rect.height<0)rect.height=0;
}

template <typename t>
inline void limit(cv::Rect_<t> &rect, t width, t height, t x = 0, t y = 0)
{
    limit(rect, cv::Rect_<t > (x, y, width, height));
}

template <typename t>
inline cv::Rect getBorder(const cv::Rect_<t > &original, cv::Rect_<t > & limited)
{
    cv::Rect_<t > res;
    res.x = limited.x - original.x;
    res.y = limited.y - original.y;
    res.width = x2(original) - x2(limited);
    res.height = y2(original) - y2(limited);
    assert(res.x >= 0 && res.y >= 0 && res.width >= 0 && res.height >= 0);
    return res;
}

inline cv::Mat subwindow(const cv::Mat &in, const cv::Rect & window, int borderType = cv::BORDER_CONSTANT)
{
    cv::Rect cutWindow = window;
    RectTools::limit(cutWindow, in.cols, in.rows);
    if (cutWindow.height <= 0 || cutWindow.width <= 0)assert(0); //return cv::Mat(window.height,window.width,in.type(),0) ;
    cv::Rect border = RectTools::getBorder(window, cutWindow);
    cv::Mat res = in(cutWindow);

    if (border != cv::Rect(0, 0, 0, 0))
    {
        cv::copyMakeBorder(res, res, border.y, border.height, border.x, border.width, borderType);
    }
    return res;
}

inline cv::Mat getGrayImage(cv::Mat img)
{
    cv::cvtColor(img, img, CV_BGR2GRAY);
    img.convertTo(img, CV_32F, 1 / 255.f);
    return img;
}

}



