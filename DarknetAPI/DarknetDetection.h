//
// Created by frivas on 26/07/16.
//

#ifndef DARKNET_DARKNETDETECTION_H
#define DARKNET_DARKNETDETECTION_H

#include <box.h>
#include <vector>
#include <string>
#include <rapidjson/document.h>
#include <opencv2/core/core.hpp>


struct DarknetDetection {
    int classId;
    cv::Rect detectionBox;
    std::string serialize();
    double probability;
    rapidjson::Value getJsonValue(rapidjson::Document& document) const;


    DarknetDetection(const box& detectionBox, const int  classId, double probability);
    DarknetDetection(const float x, const float y, const float w, const float h, const int classId,double probability);
    DarknetDetection(const rapidjson::Value& jsonValue);
};

struct DarknetDetections {
    std::vector<DarknetDetection> data;
    std::vector<DarknetDetection> get();

    DarknetDetections();
    DarknetDetections(const rapidjson::Value& jsonValue);
    std::string serialize();
    rapidjson::Value getJsonValue(rapidjson::Document& document) const;
    void push_back(DarknetDetection& d);


};


#endif //DARKNET_DARKNETDETECTION_H
