//
// Created by frivas on 26/07/16.
//

#include "DarknetDetection.h"
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <iostream>
#include <fstream>
#include <sstream>

std::string DarknetDetection::serialize() {
    rapidjson::Document document;
    document.SetObject();
    rapidjson::Value value;//= getJsonValue(document);
    rapidjson::StringBuffer buffer;
    //rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    value.Accept(writer);

    const char *jsonString = buffer.GetString();
    return std::string(jsonString);
}

DarknetDetection::DarknetDetection(const box &detectionBox, const int classId, double probability) {
    this->detectionBox.x=detectionBox.x;
    this->detectionBox.y=detectionBox.y;
    this->detectionBox.width=detectionBox.w;
    this->detectionBox.height=detectionBox.h;
    this->classId=classId;
    this->probability=probability;
}

DarknetDetection::DarknetDetection(const float x, const float y, const float w, const float h, const int classId, double probability) {
    detectionBox.x=x;
    detectionBox.y=y;
    detectionBox.width=w;
    detectionBox.height=h;
    this->classId=classId;
    this->probability=probability;
}

DarknetDetection::DarknetDetection(const rapidjson::Value &jsonValue) {
    auto end = jsonValue.MemberEnd();

    auto xIt = jsonValue.FindMember("x");
    if(xIt == end)
    {
        std::cerr << "x node not found!" << std::endl;
        return;
    }
    this->detectionBox.x = xIt->value.GetDouble();

    auto yIt = jsonValue.FindMember("y");
    if(yIt == end)
    {
        std::cerr << "y node not found!" << std::endl;
        return;
    }
    this->detectionBox.y = yIt->value.GetDouble();


    auto wIt = jsonValue.FindMember("width");
    if(wIt == end)
    {
        std::cerr << "w node not found!" << std::endl;
        return;
    }
    this->detectionBox.width = wIt->value.GetDouble();

    auto hIt = jsonValue.FindMember("height");
    if(hIt == end)
    {
        std::cerr << "w node not found!" << std::endl;
        return;
    }
    this->detectionBox.height = hIt->value.GetDouble();

    auto idIt = jsonValue.FindMember("id");
    if(idIt == end)
    {
        std::cerr << "id node not found!" << std::endl;
        return;
    }
    this->classId = idIt->value.GetDouble();

}


rapidjson::Value DarknetDetection::getJsonValue(rapidjson::Document &document) const {
    document.SetObject();

    auto &allocator = document.GetAllocator();
    rapidjson::Value value;

    value.SetObject();


    value.AddMember("x", detectionBox.x,allocator);
    value.AddMember("y", detectionBox.y, allocator);
    value.AddMember("width", detectionBox.width, allocator);
    value.AddMember("height", detectionBox.height, allocator);
    value.AddMember("id", classId, allocator);

    return value;
}


std::vector<DarknetDetection> DarknetDetections::get() {
    return data;
}

DarknetDetections::DarknetDetections(const rapidjson::Value &jsonValue) {
    data.clear();
    if (!jsonValue.IsArray()){
        std::cerr << "node is not an array!" << std::endl;
        return;
    }

    for (rapidjson::SizeType nodes = 0; nodes < jsonValue.Size(); nodes++) {
        DarknetDetection d(jsonValue[nodes]);
        data.push_back(d);
    }
}

rapidjson::Value DarknetDetections::getJsonValue(rapidjson::Document &document) const {


    auto &allocator = document.GetAllocator();
    rapidjson::Value arrayDetections;

    arrayDetections.SetArray();
    for (auto it = data.begin(), end = data.end(); it != end; ++it){
        rapidjson::Value detectionValue = it->getJsonValue(document);
        arrayDetections.PushBack(detectionValue,allocator);
    }
    return arrayDetections;
}

std::string DarknetDetections::serialize() {

    if (data.size()) {
        rapidjson::Document document;
        document.SetObject();
        rapidjson::Value value = getJsonValue(document);
        rapidjson::StringBuffer buffer;
        //rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        value.Accept(writer);

        const char *jsonString = buffer.GetString();
        return std::string(jsonString);
    }
    else
        return std::string("[]");
}


DarknetDetections::DarknetDetections() {

}

void DarknetDetections::push_back(DarknetDetection &d) {
    data.push_back(d);
}











