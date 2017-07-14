//
// Created by frivas on 26/07/16.
//



#include <DarknetAPI/DarknetAPI.h>
#include <iostream>
#include <highgui.h>
#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>


int main(int argc, char * argv[]){



//    c_test_detector("/home/frivas/devel/WS_git/darknet/code/build/test/voc.names",
//                    "/home/frivas/devel/WS_git/darknet/code/build/test/yolo.net","/home/frivas/devel/WS_git/darknet/code/build/test/yolo.weights", "/home/frivas/devel/WS_git/darknet/code/build/test/health_check.png", .24, .5);

    boost::shared_ptr<DarknetAPI > api(new DarknetAPI("/home/frivas/devel/external/darknet/cfg/yolo.cfg", "/home/frivas/devel/external/darknet/yolo.weights"));


    cv::Mat imageCV = cv::imread("/mnt/large/pentalo/deep/datasets/voc/original_format/VOCdevkit/VOC2007/JPEGImages/000004.jpg");
    DarknetDetections d= api->process(imageCV);


    for (auto it = d.data.begin(), end=d.data.end(); it !=end; ++it){
        std::cout << it->classId << " " << it->probability << std::endl;
//        std::cout<< ClassType(it->classId).getName() << ": " << it->probability << std::endl;
    }

    std::cout << "detections: " << d.data.size() << std::endl;
    std::cout << d.data[1].detectionBox.x << std::endl;
    std::cout << d.serialize() << std::endl;
    std::cout << "Done" << std::endl;
    return 0;
}