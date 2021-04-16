#!/usr/bin/env pwsh

$url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
Invoke-WebRequest -Uri $url -OutFile "yolov4-tiny.weights"

$url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
Invoke-WebRequest -Uri $url -OutFile "yolov4.weights"

$url = "https://drive.google.com/u/0/uc?id=18yYZWyKbo4XSDVyztmsEcF9B_6bxrhUY&export=download"
Invoke-WebRequest -Uri $url -OutFile "yolov3-tiny-prn.weights"

$url = "https://pjreddie.com/media/files/yolov3.weights"
Invoke-WebRequest -Uri $url -OutFile "yolov3.weights"

$url = "https://pjreddie.com/media/files/yolov3-openimages.weights"
Invoke-WebRequest -Uri $url -OutFile "yolov3-openimages.weights"

$url = "https://pjreddie.com/media/files/yolov2.weights"
Invoke-WebRequest -Uri $url -OutFile "yolov2.weights"

$url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
Invoke-WebRequest -Uri $url -OutFile "yolov3-tiny.weights"

$url = "https://pjreddie.com/media/files/yolov2-tiny.weights"
Invoke-WebRequest -Uri $url -OutFile "yolov2-tiny.weights"

$url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29"
Invoke-WebRequest -Uri $url -OutFile "yolov4-tiny.conv.29"

$url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137"
Invoke-WebRequest -Uri $url -OutFile "yolov4.conv.137"

$url = "https://pjreddie.com/media/files/darknet53.conv.74"
Invoke-WebRequest -Uri $url -OutFile "darknet53.conv.74"

$url = "https://pjreddie.com/media/files/darknet19_448.conv.23"
Invoke-WebRequest -Uri $url -OutFile "darknet19_448.conv.23"
