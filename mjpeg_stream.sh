rem Run this file and then open URL in Chrome/Firefox: rem http://localhost:8090
rem Or open: http://ip-address:8090

./darknet detector demo ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights test50.mp4 -i 0 -thresh 0.25 -http_port 8090



pause