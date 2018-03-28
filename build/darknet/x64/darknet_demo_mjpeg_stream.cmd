rem Run this file and then open URL in Chrome/Firefox: rem http://localhost:8090
rem Or open: http://ip-address:8090

darknet.exe detector demo data/voc.data cfg/yolov2-voc.cfg yolo-voc.weights test.mp4 -i 0 -http_port 8090 -dont_show


pause