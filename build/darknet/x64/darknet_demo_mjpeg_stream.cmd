rem Run this file and then open URL in Chrome/Firefox: rem http://localhost:8090
rem Or open: http://ip-address:8090

darknet.exe detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights test.mp4 -i 0 -mjpeg_port 8090 -dont_show -ext_output


pause