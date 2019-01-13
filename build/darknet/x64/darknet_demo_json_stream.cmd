rem Run this file and then open URL in Chrome/Firefox: rem http://localhost:8070
rem Or open: http://ip-address:8070

darknet.exe detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights test.mp4 -i 0 -json_port 8070 -ext_output


pause