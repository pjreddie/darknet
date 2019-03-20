# Run this file and then open URL in Chrome/Firefox in 2 tabs: http://localhost:8070 and http://localhost:8090
# Or open: http://ip-address:8070 and http://ip-address:8090
# to get <ip-address> run: sudo ifconfig

./darknet detector demo ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights test50.mp4 -json_port 8070 -mjpeg_port 8090 -ext_output

