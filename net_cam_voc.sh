#rm test_dnn_out.avi

./darknet detector demo ./cfg/voc.data ./cfg/yolo-voc.cfg ./yolo-voc.weights rtsp://admin:admin12345@192.168.0.228:554 -i 0 -thresh 0.24



