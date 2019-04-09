
rem darknet.exe detector test cfg/voc.data cfg/yolov2-voc.cfg yolo-voc.weights 009460.jpg


darknet.exe detector test cfg/voc.data cfg/yolov2-voc.cfg yolo-voc.weights -i 0 -thresh 0.24 dog.jpg


pause