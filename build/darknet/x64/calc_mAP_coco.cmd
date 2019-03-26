rem # How to calculate Yolo v3 mAP on MS COCO

rem darknet.exe detector map cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights -points 101


darknet.exe detector map cfg/coco.data cfg/yolov3-spp.cfg yolov3-spp.weights -points 101


rem darknet.exe detector map cfg/coco.data cfg/yolov3.cfg yolov3.weights -points 101


rem darknet.exe detector map cfg/coco.data cfg/yolov3.cfg yolov3.weights -iou_thresh 0.75 -points 101



pause
