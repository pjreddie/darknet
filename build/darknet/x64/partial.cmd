rem Download weights for - DenseNet201, ResNet50 and ResNet152 by this link: https://pjreddie.com/darknet/imagenet/
rem Download Yolo/Tiny-yolo: https://pjreddie.com/darknet/yolo/
rem Download Yolo9000: http://pjreddie.com/media/files/yolo9000.weights


darknet.exe partial cfg/yolov4-csp-swish.cfg yolov4-csp-swish.weights yolov4-csp-swish.conv.164 164


darknet.exe partial cfg/yolov4-csp-x-swish.cfg yolov4-csp-x-swish.weights yolov4-csp-x-swish.conv.192 192


pause

darknet.exe partial cfg/yolov4-csp.cfg yolov4-csp.weights yolov4-csp.conv.142 142

darknet.exe partial cfg/yolov4x-mish.cfg yolov4x-mish.weights yolov4x-mish.conv.166 166




rem darknet.exe partial cfg/yolov4-p5.cfg yolov4-p5.weights yolov4-p5.conv.232 232

rem darknet.exe partial cfg/yolov4-p6.cfg yolov4-p6.weights yolov4-p6.conv.289 289



rem darknet.exe partial cfg/tiny-yolo-voc.cfg tiny-yolo-voc.weights tiny-yolo-voc.conv.13 13

rem darknet.exe partial cfg/yolov4-tiny.cfg yolov4-tiny.weights yolov4-tiny.conv.29 29


rem darknet.exe partial cfg/yolov4-sam-mish.cfg cfg/yolov4-sam-mish.weights cfg/yolov4-sam-mish.conv.137 137

rem darknet.exe partial cfg/yolov4-sam-mish.cfg cfg/yolov4-sam-mish.weights cfg/yolov4-sam-mish.conv.105 105


pause

darknet.exe partial cfg/csdarknet53-omega.cfg csdarknet53-omega_final.weights csdarknet53-omega.conv.105 105


darknet.exe partial cfg/cd53paspp-omega.cfg cd53paspp-omega_final.weights cd53paspp-omega.conv.137 137


darknet.exe partial cfg/yolov4.cfg backup/yolov4_final.weights yolov4.conv.137 137


darknet.exe partial cfg/csresnext50.cfg csresnext50.weights csresnext50.conv.75 75


darknet.exe partial cfg/darknet53_448.cfg darknet53_448.weights darknet53.conv.74 74


darknet.exe partial cfg/darknet53_448_xnor.cfg darknet53_448_xnor.weights darknet53_448_xnor.conv.74 74


darknet.exe partial cfg/yolov2-tiny-voc.cfg yolov2-tiny-voc.weights yolov2-tiny-voc.conv.13 13


darknet.exe partial cfg/yolov2-tiny.cfg yolov2-tiny.weights yolov2-tiny.conv.13 13


darknet.exe partial cfg/yolo-voc.cfg yolo-voc.weights yolo-voc.conv.23 23


darknet.exe partial cfg/yolov2.cfg yolov2.weights yolov2.conv.23 23


darknet.exe partial cfg/yolov3.cfg yolov3.weights yolov3.conv.81 81


darknet.exe partial cfg/yolov3-spp.cfg yolov3-spp.weights yolov3-spp.conv.85 85


darknet.exe partial cfg/yolov3-tiny.cfg yolov3-tiny.weights yolov3-tiny.conv.15 15


darknet.exe partial cfg/yolov3-tiny.cfg yolov3-tiny.weights yolov3-tiny.conv.14 14

darknet.exe partial cfg/yolov3-tiny.cfg yolov3-tiny.weights yolov3-tiny.conv.13 13


darknet.exe partial cfg/yolo9000.cfg yolo9000.weights yolo9000.conv.22 22


darknet.exe partial cfg/densenet201.cfg densenet201.weights densenet201.57 57


darknet.exe partial cfg/densenet201.cfg densenet201.weights densenet201.300 300


darknet.exe partial cfg/resnet50.cfg resnet50.weights resnet50.65 65


darknet.exe partial cfg/resnet152.cfg resnet152.weights resnet152.201 201


pause