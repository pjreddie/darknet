rem Download weights for - DenseNet201, ResNet50 and ResNet152 by this link: https://pjreddie.com/darknet/imagenet/
rem Download Yolo/Tiny-yolo: https://pjreddie.com/darknet/yolo/
rem Download Yolo9000: http://pjreddie.com/media/files/yolo9000.weights


darknet.exe partial cfg/tiny-yolo-voc.cfg tiny-yolo-voc.weights tiny-yolo-voc.conv.13 13


darknet.exe partial cfg/yolo-voc.cfg yolo-voc.weights yolo-voc.conv.23 23


darknet.exe partial cfg/yolo9000.cfg yolo9000.weights yolo9000.conv.22 22


darknet.exe partial cfg/densenet201.cfg densenet201.weights densenet201.57 57


darknet.exe partial cfg/densenet201.cfg densenet201.weights densenet201.300 300


darknet.exe partial cfg/resnet50.cfg resnet50.weights resnet50.65 65


darknet.exe partial cfg/resnet152.cfg resnet152.weights resnet152.200 200


pause