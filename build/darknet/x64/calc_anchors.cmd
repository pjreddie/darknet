rem # How to calculate Yolo v2 anchors using K-means++


darknet.exe detector calc_anchors data/voc.data -num_of_clusters 9 -width 416 -height 416


rem darknet.exe detector calc_anchors data/voc.data -num_of_clusters 9 -width 416 -height 416 -show




pause
