"""Detector functions with different imread methods"""

import ctypes
from darknet_libwrapper import *
from scipy.misc import imread
import cv2

def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = c_array(ctypes.c_float, arr)
    im = IMAGE(w,h,c,data)
    return im

def _detector(net, meta, image, thresh=.5, hier=.5, nms=.45):
    cuda_set_device(0)
    num = ctypes.c_int(0)
    num_ptr = ctypes.pointer(num)
    network_predict_image(net, image)
    dets = get_network_boxes(net, image.w, image.h, thresh, hier, None, 0, num_ptr)
    num = num_ptr[0]
    if (nms):
         do_nms_sort(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

# Darknet
net = load_network("cfg/yolov2-tiny.cfg", "yolov2-tiny.weights", 0)
meta = get_metadata("cfg/coco.data")
im = load_image_color('data/dog.jpg', 0, 0)
result = _detector(net, meta, im)
print 'Darknet:\n', result

# scipy
arr= imread('data/dog.jpg')
im = array_to_image(arr)
result = _detector(net, meta, im)
print 'Scipy:\n', result

# OpenCV
arr = cv2.imread('data/dog.jpg')
im = array_to_image(arr)
rgbgr_image(im)
result = _detector(net, meta, im)
print 'OpenCV:\n', result
