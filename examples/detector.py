"""
Represent python module from examples/detector.c
NOTICE: before loading network, verified runtime file paths in cfg or data file,
        because libdarknet may uses file relative instead of calller working directory
"""

import ctypes
from darknet_libwrapper import *

def run_detector(*argv):
    """argv: test data cfg weight jpg"""
    if argv[2] == 'test':
        argv = [x for x in argv if x != 'test']
        argv.append('.5') #thresh
        argv.append('.5') #hier_thresh
        argv.append('.45') #nms
        test_detector(*argv)
    else:
        print('Not Implementation')

def test_detector(*argv):
    """argv: data cfg weight jpg thresh hier nms"""
    print('test data:{2} cfg:{3} weight:{4} img:{5}'.format(*argv))
    cuda_set_device(0)
    thresh = float(argv[6])
    hier_thresh = float(argv[7])
    nms = float(argv[8])
    net = load_network(argv[3], argv[4], 0)
    meta = get_metadata(argv[2])
    im = load_image_color(argv[5], 0, 0)
    num = ctypes.c_int(0)
    num_ptr = ctypes.pointer(num)
    network_predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, num_ptr)
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
    print('result:', res)

### folloing splitted test detector functions as helper for module imports

def create_network(cfg_file, weight_file):
    net = load_network(cfg_file, weight_file, 0)
    return net

def create_metadata(data_file):
    meta = get_metadata(data_file)
    return meta

def predict_image(net, meta, image, thresh, hier, nms):
    network_predict_image(net, image)
    num = ctypes.c_int(0)
    num_ptr = ctypes.pointer(num)
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
    print res
    free_detections(dets, num)
    return res