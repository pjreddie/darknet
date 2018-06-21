# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

from darknet_libwrapper import *
#import darknet as dn
#import pdb

#dn.set_gpu(0)
#net = dn.load_net("cfg/yolov2-tiny.cfg", "yolov2-tiny.weights", 0)
#meta = dn.load_meta("cfg/coco.data")
## And then down here you could detect a lot more images like:
#r = dn.detect(net, meta, "data/eagle.jpg")
#print r
#r = dn.detect(net, meta, "data/giraffe.jpg")
#print r
#r = dn.detect(net, meta, "data/horses.jpg")
#print r
#r = dn.detect(net, meta, "data/person.jpg")
#print r

def run_detector(*argv):
    """test data cfg weight jpg"""
    if argv[2] == 'test':
        argv = [x for x in argv if x != 'test']
        argv.append('.5') #thresh
        argv.append('.5') #hier_thresh
        argv.append('.45') #nms
        test_detector(*argv)
    else:
        print('Not Implementation')

def test_detector(*argv):
    """cfg weight jpg"""
    print('test data:{2} cfg:{3} weight:{4} img:{5}'.format(*argv))
    cuda_set_device(0)
    thresh = float(argv[6])
    hier_thresh = float(argv[7])
    nms = float(argv[8])
    net = load_network(argv[3], argv[4], 0)
    meta = get_metadata(argv[2])
    im = load_image_color(argv[5], 0, 0)
    num = c_int(0)
    num_ptr = pointer(num)
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


