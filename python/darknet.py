#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Darknet pytno interface
"""

import argparse
from ctypes import c_char_p, c_float, c_int, c_void_p, pointer
from ctypes import CDLL, POINTER, RTLD_GLOBAL, Structure
import random


def sample(probs):
    """
    sample function
    """

    probs_sum = sum(probs)
    probs = [a/probs_sum for a in probs]
    rand = random.uniform(0, 1)
    for idx, prob in enumerate(probs):
        rand = rand - prob
        if rand <= 0:
            return idx
    return len(probs)-1


def c_array(ctype, values):
    """
    convert to carray from value
    """

    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    """
    Structure definision of BBOX
    """

    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    """
    Structure definision of DETECTION
    """

    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    """
    Structure definision of IMAGE
    """

    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    """
    Structure definision of META DATA
    """

    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


class Darknet(object):
    """
    Darknet class
    """

    def __init__(self,
                 libfilepath,
                 cfgfilepath,
                 datafilepath,
                 weightsfilepath):
        """
        Initialize metod
        """

        self.libfilepath = libfilepath
        self.cfgfilepath = cfgfilepath
        self.datafilepath = datafilepath
        self.weightsfilepath = weightsfilepath

        self.net = None
        self.meta = None

        self.lib = CDLL(self.libfilepath, RTLD_GLOBAL)
        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int

        self.predict = self.lib.network_predict
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        self.set_gpu = self.lib.cuda_set_device
        self.set_gpu.argtypes = [c_int]

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [
            c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.make_network_boxes = self.lib.make_network_boxes
        self.make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict = self.lib.network_predict
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.do_nms_obj = self.lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = self.lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = self.lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

    def load_conf(self):
        """
        loading network from weights file
        """
        self.net = self.load_net(self.cfgfilepath,
                                 self.weightsfilepath,
                                 0)
        self.meta = self.load_meta(self.datafilepath)

    def detect(self, imagefilepath, thresh=.5, hier_thresh=.5, nms=.45):
        """
        detecting
        """

        image = self.load_image(imagefilepath, 0, 0)
        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(self.net, image)
        dets = self.get_network_boxes(
            self.net, image.w, image.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if nms:
            self.do_nms_obj(dets, num, self.meta.classes, nms)

        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if dets[j].prob[i] > 0:
                    bbox = dets[j].bbox
                    res.append(
                        (self.meta.names[i], dets[j].prob[i], (bbox.x, bbox.y, bbox.w, bbox.h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_image(image)
        self.free_detections(dets, num)
        return res


    def classify(self, imagefilepath):
        """
        classify
        """

        image = self.load_image(imagefilepath)
        out = self.predict_image(self.net, image)
        res = []
        for i in range(self.meta.classes):
            res.append((self.meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res


def importargs():
    """
    Get arguments
    """

    parser = argparse.ArgumentParser("This Darknet python sample")
    parser.add_argument("--libfilepath", "-lf",
                        default="./libdarknet.so",
                        type=str,
                        help="filepath of libdarknet.default:./libdarknet.so")

    parser.add_argument("--cfgfilepath", "-cf",
                        default="./cfg/yolov3.cfg",
                        type=str,
                        help="cfgfilepath.default ./cfg/yolov3.cfg")

    parser.add_argument("--datafilepath", "-df",
                        default="./cfg/coco.data",
                        type=str,
                        help="datafilepath.default: ./cfg/coco.data")

    parser.add_argument("--weightsfilepath", "-wf",
                        default="./yolov3.weights",
                        type=str,
                        help="weightsfilepath.default: ./yolov3.weights")

    parser.add_argument("--imagefilepath", "-if",
                        default="./data/dog.jpg",
                        type=str,
                        help="imagefilepath.default: ./data/dog.jpg")

    args = parser.parse_args()

    return args.libfilepath, args.cfgfilepath, \
        args.datafilepath, args.weightsfilepath, args.imagefilepath


def main():
    """
    main
    """

    libfilepath, cfgfilepath, \
        datafilepath, weightsfilepath, imgfilepath = importargs()

    darknet = Darknet(libfilepath=libfilepath,
                      cfgfilepath=cfgfilepath.encode(),
                      weightsfilepath=weightsfilepath.encode(),
                      datafilepath=datafilepath.encode())

    darknet.load_conf()
    res = darknet.detect(imgfilepath.encode())
    print(res)
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    # print r[:10]
    #net = load_net("cfg/tiny-yolo.cfg", "tiny-yolo.weights", 0)
    #meta = load_meta("cfg/coco.data")
    #r = detect(net, meta, "data/dog.jpg")
    # print r


if __name__ == "__main__":
    main()
