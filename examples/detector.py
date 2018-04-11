from __future__ import print_function

import argparse
import os
import skimage.io
import sys
import time

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
NET_DIR = os.path.realpath(os.path.join(ROOT_DIR, 'python/'))

os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(NET_DIR)

import darknet as dn
import pdb


def init_network(cfg_file=bytes(os.path.join(ROOT_DIR, "cfg/yolov3.cfg"), encoding='utf-8'),
                 weights_file=b"yolov3.weights",
                 data_cfg=bytes(os.path.join(ROOT_DIR, "cfg/coco.data"), encoding='utf-8'),
                 use_gpu=False):

    # Check if files exist
    if not os.path.exists(cfg_file):
        raise ValueError("ERROR: cfg_file does not exist! Given:", cfg_file)

    if not os.path.exists(weights_file):
        raise ValueError("ERROR: weights_file does not exist! Given:", weights_file)

    if not os.path.exists(data_cfg):
        raise ValueError("ERROR: data_cfg does not exist! Given:", data_cfg)

    if type(weights_file) is str:
        weights_file = bytes(os.path.realpath(weights_file), encoding='utf-8')

    if type(cfg_file) is str:
        cfg_file = bytes(os.path.realpath(cfg_file), encoding='utf-8')

    if type(data_cfg) is str:
        data_cfg = bytes(os.path.realpath(data_cfg), encoding='utf-8')

    # Don't use GPU?
    if use_gpu:
        dn.set_gpu(1)
    else:
        dn.set_gpu(0)

    # Load the network
    net = dn.load_net(cfg_file, weights_file, 0)

    # Load metadata for labels
    meta = dn.load_meta(data_cfg)

    return net, meta


def detect_using_yolo(net, meta, image_file):

    if not os.path.exists(image_file):
        raise ValueError("ERROR: image_file does not exist! Given:", image_file)

    # Check if image_file is string or bytes
    if type(image_file) is str:
        image_file = bytes(image_file, encoding='utf-8')

    r = dn.detect(net, meta, image_file)

    return r


if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='YOLO using python')
    parser.add_argument('--weights_file', '-w', type=str, required=True, help='Full path to the weights file, yolov3.weights (or other yolo weights)')
    parser.add_argument('--cfg_file', type=str, default=os.path.join(ROOT_DIR, 'cfg/yolov3.cfg'), help="Full path to the desired config file")
    parser.add_argument('--data_cfg', type=str, default=os.path.join(ROOT_DIR, 'cfg/coco.data'), help="Full path to the desired data file (for object labels)")
    parser.add_argument('--use_gpu', '-g', action='store_true', help="Use GPU or not")
    parser.add_argument('--image', '-i', type=str, default=None, help="Input image to detect objects in, or text file containing paths to images")
    args  = parser.parse_args()

    print(args)

    if args.image == None:
        args.image = b"data/dog.jpg"

    try:
        # Load network
        net, meta = init_network(cfg_file=args.cfg_file, weights_file=args.weights_file, data_cfg=args.data_cfg, use_gpu=args.use_gpu)
       
        # Try detecting
        r = dn.detect(net, meta, args.image)
        print(r)
    
    except ValueError as e:
        print(e)
        sys.exit()
    
