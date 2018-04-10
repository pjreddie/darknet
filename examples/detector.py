from __future__ import print_function

import argparse
import os
import sys

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
NET_DIR = os.path.realpath(os.path.join(ROOT_DIR, 'python/'))

sys.path.append(ROOT_DIR)
sys.path.append(NET_DIR)

import darknet as dn
import pdb

# Don't use GPU?
dn.set_gpu(0)


def init_network(cfg_file=bytes(os.path.join(ROOT_DIR, "cfg/yolov3.cfg"), encoding='utf-8'),
                 weights_file=b"yolov3.weights",
                 data_cfg=bytes(os.path.join(ROOT_DIR, "cfg/coco.data"), encoding='utf-8')):

    # Check if files exist
    if not os.path.exists(cfg_file):
        raise ValueError("ERROR: cfg_file does not exist! Given:", cfg_file)

    if not os.path.exists(weights_file):
        raise ValueError("ERROR: weights_file does not exist! Given:", weights_file)

    if not os.path.exists(data_cfg):
        raise ValueError("ERROR: data_cfg does not exist! Given:", data_cfg)

    # Load the network
    net = dn.load_net(cfg_file, weights_file, 0)

    # Load metadata for labels
    meta = dn.load_meta(data_cfg)

    return net, meta


if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='YOLO using python')
    parser.add_argument('--weights_file', '-w', type=str, required=True, help='Full path to the weights file, yolov3.weights (or other yolo weights)')
    parser.add_argument('--cfg_file', type=str, default=os.path.join(ROOT_DIR, "cfg/yolov3.cfg"), help="Full path to the desired config file")
    parser.add_argument('--data_cfg', type=str, default=os.path.join(ROOT_DIR, "cfg/coco.data"), help="Full path to the desired data file (for object labels)")
    args  = parser.parse_args()

    print(args)

    weights_file = bytes(os.path.realpath(args.weights_file), encoding='utf-8')
    print(os.path.realpath(args.cfg_file), args.cfg_file)
    cfg_file = bytes(os.path.realpath(args.cfg_file), encoding='utf-8')
    data_cfg = bytes(os.path.realpath(args.data_cfg), encoding='utf-8')

    # Load network
    try:
        net, meta = init_network(cfg_file=cfg_file, weights_file=weights_file, data_cfg=data_cfg)
    except ValueError as e:
        print(e)
        sys.exit()
    
    # Try detecting
    r = dn.detect(net, meta, b"data/dog.jpg")
    print(r)
    
    # And then down here you could detect a lot more images like:
    r = dn.detect(net, meta, b"data/eagle.jpg")
    print(r)
    r = dn.detect(net, meta, b"data/giraffe.jpg")
    print(r)
    r = dn.detect(net, meta, b"data/horses.jpg")
    print(r)
    r = dn.detect(net, meta, b"data/person.jpg")
    print(r)

