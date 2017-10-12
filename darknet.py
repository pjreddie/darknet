
from ctypes import *
import math
import random

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    return (ctype * len(values))(*values)

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict_p
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

detect = lib.network_predict_p
detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network_p
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    boxes = make_boxes(net)
    probs = make_probs(net)
    num =   num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0.0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res

from tqdm import tqdm
import pdb
import sys
sys.path.insert(0, "/ais/gobi2/makarand/Software/opencv-3.2.0_install/lib/python2.7/site-packages")

import cv2
import visdom

from PIL import Image

import numpy as np
import scipy.misc
viz = visdom.Visdom()

def darknet_python():


    FRAME_DIR = '/ais/gobi5/movie4d/movie4d_dataset/frames'
    VIDEO_DIR = '/ais/gobi5/movie4d/video_clips'
    DETECT_DIR = '/ais/gobi5/movie4d/movie4d_dataset/coco_frame'
    
    # local testing
    # VIDEO_DIR = '/Users/Jarvis/Desktop/test_vid' 
    # DETECT_DIR = '/Users/Jarvis/Desktop/test_detect' 

    net = load_net('./cfg/yolo.cfg', './yolo.weights', 0)
    meta = load_meta('./cfg/coco.data')

    viz_win = None

    # for root, dirs, files in reversed(list(os.walk(VIDEO_DIR))):
    for root, dirs, files in os.walk(VIDEO_DIR):
        for video_file in tqdm(files):
            if video_file.endswith('.mp4'):

                imdb_key = root.split('/')[-1]
                clip_num = video_file.split('.')[0]
                
                detection_dir = os.path.join(DETECT_DIR, imdb_key)
                if not os.path.exists(detection_dir): os.makedirs(detection_dir)
                clip_dir = os.path.join(detection_dir, clip_num)
                if not os.path.exists(clip_dir): os.makedirs(clip_dir)

                fn = os.path.join(root, video_file)
                cam = cv2.VideoCapture(fn)

                try: cam.isOpened()
                except: 
                    print 'Error loading file...'
                    continue

                frame_num = 0

                while 1:
                    ret, img = cam.read()
                    if not ret: break # done reading the video

                    # visualization in visdom
                    # actual_img = np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
                    # actual_img = actual_img[[2, 1, 0],:,:]
                    # viz_win = viz.image(actual_img, win=viz_win)

                    if frame_num % 2 == 0: # use darknet to run detector on the frame

                        # darknet interface only supports read from image file
                        temp_fn = 'one_frame.png'
                        myimg = cv2.imread(temp_fn)
                        cv2.imwrite(temp_fn, img)

                        # darknet runs
                        r = detect(net, meta, temp_fn)
                        
                        # save the prediction
                        detection_fn = '%s.p' % str(frame_num) 
                        detection_path = os.path.join(clip_dir, detection_fn)
                        pickle.dump(r, open(detection_path, 'wb'))

                    frame_num += 1
    

    print 'All done...'
    return None

from pprint import pprint
import os
import pdb
import pickle

if __name__ == "__main__":
    
    print 'Start of main...'
    darknet_python()
    print 'End of task...'

    net = load_net('./cfg/yolo.cfg', './yolo.weights', 0)
    meta = load_meta('./cfg/coco.data')
    r = detect(net, meta, "./data/dog.jpg")
    pprint(r)
    









