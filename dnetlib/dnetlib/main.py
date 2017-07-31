from ctypes import *
import math
import random
import numpy as np
import Image

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

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]
                

class BOX(Structure):
    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("w", c_float),
        ("h", c_float)]
        
class DETECTION(Structure):
    _fields_ = [
        ("box", BOX),
        ("classindex", c_int),
        ("classname", c_char_p),
        ("prob", c_float)]


class Network(object):
    net = None
    cfg = ""
    names = ""
    thresh = 0.25
    # The class "constructor" - It's actually an initializer 
    def __init__(self, net, cfg, names,thresh):
        self.net = net
        self.cfg = cfg
        self.names = names
        self.thresh = thresh

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network_p
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE


load_image_data = lib.load_image_data
load_image_data.argtypes = [c_char_p, c_int, c_int, c_int]
load_image_data.restype = IMAGE

predict_image = lib.predict
lib.predict.restype = POINTER(DETECTION)

draw_detections = lib.draw_detections_im
save_image = lib.save_image

def getdict(struct):
    result = {}
    for field, _ in struct._fields_:
         value = getattr(struct, field)
         # if the type is not a primitive and it evaluates to False ...
         if (type(value) not in [int, long, float, bool]) and not bool(value):
             # it's a null pointer
             value = None
         elif hasattr(value, "_length_") and hasattr(value, "_type_"):
             # Probably an array
             value = list(value)
         elif hasattr(value, "_fields_"):
             # Probably another struct
             value = getdict(value)
         result[field] = value
    return result
    
def classify(network,im,thresh = 0):
   if thresh == 0:
	thresh = network.thresh
   p = predict_image(network.net, im ,c_float(thresh),network.names)
   return p

def getDict(cla):
   out = []
   for a in range(0,49):
        if (cla[a].prob > 0):
           out.append(getdict(cla[a]))
   return out

def loadNetwork(cfg, weights, names,thresh=0.25):
    net = load_net(cfg, weights, 0)
    return Network(net, weights, names,thresh)

def loadImage(image):
    return load_image(image, 0, 0);

def drawDetections(image,dec):
    draw_detections(image,dec)

def saveImage(image,out):
    save_image(image,out)

if __name__ == "__main__":
    net = loadNetwork("yolo.cfg", "yolo.weights", "coco.names")
    imp = Image.open("t.png")
    #im = loadImage("dog.jpg")
    im2 = load_image_data(imp.bits,imp.size[0],imp.size[1],3);
    r = classify(net,im2)
    print getDict(r)

