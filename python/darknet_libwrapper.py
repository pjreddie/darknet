"""darknet c library warpper as Python module"""
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
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("./libdarknet.so", RTLD_LOCAL)

def network_width(net_ptr):
    lib.network_width.argtypes = [c_void_p]
    lib.network_width.restype = c_int
    return lib.network_width(net_ptr)

def network_height(net_ptr):
    lib.network_height.argtypes = [c_void_p]
    lib.network_height.restype = c_int
    return lib.network_height(net_ptr)

def network_predict(net_ptr, data_ptr):
    """
    arg0: pointer of network
    arg1: pointer of input data
    return: pointer of network output data
    """
    lib.network_predict.argtypes = [c_void_p, POINTER(c_float)]
    lib.network_predict.restype = POINTER(c_float)
    return lib.network_predict(net_ptr, data_ptr)

def cuda_set_device(gpu_index):
    lib.cuda_set_device.argtypes = [c_int]
    lib.cuda_set_device(gpu_index)

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

def get_network_boxes(net_ptr, w, h, thresh, hier, map_ptr, relative, num_ptr):
    """
    arg0: pointer of network
    arg1,arg2: image size
    arg3: confidence threshold
    return: pointer of array of detection structure
    """
    lib.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
    lib.get_network_boxes.restype = POINTER(DETECTION)
    return lib.get_network_boxes(net_ptr, w, h, thresh, hier, map_ptr, relative, num_ptr)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

def free_detections(dets_ptr, num_dets):
    lib.free_detections.argtypes = [POINTER(DETECTION), c_int]
    lib.free_detections(dets_ptr, num_dets)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

def load_network(cfg_file, weight_file, clean_seen):
    """
    arg0: cfg file path
    arg1: weights file path
    arg2: is clean network seen
    return: pointer of network
    """
    lib.load_network.argtypes = [c_char_p, c_char_p, c_int]
    lib.load_network.restype = c_void_p
    return lib.load_network(cfg_file, weight_file, clean_seen)

def do_nms_obj(dets_ptr, num_dets, classes, nms):
    """
    arg0: pointer of detection array
    arg1: number of detections
    arg2: number of classes
    arg3: nms threshold
    """
    lib.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]
    lib.do_nms_obj(dets_ptr, num_dets, classes, nms)

def do_nms_sort(dets_ptr, num_dets, classes, nms):
    """
    arg0: pointer of detection array
    arg1: number of detections
    arg2: number of classes
    arg3: nms threshold
    """
    lib.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]
    lib.do_nms_sort(dets_ptr, num_dets, classes, nms)

def free_image(image):
    lib.free_image.argtypes = [IMAGE]
    lib.free_image(image)

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

def get_metadata(file_path):
    """
    arg0: dataset's data file path
    return: metadata structure
    """
    lib.get_metadata.argtypes = [c_char_p]
    lib.get_metadata.restype = METADATA
    return lib.get_metadata(file_path)

def load_image_color(image_file, resize_width=0, resize_height=0):
    """
    arg0: image file path
    arg1: postive width value if need resize width
    arg2: positive height if need resize height
    return image structure
    """
    lib.load_image_color.argtypes = [c_char_p, c_int, c_int]
    lib.load_image_color.restype = IMAGE
    return lib.load_image_color(image_file, resize_width, resize_height)

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

def network_predict_image(net_ptr, image):
    """
    arg0: pointer of network
    arg1: image sturcture
    return: pointer of network output data
    """
    lib.network_predict_image.argtypes = [c_void_p, IMAGE]
    lib.network_predict_image.restype = POINTER(c_float)
    return lib.network_predict_image(net_ptr, image)

#def classify(net, meta, im):
#    out = network_predict_image(net, im)
#    res = []
#    for i in range(meta.classes):
#        res.append((meta.names[i], out[i]))
#    res = sorted(res, key=lambda x: -x[1])
#    return res

#def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
#    im = load_image_color(image, 0, 0)
#    num = c_int(0)
#    pnum = pointer(num)
#    network_predict_image(net, im)
#    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
#    num = pnum[0]
#    if (nms):
#         do_nms_obj(dets, num, meta.classes, nms)

#    res = []
#    for j in range(num):
#        for i in range(meta.classes):
#            if dets[j].prob[i] > 0:
##                b = dets[j].bbox
#                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
#    res = sorted(res, key=lambda x: -x[1])
#    free_image(im)
#    free_detections(dets, num)
#    return res
    
#if __name__ == "__main__":
#    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
#    #im = load_image("data/wolf.jpg", 0, 0)
#    #meta = load_meta("cfg/imagenet1k.data")
#    #r = classify(net, meta, im)
#    #print r[:10]
#    net = load_net("cfg/tiny-yolo.cfg", "tiny-yolo.weights", 0)
#    meta = load_meta("cfg/coco.data")
#    r = detect(net, meta, "data/dog.jpg")
#    print r
    

