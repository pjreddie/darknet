"""darknet c library warpper as Python module"""

import ctypes
import math
import random
# For dll open path from file relative or caller working directory
import os
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libdarknet.so')
if os.path.exists(lib_path):
    pass
elif os.path.exists(os.path.join(os.getcwd(), 'libdarknet.so')):
    lib_path = os.path.join(os.getcwd(), 'libdarknet.so')
else:
    print 'libdarknet.so can not be found!'

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

class BOX(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float),
                ("y", ctypes.c_float),
                ("w", ctypes.c_float),
                ("h", ctypes.c_float)]

class DETECTION(ctypes.Structure):
    _fields_ = [("bbox", BOX),
                ("classes", ctypes.c_int),
                ("prob", ctypes.POINTER(ctypes.c_float)),
                ("mask", ctypes.POINTER(ctypes.c_float)),
                ("objectness", ctypes.c_float),
                ("sort_class", ctypes.c_int)]


class IMAGE(ctypes.Structure):
    _fields_ = [("w", ctypes.c_int),
                ("h", ctypes.c_int),
                ("c", ctypes.c_int),
                ("data", ctypes.POINTER(ctypes.c_float))]

class METADATA(ctypes.Structure):
    _fields_ = [("classes", ctypes.c_int),
                ("names", ctypes.POINTER(ctypes.c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = ctypes.CDLL(lib_path, ctypes.RTLD_LOCAL)

def network_width(net_ptr):
    """Get network width"""
    lib.network_width.argtypes = [ctypes.c_void_p]
    lib.network_width.restype = ctypes.c_int
    return lib.network_width(net_ptr)

def network_height(net_ptr):
    """Get network height"""
    lib.network_height.argtypes = [ctypes.c_void_p]
    lib.network_height.restype = ctypes.c_int
    return lib.network_height(net_ptr)

def network_predict(net_ptr, data_ptr):
    """
    arg0: pointer of network
    arg1: pointer of input data
    return: pointer of network output data
    """
    lib.network_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    lib.network_predict.restype = ctypes.POINTER(ctypes.c_float)
    return lib.network_predict(net_ptr, data_ptr)

def cuda_set_device(gpu_index):
    """Initialize CUDA device memory"""
    lib.cuda_set_device.argtypes = [ctypes.c_int]
    lib.cuda_set_device(gpu_index)

def make_image(width, height, channel):
    """
    Create IMAGE struct and allocates data pointer, 
    required call free_image to release memory after using it 
    """
    lib.make_image.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.make_image.restype = IMAGE
    return lib.make_image(width, height, channel)

def get_network_boxes(net_ptr, w, h, thresh, hier, map_ptr, relative, num_ptr):
    """
    arg0: pointer of network
    arg1,arg2: image size
    arg3: confidence threshold
    return: pointer of array of DETECTION structure
    """
    lib.get_network_boxes.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int)]
    lib.get_network_boxes.restype = ctypes.POINTER(DETECTION)
    return lib.get_network_boxes(net_ptr, w, h, thresh, hier, map_ptr, relative, num_ptr)

# libdarknet.so did not exported, use get_network_boxes instead
#make_network_boxes = lib.make_network_boxes
#make_network_boxes.argtypes = [ctypes.c_void_p]
#make_network_boxes.restype = ctypes.POINTER(DETECTION)

def free_detections(dets_ptr, num_dets):
    """Release array of DETECTION struct"""
    lib.free_detections.argtypes = [ctypes.POINTER(DETECTION), ctypes.c_int]
    lib.free_detections(dets_ptr, num_dets)

def free_ptrs(void_ptr, size):
    """Release void pointer with given bytes size(c style)"""
    lib.free_ptrs.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    lib.free_ptrs(void_ptr, size)

def reset_network_state(net_ptr, layer_index):
    """Cleanup output of given network layer"""
    lib.reset_network_state.argtypes[ctypes.c_void_p, ctypes.c_int]
    lib.reset_network_state(net_ptr, layer_index)

# libdarknet.so did not exported, use reset_network_state given layer 0 instead
def reset_rnn(net_ptr):
    reset_network_state(net_ptr, 0)
#reset_rnn = lib.reset_rnn
#reset_rnn.argtypes = [ctypes.c_void_p]

def load_network(cfg_file, weight_file, clean_seen):
    """
    arg0: cfg file path
    arg1: weights file path
    arg2: is clean network seen
    return: pointer of network
    """
    lib.load_network.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.load_network.restype = ctypes.c_void_p
    return lib.load_network(cfg_file, weight_file, clean_seen)

def do_nms_obj(dets_ptr, num_dets, classes, nms):
    """
    arg0: pointer of detection array
    arg1: number of detections
    arg2: number of classes
    arg3: nms threshold
    """
    lib.do_nms_obj.argtypes = [ctypes.POINTER(DETECTION), ctypes.c_int, ctypes.c_int, ctypes.c_float]
    lib.do_nms_obj(dets_ptr, num_dets, classes, nms)

def do_nms_sort(dets_ptr, num_dets, classes, nms):
    """
    arg0: pointer of detection array
    arg1: number of detections
    arg2: number of classes
    arg3: nms threshold
    """
    lib.do_nms_sort.argtypes = [ctypes.POINTER(DETECTION), ctypes.c_int, ctypes.c_int, ctypes.c_float]
    lib.do_nms_sort(dets_ptr, num_dets, classes, nms)

def free_image(image):
    """Release IMAGE struct included allocated data pointer"""
    lib.free_image.argtypes = [IMAGE]
    lib.free_image(image)

def letterbox_image(image, width, height):
    """Resize to given size kept ratio"""
    lib.letterbox_image.argtypes = [IMAGE, ctypes.c_int, ctypes.c_int]
    lib.letterbox_image.restype = IMAGE
    return lib.letterbox_image(image, width, height)

def get_metadata(file_path):
    """
    arg0: dataset's data file path
    return: metadata structure
    """
    lib.get_metadata.argtypes = [ctypes.c_char_p]
    lib.get_metadata.restype = METADATA
    return lib.get_metadata(file_path)

def load_image_color(image_file, resize_width=0, resize_height=0):
    """
    arg0: image file path
    arg1: postive width value if need resize width
    arg2: positive height if need resize height
    return image structure
    """
    lib.load_image_color.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.load_image_color.restype = IMAGE
    return lib.load_image_color(image_file, resize_width, resize_height)

def rgbgr_image(image):
    lib.rgbgr_image.argtypes = [IMAGE]
    lib.rgbgr_image(image)

def network_predict_image(net_ptr, image):
    """
    arg0: pointer of network
    arg1: image sturcture
    return: pointer of network output data
    """
    lib.network_predict_image.argtypes = [ctypes.c_void_p, IMAGE]
    lib.network_predict_image.restype = ctypes.POINTER(ctypes.c_float)
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
