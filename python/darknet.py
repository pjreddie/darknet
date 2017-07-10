from ctypes import *

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

def load_meta(f):
    lib.get_metadata.argtypes = [c_char_p]
    lib.get_metadata.restype = METADATA
    return lib.get_metadata(f)

def load_net(cfg, weights):
    load_network = lib.load_network_p
    load_network.argtypes = [c_char_p, c_char_p, c_int]
    load_network.restype = c_void_p
    return load_network(cfg, weights, 0)

def load_img(f):
    load_image = lib.load_image_color
    load_image.argtypes = [c_char_p, c_int, c_int]
    load_image.restype = IMAGE
    return load_image(f, 0, 0)

def letterbox_img(im, w, h):
    letterbox_image = lib.letterbox_image
    letterbox_image.argtypes = [IMAGE, c_int, c_int]
    letterbox_image.restype = IMAGE
    return letterbox_image(im, w, h)

def predict(net, im):
    pred = lib.network_predict_image
    pred.argtypes = [c_void_p, IMAGE]
    pred.restype = POINTER(c_float)
    return pred(net, im)

def classify(net, meta, im):
    out = predict(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, im):
    out = predict(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

if __name__ == "__main__":
    net = load_net("cfg/densenet.cfg", "/home/pjreddie/trained/densenet201.weights")
    im = load_img("data/wolf.jpg")
    meta = load_meta("cfg/imagenet1k.data")
    r = classify(net, meta, im)
    print r[:10]

