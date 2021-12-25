import cv2
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from image_processing import *

from darknet import *
import configparser

def from_yolo_to_cor(img, box):
    img_h, img_w, _ = img.shape
    # x1, y1 = ((x + witdth)/2)*img_width, ((y + height)/2)*img_height
    # x2, y2 = ((x - witdth)/2)*img_width, ((y - height)/2)*img_height
    x1, y1 = int(box[0] + (box[2])/2.0), int((box[1]) + box[3]/2.0)
    x2, y2 = int(box[0] - (box[2])/2.0), int((box[1]) - box[3]/2.0)
    return (x1, y1, x2, y2)

def draw(path, r):
    img = cv2.imread('{}'.format(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 20, 147)
    thickness = 3

    for i in range(len(r)):
        #print(r[i][2])
        box = from_yolo_to_cor(img, r[i][2])
        #box = r[i][2]
        print(box)
        org = (int(box[2]), int(box[3]*0.98))
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        img = cv2.putText(img, r[i][0].decode("utf-8"), org, font, fontScale, color, thickness, cv2.LINE_AA)

    img = display_obj_count(img, len(r))
    #plt.imshow(img)
    #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #plt.show()

    return img

if __name__ == "__main__":

    # check
    if len(sys.argv) != 4:
        print("Usage: python3 {} <ini_file> <image> <output_file_path>".format(sys.argv[0]))
        sys.exit()

    # read cfg.ini
    config = configparser.ConfigParser()
    config.read('{}'.format(sys.argv[1]))
    cfg_path = config.get("cfg", 'path')
    weights_path = config.get("weights", 'path')
    data_path = config.get("data", 'path')

    # load model
    #net = load_net(b"../cfg/yolov3_ssig.cfg", b"../../yolov3_ssig_final.weights", 0)
    #bytes(cfg_path, 'utf-8')
    net = load_net(bytes(cfg_path, 'utf-8'), bytes(weights_path, 'utf-8'), 0)
    #meta = load_meta(b"../cfg/ssig.data")
    meta = load_meta(bytes(data_path, 'utf-8'))

    # b"../data/car_plate1/slices/0.jpg"
    r = detect(net, meta, bytes(sys.argv[2], 'utf-8'))
    print(r)

    count = len(r)
    img = draw("{}".format(sys.argv[2]), r)

    img = Image.fromarray(img)
    if sys.argv[3]!=None:
        img.save(sys.argv[3])
