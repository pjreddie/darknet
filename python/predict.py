import cv2
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from darknet import *

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
    fontScale = 0.8
    color = (255, 0, 0)
    thickness = 2

    for i in range(len(r)):
        #print(r[i][2])
        box = from_yolo_to_cor(img, r[i][2])
        #box = r[i][2]
        print(box)
        org = (int(box[2]), int(box[3]*0.98))
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0), 2)
        img = cv2.putText(img, r[i][0].decode("utf-8"), org, font, fontScale, color, thickness, cv2.LINE_AA)

    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    return img

if __name__ == "__main__":
    net = load_net(b"../cfg/yolov3_ssig.cfg", b"../../../Documents/darknet/backup/yolov3_ssig_final.weights", 0)
    meta = load_meta(b"../cfg/ssig.data")

    # b"../data/car_plate1/slices/0.jpg"
    r = detect(net, meta, bytes(sys.argv[1], 'utf-8'))
    img = draw("{}".format(sys.argv[1]), r)
    img = Image.fromarray(img)

    if sys.argv[2]!=None:
        img.save(sys.argv[2])
    print(r)
