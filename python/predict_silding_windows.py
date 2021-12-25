# predict with tiler

import cv2
import sys
import time

import os
import cv2
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os.path import isfile, join

import image_slicer

import configparser
from darknet import *
from image_processing import *

#if '/home/vj/Documents/tiler' not in sys.path:
#    sys.path.insert(0, '/home/vj/Documents/tiler')
if '{}/../tiler/'.format(os.getcwd()) not in sys.path:
    sys.path.insert(0, '{}/../tiler/'.format(os.getcwd()))
from tiler import *
from matplotlib import pyplot as plt

def distance(box1, box2):
    x = (box1[0] - box2[0])**2
    y = (box1[1] - box2[1])**2
    return (x+y)**0.5

def draw_cor(path, boxes):
    img = cv2.imread('{}'.format(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255, 20, 147)
    thickness = 2
    for box in boxes:
        print("box",box, type(box))
        org = (int(box[2]), int(box[3]*0.98))
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        img = cv2.putText(img, 'car_plate', org, font, fontScale, color, thickness, cv2.LINE_AA)

    return img

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
    fontScale = 0.5
    color = (255, 20, 147)
    thickness = 2

    for i in range(len(r)):
        #print(r[i][2])
        box = from_yolo_to_cor(img, r[i][2])
        #box = r[i][2]
        print(box)
        org = (int(box[2]), int(box[3]*0.98))
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        img = cv2.putText(img, r[i][0].decode("utf-8"), org, font, fontScale, color, thickness, cv2.LINE_AA)

    #plt.imshow(img)
    #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #plt.show()

    return img

def convert_frames_to_video(pathIn,pathOut,fps=25):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    #for sorting the file names properly
    files.sort(key = lambda x: int(x[6:-4]))

    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


if __name__=="__main__":

    # check
    if len(sys.argv) != 3:
        print("Usage: python3 {} <ini_file> <image>".format(sys.argv[0]))
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

    temp = 'temp/temp.jpg'
    pred_temp = 'pred.jpg'

    img = read_image(sys.argv[2])
    boxes = image_boxes(img, [416,416])
    org_boxes = []

    for i in range(len(boxes)):
        im = cv2.cvtColor(boxes[i]['img'], cv2.COLOR_BGR2RGB)
        if im.shape[0]==416 and im.shape[1]==416:
            print('res: '+str(im.shape)+'\n'+'pos: '+str(boxes[i]['pos']))
            print()
            # save a tile
            cv2.imwrite(temp, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

            # predict
            r = detect(net, meta, bytes(temp, 'utf-8'))
            #print(r)
            img = draw("{}".format(temp), r)
            img = Image.fromarray(img)
            img.save(pred_temp)

            # plt read pred.jpg
            im = cv2.imread(pred_temp)
            imb = cv2.resize(im, (600,600))
            cv2.imshow('res: '+str(im.shape)+' _ '+'pos: '+str(boxes[i]['pos']), imb)
            key = cv2.waitKey(300)
            cv2.destroyAllWindows()

            # save boxes
            for j in range(len(r)):
                box = []
                box_temp = from_yolo_to_cor(im, r[j][2])
                box.append(box_temp[0] + boxes[i]['pos'][0])
                box.append(box_temp[1] + boxes[i]['pos'][1])
                box.append(box_temp[2] + boxes[i]['pos'][0])
                box.append(box_temp[3] + boxes[i]['pos'][1])

                org_boxes.append(box)
                for b in org_boxes:
                    d = distance(b, box)
                    print(d)
                    if d<20. and d!=0.0:
                        org_boxes.pop()

            if key == ord('q'):
                break

    print(org_boxes)
    print("img", img)
    img = draw_cor(sys.argv[2], org_boxes)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('ouput', img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img.save('temp/output_tiling.jpg')
