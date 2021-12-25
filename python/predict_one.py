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

    # spilt image into 4 slices
    s = image_slicer.slice(sys.argv[2], 4, save=False)
    for i in range(len(s)):
        s[i].save('4slice_tmp/{}.jpg'.format(i))

    pred_slice = []
    pathIn = '4slice_tmp/'
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    files.sort(key = lambda x: int(x[0:-4]))
    count = 0
    for i in range(len(files)):
        filename=pathIn + files[i]
        r = detect(net, meta, bytes(filename, 'utf-8'))
        count = count + len(r)

        img = draw("{}".format(filename), r)
        img = Image.fromarray(img)
        img.save(filename)

        pred_slice.append(Image.open(filename))

    m12 = get_concat_h(pred_slice[0], pred_slice[1])
    m34 = get_concat_h(pred_slice[2], pred_slice[3])
    im = get_concat_v(m12, m34)
    #im.save('output/pred_4slices_output.jpg')
    im = im.convert('RGB')
    im = np.array(im)
    im = display_obj_count(img=im, count=count)
    #im.save('output/pred_4slices_output.jpg')

    #im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #im = Image.fromarray(im)
    Image.fromarray(im).save('temp/pred_4slices_output.jpg')

    plt.imshow(im)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
