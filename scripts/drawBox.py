# coding=utf-8

import cv2
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets = ['lxy']

classes = ['scissors', "tree"]


def draw_box(img_path, ann_path, output_path):
    in_file = open(ann_path)
    input_img = cv2.imread(img_path)
    tree = ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        cls = obj.find('name').text
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymin').text),
             int(xmlbox.find('ymax').text))
        cv2.rectangle(input_img, (b[0], b[2]), (b[1], b[3]), (0, 0, 255))
    cv2.imwrite(out_path, input_img)


def IsSubString(SubStrList, Str):
    flag = True
    for substr in SubStrList:
        if not (substr in Str):
            flag = False

    return flag


def GetFileList(FindPath, FlagStr=[]):
    import os
    FileList = []
    FileNames = os.listdir(FindPath)
    if (len(FileNames) > 0):
        for fn in FileNames:
            if (len(FlagStr) > 0):
                if (IsSubString(FlagStr, fn)):
                    FileList.append(fn[:-4])
            else:
                FileList.append(fn)

    if (len(FileList) > 0):
        FileList.sort()

    return FileList


wd = getcwd()

for worker in sets:
    for cat in classes:
        if not os.path.exists('%s/%s/annotations/' % (worker, cat)):
            os.makedirs('%s/%s/annotations/' % (worker, cat))
        image_ids = GetFileList('' + worker + '/' + cat + '/annotations/', ['xml'])
        for image_id in image_ids:
            print(image_id)
            img_path = worker + '/' + cat + '/images/' + image_id + '.jpg'
            ann_path = worker + '/' + cat + '/annotations/' + image_id + '.xml'
            out_path = worker + '/' + cat + '/output/' + image_id + '.jpg'
            draw_box(img_path, ann_path, out_path)
