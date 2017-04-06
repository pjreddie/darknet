# coding=utf-8

import cv2
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets = ['ImageNet','20170401']

#classes = ['door', 'jack', 'respirator']

wd = getcwd()


def get_dirs(sub_dir):
    dirs = []
    dirs_name = os.listdir(wd+'/'+sub_dir)
    for dir_name in dirs_name:
        dirs.append(dir_name)
    return dirs


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
        cv2.rectangle(input_img, (b[0], b[2]), (b[1], b[3]), (0, 0, 255), thickness=2)
        # cv2.putText(input_img, cls, (b[0], b[2]), cv2.FONT_HERSHEY_COMPLEX_SMALL, (0, 255, 0))
        cv2.putText(input_img, cls, (b[0], b[2]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), thickness=2)
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


for worker in sets:
    dirs = get_dirs(worker)
    for cat in dirs:
        if not os.path.exists('%s/%s/annotations/' % (worker, cat)):
            os.makedirs('%s/%s/annotations/' % (worker, cat))
        if not os.path.exists('%s/%s/output/' % (worker, cat)):
            os.makedirs('%s/%s/output/' % (worker, cat))
        image_ids = GetFileList(worker + '/' + cat + '/annotations/', ['xml'])
        for image_id in image_ids:
            print(image_id)
            img_path = worker + '/' + cat + '/images/' + image_id + '.jpg'
            ann_path = worker + '/' + cat + '/annotations/' + image_id + '.xml'
            out_path = worker + '/' + cat + '/output/' + image_id + '.jpg'
            #print(img_path)
            #print(out_path)
            draw_box(img_path, ann_path, out_path)
