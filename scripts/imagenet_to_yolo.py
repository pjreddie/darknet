# coding=utf-8

# 使用说明
#将该文件放在ILSVRC2016/bject_detection/ILSVRC目录下，并将Data文件夹重命名为JPEGImages

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join


def get_dirs():
    dirs = ['DET/train/ILSVRC2014_train_0006', 'DET/train/ILSVRC2014_train_0005', 'DET/train/ILSVRC2014_train_0004',
            'DET/train/ILSVRC2014_train_0003', 'DET/train/ILSVRC2014_train_0002', 'DET/train/ILSVRC2014_train_0001',
            'DET/train/ILSVRC2014_train_0000', 'DET/val']
    dirs_2013 = os.listdir('JPEGImages/DET/train/ILSVRC2013_train/')
    for dir_2013 in dirs_2013:
        dirs.append('DET/train/ILSVRC2013_train/' + dir_2013)
    return dirs


def get_classes_and_index(path):
    D = {}
    f = open(path)
    for line in f:
        temp = line.rstrip().split(',', 2)
        D[temp[1]] = temp[0]
    return D


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_dir, image_id):
    in_file = open('Annotations/%s/%s.xml' % (image_dir, image_id))
    obj_num = 0
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        obj_num = obj_num + 1
        if obj_num == 1:
            if not os.path.exists('labels/%s' % (image_dir)):
                os.makedirs('labels/%s' % (image_dir))
            out_file = open('labels/%s/%s.txt' % (image_dir, image_id), 'w')
        cls_id = classes[cls]
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    if obj_num>0:
        list_file = open('Lists/%s.txt' % image_dir.split('/')[-1], 'a')
        list_file.write('%s/JPEGImages/%s/%s.JPEG\n' % (wd, image_dir, image_id))
        list_file.close()


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


classes = get_classes_and_index('/mnt/large4t/pengchong_data/Tools/Yolo_paul/darknet/data/imagenet_list.txt')
dirs = get_dirs()

wd = getcwd()

for image_dir in dirs:
    if not os.path.exists('JPEGImages/' + image_dir):
        print("JPEGImages/%s dir not exist" % image_dir)
        continue
    image_ids = GetFileList('Annotations/' + image_dir, ['xml'])
    for image_id in image_ids:
        print(image_id)
        convert_annotation(image_dir, image_id)
