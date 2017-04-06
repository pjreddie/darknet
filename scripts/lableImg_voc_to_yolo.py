# coding=utf-8

# 使用说明

# 要转换的数据集目录结构为：
# Paul/time/class/annotations/xml文件
# Paul/time/class/images/jpg文件
# Paul/time/class/labels/即将生成的yolo需要的txt文件

# 该文件需放在Paul目录下，该目录下将会生成名为“日期”的txt文件，文件内容为日期文件夹下所有图片的路径

# 有多少个日期的文件夹，就将多少个文件夹的名字加入sets

# 需要生成多少种物体的标签，就将多少种物体加入classes
# labels目录下生成的txt文件中的第一个数字就是物体种类在classes中的索引


import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets = ['20170401', 'ImageNet']


# classes = ['dog','person','car','train','sofa']

def get_classes_and_index(path):
    D = {}
    f = open(path)
    for line in f:
        temp = line.rstrip().split(',', 2)
        print("temp[0]:" + temp[0] + "\n")
        print("temp[1]:" + temp[1] + "\n")
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


def convert_annotation(path, image_id):
    if not os.path.exists('%s/labels/' % path):
        os.makedirs('%s/labels/' % path)
    in_file = open('%s/annotations/%s.xml' % (path, image_id))
    out_file = open('%s/labels/%s.txt' % (path, image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes[cls]
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


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


def get_dirs(time):
    dirs = []
    dirs_temp = os.listdir(time)
    for dir_name in dirs_temp:
        dirs.append(time + '/' + dir_name)
    return dirs


wd = getcwd()

classes = get_classes_and_index('/raid/pengchong_data/Tools/Paul_YOLO/data/Paul_list.txt')

for time in sets:
    dirs = get_dirs(time)
    list_file = open('%s.txt' % time, 'w')
    for path in dirs:
        print(path)
        if not os.path.exists('%s/annotations/' % path):
            os.makedirs('%s/annotations/' % path)
        image_ids = GetFileList(path + '/annotations/', ['xml'])
        for image_id in image_ids:
            print(image_id)
            list_file.write('%s/%s/images/%s.jpg\n' % (wd, path, image_id))
            convert_annotation(path, image_id)
    list_file.close()
