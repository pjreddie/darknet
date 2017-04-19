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
import shutil

sets = ['20170401', '20170414']


# 获取所需要的类名和id
# path为类名和id的对应关系列表的地址（标注文件中可能有很多类，我们只加载该path指向文件中的类）
# 返回值是一个字典，键名是类名，键值是id
def get_classes_and_index(path):
    D = {}
    f = open(path)
    for line in f:
        temp = line.rstrip().split(',', 2)
        print("temp[0]:" + temp[0] + "\n")
        print("temp[1]:" + temp[1] + "\n")
        D[temp[1].replace(' ', '')] = temp[0]
    return D


# 将ROI的坐标转换为yolo需要的坐标
# size是图片的w和h
# box里保存的是ROI的坐标（x，y的最大值和最小值）
# 返回值为ROI中心点相对于图片大小的比例坐标，和ROI的w、h相对于图片大小的比例
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


# 将labelImg 生成的xml文件转换为yolo需要的txt文件
# path到类名一级的目录路径
# image_id图片名
def convert_annotation(path, image_id):
    in_file = open('%s/annotations/%s.xml' % (path, image_id))
    out_file = open('%s/labels/%s.txt' % (path, image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text.replace(' ', '')
        # 如果该类物体不在我们的yolo训练列表中，跳过
        if cls not in classes:
            continue
        cls_id = classes[cls]  # 获取该类物体在yolo训练列表中的id
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


# 获取FindPath路径下指定格式（FlagStr）的文件名（不包含后缀名）列表
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


# 获取目录下子目录的目录名列表
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
    list_file = open('%s.txt' % time, 'w')  # 数据集的图片list保存路径
    for path in dirs:
        print(path)
        if not os.path.exists('%s/annotations/' % path):
            os.makedirs('%s/annotations/' % path)
        if not os.path.exists('%s/labels/' % path):
            os.makedirs('%s/labels/' % path)
        else:
            shutil.rmtree('%s/labels/' % path)
            os.makedirs('%s/labels/' % path)
        image_ids = GetFileList(path + '/annotations/', ['xml'])
        for image_id in image_ids:
            print(image_id)
            list_file.write('%s/%s/images/%s.jpg\n' % (wd, path, image_id))
            convert_annotation(path, image_id)
    list_file.close()
