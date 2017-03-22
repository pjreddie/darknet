# coding=utf-8

# 使用说明

# 要转换的数据集目录结构为：
# BYD/time/annotations/xml文件
# BYD/time/images/jpg文件
# BYD/time/labels/即将生成的yolo需要的txt文件

# 该文件需放在BYD同级目录下，该目录下将会生成名为“日期”的txt文件，文件内容为日期文件夹下所有图片的路径

# 有多少个日期的文件夹，就将多少个文件夹的名字加入sets

# 需要生成多少种物体的标签，就将多少种物体加入classes
# labels目录下生成的txt文件中的第一个数字就是物体种类在classes中的索引


import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets = ['20170221']
#dirs = ['DET/train/ILSVRC2014_train_0006', 'DET/train/ILSVRC2014_train_0005', 'DET/train/ILSVRC2014_train_0004',
#        'DET/train/ILSVRC2014_train_0003', 'DET/train/ILSVRC2014_train_0002', 'DET/train/ILSVRC2014_train_0001',
#        'DET/train/ILSVRC2014_train_0000', 'DET/val']

dirs_2013 = os.listdir('Data/DET/train/ILSVRC2013_train/')
dirs = []

for dir_2013 in dirs_2013:
    dirs.append('DET/train/ILSVRC2013_train/'+dir_2013)


classes = ['n07739125', 'n02769748', 'n02778669', 'n02779435', 'n07753592', 'n02799071', 'n02802426', 'n02828884',
           'n02834778', 'n01503061', 'n02870526', 'n02871439', 'n02876657', 'n02880940', 'n02881193', 'n02881546',
           'n07714990', 'n02274259', 'n02958343', 'n07730207', 'n02992529', 'n03001627', 'n03017168', 'n03046257',
           'n03063338', 'n03085013', 'n03793489', 'n03141823', 'n07718472', 'n07930864', 'n02084071', 'n03221720',
           'n03222176', 'n03222318', 'n03249569', 'n03249956', 'n03255030', 'n03483316', 'n07690152', 'n03481172',
           'n02774152', 'n07714571', 'n03513137', 'n03513376', 'n07697537', 'n03613294', 'n03613592', 'n03636248',
           'n03636649', 'n03642806', 'n05716342', 'n03759954', 'n04965179', 'n03908618', 'n00007846', 'n07753275',
           'n03942813', 'n07873807', 'pomegranate', 'n03995372', 'n04004767', 'n04039381', 'n04070727', 'n04074963',
           'n04154565', 'n04254680', 'n04256520', 'n07745940', 'n04356056', 'n04379243', 'n04379964', 'n04404412',
           'n04409515', 'n04433377', 'n04507155', 'n04522168', 'n04540053', 'n04587648', 'n04588739']


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
    if not os.path.exists('Labels/%s'%(image_dir)):
        os.makedirs('Labels/%s'%(image_dir))
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
            out_file = open('Labels/%s/%s.txt' % (image_dir, image_id), 'w')
        cls_id = classes.index(cls)
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


wd = getcwd()

for image_dir in dirs:
    if not os.path.exists('Data/' + image_dir):
        print("Data/%s dir not exist"%image_dir)
        continue
    image_ids = GetFileList('Annotations/' + image_dir, ['xml'])
    list_file = open('%s.txt' % image_dir.split('/')[-1], 'w')
    for image_id in image_ids:
        print(image_id)
        list_file.write('%s/Data/%s/%s.JPEG\n' % (wd, image_dir, image_id))
        convert_annotation(image_dir, image_id)
    list_file.close()
