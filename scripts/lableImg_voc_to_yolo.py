# coding=utf-8

#使用说明

#要转换的数据集目录结构为：
#BYD/time/annotations/xml文件
#BYD/time/images/jpg文件
#BYD/time/labels/即将生成的yolo需要的txt文件

#该文件需放在BYD同级目录下，该目录下将会生成名为“日期”的txt文件，文件内容为日期文件夹下所有图片的路径

#有多少个日期的文件夹，就将多少个文件夹的名字加入sets

#需要生成多少种物体的标签，就将多少种物体加入classes
# labels目录下生成的txt文件中的第一个数字就是物体种类在classes中的索引


import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=['20170221']

classes = ['dog','person','car','train','sofa']


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(time, image_id):
    in_file = open('BYD/%s/annotations/%s.xml'%(time, image_id))
    out_file = open('BYD/%s/labels/%s.txt'%(time, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def IsSubString(SubStrList,Str):
    flag=True
    for substr in SubStrList:
        if not(substr in Str):
            flag=False

    return flag


def GetFileList(FindPath,FlagStr=[]):
    import os
    FileList=[]
    FileNames=os.listdir(FindPath)
    if (len(FileNames)>0):
       for fn in FileNames:
           if (len(FlagStr)>0):
               if (IsSubString(FlagStr,fn)):
                   FileList.append(fn[:-4])
           else:
               FileList.append(fn)

    if (len(FileList)>0):
        FileList.sort()

    return FileList

wd = getcwd()

for time in sets:
    if not os.path.exists('BYD/%s/annotations/'%time):
        os.makedirs('BYD/%s/annotations/'%time)
    image_ids=GetFileList('BYD/'+time+'/annotations/',['xml'])
    list_file = open('%s.txt'%time, 'w')
    for image_id in image_ids:
        print(image_id)
        list_file.write('%s/BYD/%s/images/%s.jpg\n'%(wd, time, image_id))
        convert_annotation(time, image_id)
    list_file.close()

