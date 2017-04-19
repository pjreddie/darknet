# coding=utf-8
# 该工具用来统计数据集中每类物体有多少图片和ROI
# 使用该工具会生成num_list.log，内容为物体id，名字，ROI数目和图片数目
# 使用该工具会生成less_list.log，内容为图片数目少于1000的物体id，名字，ROI数目和图片数目

import os
from os import listdir, getcwd
from os.path import join
import shutil

# 共有多少种物体
class_num = 97


# 获取标记文件list
# path为图片路径list文件地址
# 返回值是一个列表，列表里保存了所有的标记文件的路径
def get_label_path_list(path):
    label_path_list = []
    f = open(path)
    for line in f:
        label_path = line.rstrip().replace('images', 'labels')
        label_path = label_path.replace('JPEGImages', 'labels')
        label_path = label_path.replace('.jpg', '.txt')
        label_path = label_path.replace('.JPEG', '.txt')
        label_path_list.append(label_path)
    return label_path_list


# 获取每类物体ROI的数量
# label_path_list是标记文件list
# 返回值是一个列表，列表的索引是类的id，值为该类物体的ROI数量
def get_cat_roi_num(label_path_list):
    val_cat_num = []
    for i in range(0, class_num):
        val_cat_num.append(0)

    for line in label_path_list:
        label_list = open(line)
        for label in label_list:
            temp = label.rstrip().split(" ", 4)
            id = int(temp[0])
            val_cat_num[id] = val_cat_num[id] + 1
        label_list.close()
    return val_cat_num


# 获取每类物体的图片数量
# label_path_list是标记文件list
# 返回值是一个列表，列表的索引是类的id，值为该类物体的图片数量
def get_cat_file_num(label_path_list):
    val_cat_num = []

    for i in range(0, class_num):
        val_cat_num.append(0)

    for line in label_path_list:
        label_list = open(line)

        flags = []
        for i in range(0, class_num):
            flags.append(0)

        for label in label_list:
            id = int(label.rstrip().split(" ", 4)[0])
            if (id < class_num):
                flags[id] = 1

        for i in range(0, class_num):
            if (flags[i] == 1):
                val_cat_num[i] = val_cat_num[i] + 1

        label_list.close()
    return val_cat_num


# 获取物体名list
# path是物体名list文件地址
# 返回值是一个列表，列表的索引是类的id，值为该类物体的名字
def get_name_list(path):
    name_list = []
    f = open(path)
    for line in f:
        temp = line.rstrip().split(',', 2)
        name_list.append(temp[1])
    return name_list


path = "/raid/pengchong_data/Data/filelists/imagelist.txt"
label_path_list = get_label_path_list(path)
name_list = get_name_list("/raid/pengchong_data/Tools/Paul_YOLO/data/paul_list.txt")
cat_roi_num = get_cat_roi_num(label_path_list)
cat_file_num = get_cat_file_num(label_path_list)

num_list = open("num_list.log", 'w')
less_list = open("less_list.log", 'w')

for i in range(0, class_num):
    print(i)
    num_list.write("%d, %s, %d, %d \n" % (i, name_list[i], cat_roi_num[i], cat_file_num[i]))
    if (cat_file_num[i] < 1000):
        less_list.write("%d, %s, %d, %d \n" % (i, name_list[i], cat_roi_num[i], cat_file_num[i]))

num_list.close()
less_list.close()
