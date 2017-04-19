# coding=utf-8

import os
import shutil


# 获取path中的name list
# path ：name 和 id的映射关系列表文件的地址
# 返回值是一个list ，索引是ID，值是name
def get_name_list(path):
    name_list = []
    f = open(path)
    for line in f:
        temp = line.rstrip().replace(' ', '').split(',', 2)
        name_list.append(temp[1])
    return name_list


# 获取图片列表中含有指定id物体的图片路径
# path是图片列表的地址
# id是指定的物体在yolo训练中的id
# 返回值是一个列表，列表内容是含有指定id物体的图片路径
def get_cat_file_list(path, id):
    cat_file_list = []
    f = open(path)
    for line in f:
        label_path = line.rstrip().replace('images', 'labels')
        label_path = label_path.replace('JPEGImages', 'labels')
        label_path = label_path.replace('.jpg', '.txt')
        label_path = label_path.replace('.JPEG', '.txt')
        label_list = open(label_path)
        for label in label_list:
            temp = label.rstrip().split(" ", 4)
            if (id == int(temp[0])):
                cat_file_list.append(line.rstrip())
                break
        label_list.close()
    f.close()
    return cat_file_list


# 将path_list中的图片拷贝到指定的目录
# dir_name是新目录路径中与原目录路径差异的部分
def copy_images(path_list, dir_name):
    for path in path_list:
        new_path = path.rstrip().replace('Data', dir_name)
        temp = new_path.split('/')
        new_dir = ''
        for i in range(0, len(temp) - 1):
            new_dir = new_dir + '/' + temp[i]

        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        shutil.copy(path.rstrip(), new_path)
        # print(new_path)


# 将path_list中的图片对应的标注文件拷贝到指定的目录
# dir_name是新目录路径中与原目录路径差异的部分
def copy_annotations(path_list, dir_name):
    for path in path_list:
        new_path = path.rstrip().replace('Data', dir_name)
        new_path = new_path.replace('images', 'annotations')
        new_path = new_path.replace('JPEGImages', 'annotations')
        new_path = new_path.replace('.jpg', '.xml')
        new_path = new_path.replace('.JPEG', '.xml')
        temp = new_path.split('/')
        new_dir = ''
        for i in range(0, len(temp) - 1):
            new_dir = new_dir + '/' + temp[i]

        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        shutil.copy(path.rstrip(), new_path)
        # print(new_path)


# 要拷贝的图片包含的物体ID
needcopy = [6, 28, 15, 25]

for id in needcopy:
    path_list = get_cat_file_list('/raid/pengchong_data/Data/filelists/imagelist.txt', id)
    name_list = get_name_list("/raid/pengchong_data/Tools/Paul_YOLO/data/paul_list.txt")
    dir_name = 'Data/CopyImages/' + name_list[id]
    print(dir_name)

    copy_images(path_list, dir_name)
    copy_annotations(path_list, dir_name)
