# coding=utf-8
# 该工具用来分割数据集，将数据集随机分为训练集和测试集。并提取出来自Paul、COCO和ImageNet的图片列表
import random


def partial_data_set(path):
    f = open(path)
    train_list = open('train.txt', 'w')
    val_list = open('val.txt', 'w')

    for line in f:
        if random.random()<0.2:
            val_list.write(line)
        else:
            train_list.write(line)

    f.close()
    train_list.close()
    val_list.close()

def extract_file(f_name,new_f_name,word):
    f = open(f_name)
    new_f = open(new_f_name, 'w')

    for line in f:
        if word in line:
            new_f.write(line)

    f.close()
    new_f.close()


partial_data_set('imagelist.txt')
extract_file('val.txt','imagenet_val.txt','ILSVRC2016')
extract_file('val.txt','coco_val.txt','COCO')
extract_file('val.txt','paul_val.txt','Paul')
extract_file('train.txt','imagenet_train.txt','ILSVRC2016')
extract_file('train.txt','coco_train.txt','COCO')
extract_file('train.txt','paul_train.txt','Paul')