# coding=utf-8
# 本工具和category命令结合使用
# category是在detector.c中新增的命令，主要作用是生成每类物体的evalute结果
# 执行命令 ./darknet detector category cfg/paul.data cfg/yolo-paul.cfg backup/yolo-paul_final.weights
# result目录下会生成各类物体的val结果，将本工具放在result目录下执行，会print出各种物体的evalute结果，包括
# id,avg_iou,avg_correct_iou,avg_precision,avg_recall,avg_score
# result目录下会生成low_list和high_list，内容分别为精度和recall未达标和达标的物体种类


import os
from os import listdir, getcwd
from os.path import join
import shutil

# 共有多少类物体
class_num = 97


# 每类物体的验证结果
class CategoryValidation:
    id = 0  # Category id
    path = ""  # path
    total_num = 0  # 标注文件中该类bounding box的总数
    proposals_num = 0  # validate结果中共预测了多少个该类的bounding box
    correct_num = 0  # 预测正确的bounding box（与Ground-truth的IOU大于0.5且种类正确）的数量
    iou_num = 0  # 所有大于0.5的IOU的数量
    iou_sum = 0  # 所有大于0.5的IOU的IOU之和
    correct_iou_sum = 0  # 预测正确的bounding box的IOU之和
    score_sum = 0  # 所有正确预测的bounding box的概率之和
    avg_iou = 0  # 无论预测的bounding box的object的种类是否正确，所有bounding box 与最吻合的Ground-truth求出IOU，对大于0.5的IOU求平均值：avg_iou = iou_sum/iou_num
    avg_correct_iou = 0  # 对预测正确的bounding box的IOU求平均值：avg_correct_iou = correct_iou_sum/correct_num
    avg_precision = 0  # avg_precision = correct_num/proposals_num
    avg_recall = 0  # avg_recall = correct_num/total_num
    avg_score = 0  # avg_score=score_sum/correct_num

    def __init__(self, path, val_cat_num):
        self.path = path
        f = open(path)

        for line in f:
            temp = line.rstrip().replace(' ', '').split(',', 9)
            temp[1] = int(temp[1])
            self.id = temp[1]
            self.total_num = val_cat_num[self.id]
            if (self.total_num):
                break

        for line in f:
            # path, class_id, correct, prob, best_iou, xmin, ymin, xmax, ymax
            temp = line.rstrip().split(', ', 9)
            temp[1] = int(temp[1])
            temp[2] = int(temp[2])
            temp[3] = float(temp[3])
            temp[4] = float(temp[4])
            self.proposals_num = self.proposals_num + 1.00
            if (temp[2]):
                self.correct_num = self.correct_num + 1.00
                self.score_sum = self.score_sum + temp[3]
                self.correct_iou_sum = self.correct_iou_sum + temp[4]
            if (temp[4] > 0.5):
                self.iou_num = self.iou_num + 1
                self.iou_sum = self.iou_sum + temp[4]

        self.avg_iou = self.iou_sum / self.iou_num
        self.avg_correct_iou = self.correct_iou_sum / self.correct_num
        self.avg_precision = self.correct_num / self.proposals_num
        self.avg_recall = self.correct_num / self.total_num
        self.avg_score = self.score_sum / self.correct_num

        f.close()

    # 导出识别正确的图片列表
    def get_correct_list(self):
        f = open(self.path)
        new_f_name = "correct_list_" + self.id + ".txt"
        new_f = open(new_f_name, 'w')
        for line in f:
            temp = line.rstrip().split(', ', 9)
            if (temp[2]):
                new_f.write(line)
        f.close()

    # 导出识别错误的图片列表
    def get_error_list(self):
        f = open(self.path)
        new_f_name = "error_list_" + self.id + ".txt"
        new_f = open(new_f_name, 'w')
        for line in f:
            temp = line.rstrip().split(', ', 9)
            if (temp[2] == 0):
                new_f.write(line)
        f.close()

    def print_eva(self):
        print("id=%d, avg_iou=%f, avg_correct_iou=%f, avg_precision=%f, avg_recall=%f, avg_score=%f \n" % (self.id,
                                                                                                           self.avg_iou,
                                                                                                           self.avg_correct_iou,
                                                                                                           self.avg_precision,
                                                                                                           self.avg_recall,
                                                                                                           self.avg_score))


def IsSubString(SubStrList, Str):
    flag = True
    for substr in SubStrList:
        if not (substr in Str):
            flag = False

    return flag


# 获取FindPath路径下指定格式（FlagStr）的文件名列表
def GetFileList(FindPath, FlagStr=[]):
    import os
    FileList = []
    FileNames = os.listdir(FindPath)
    if (len(FileNames) > 0):
        for fn in FileNames:
            if (len(FlagStr) > 0):
                if (IsSubString(FlagStr, fn)):
                    FileList.append(fn)
            else:
                FileList.append(fn)

    if (len(FileList) > 0):
        FileList.sort()

    return FileList


# 获取所有物体种类的ROI数目
# path是图片列表的地址
# 返回值是一个list，list的索引是物体种类在yolo中的id，值是该种物体的ROI数量
def get_val_cat_num(path):
    val_cat_num = []
    for i in range(0, class_num):
        val_cat_num.append(0)

    f = open(path)
    for line in f:
        label_path = line.rstrip().replace('images', 'labels')
        label_path = label_path.replace('JPEGImages', 'labels')
        label_path = label_path.replace('.jpg', '.txt')
        label_path = label_path.replace('.JPEG', '.txt')
        label_list = open(label_path)
        for label in label_list:
            temp = label.rstrip().split(" ", 4)
            id = int(temp[0])
            val_cat_num[id] = val_cat_num[id] + 1.00
        label_list.close()
    f.close()
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


wd = getcwd()
val_result_list = GetFileList(wd, ['txt'])
val_cat_num = get_val_cat_num("/raid/pengchong_data/Data/filelists/val.txt")
name_list = get_name_list("/raid/pengchong_data/Tools/Paul_YOLO/data/paul_list.txt")
low_list = open("low_list.log", 'w')
high_list = open("high_list.log", 'w')
for result in val_result_list:
    cat = CategoryValidation(result, val_cat_num)
    cat.print_eva()
    if ((cat.avg_precision < 0.3) | (cat.avg_recall < 0.3)):
        low_list.write("id=%d, name=%s, avg_precision=%f, avg_recall=%f \n" % (cat.id, name_list[cat.id], cat.avg_precision, cat.avg_recall))
    if ((cat.avg_precision > 0.6) & (cat.avg_recall > 0.6)):
        high_list.write("id=%d, name=%s, avg_precision=%f, avg_recall=%f \n" % (cat.id, name_list[cat.id], cat.avg_precision, cat.avg_recall))

low_list.close()
high_list.close()
