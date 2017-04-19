# coding=utf-8
# 使用说明
# 需要先安装coco tools
# git clone https://github.com/pdollar/coco.git
# cd coco/PythonAPI
# make install(可能会缺少相关依赖，根据提示安装依赖即可)
# 执行脚本前需在train2014和val2014目录下分别创建JPEGImages和labels目录，并将原来train2014和val2014目录下的图片移到JPEGImages下
# COCO数据集的filelist目录下会生成图片路径列表
# COCO数据集的子集的labels目录下会生成yolo需要的标注文件


from pycocotools.coco import COCO
import shutil
import os


# 将ROI的坐标转换为yolo需要的坐标
# size是图片的w和h
# box里保存的是ROI的坐标（x，y的最大值和最小值）
# 返回值为ROI中心点相对于图片大小的比例坐标，和ROI的w、h相对于图片大小的比例
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


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
        D[temp[1]] = temp[0]
    return D


dataDir = '/mnt/large4t/pengchong_data/Data/COCO'  # COCO数据集所在的路径
dataType = 'train2014'  # 要转换的COCO数据集的子集名
annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)  # COCO数据集的标注文件路径
classes = get_classes_and_index('/mnt/large4t/pengchong_data/Tools/Yolo_paul/darknet/data/coco_list.txt')

# labels 目录若不存在，创建labels目录。若存在，则清空目录
if not os.path.exists('%s/%s/labels/' % (dataDir, dataType)):
    os.makedirs('%s/%s/labels/' % (dataDir, dataType))
else:
    shutil.rmtree('%s/%s/labels/' % (dataDir, dataType))
    os.makedirs('%s/%s/labels/' % (dataDir, dataType))

# filelist 目录若不存在，创建filelist目录。
if not os.path.exists('%s/filelist/' % dataDir):
    os.makedirs('%s/filelist/' % dataDir)

coco = COCO(annFile)  # 加载解析标注文件
list_file = open('%s/filelist/%s.txt' % (dataDir, dataType), 'w')  # 数据集的图片list保存路径

imgIds = coco.getImgIds()  # 获取标注文件中所有图片的COCO Img ID
catIds = coco.getCatIds()  # 获取标注文件总所有的物体类别的COCO Cat ID

for imgId in imgIds:
    objCount = 0  # 一个标志位，用来判断该img是否包含我们需要的标注
    print('imgId :%s' % imgId)
    Img = coco.loadImgs(imgId)[0]  # 加载图片信息
    print('Img :%s' % Img)
    filename = Img['file_name']  # 获取图片名
    width = Img['width']  # 获取图片尺寸
    height = Img['height']  # 获取图片尺寸
    print('filename :%s, width :%s ,height :%s' % (filename, width, height))
    annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)  # 获取该图片对应的所有COCO物体类别标注ID
    print('annIds :%s' % annIds)
    for annId in annIds:
        anns = coco.loadAnns(annId)[0]  # 加载标注信息
        catId = anns['category_id']  # 获取该标注对应的物体类别的COCO Cat ID
        cat = coco.loadCats(catId)[0]['name']  # 获取该COCO Cat ID对应的物体种类名
        # print 'anns :%s' % anns
        # print 'catId :%s , cat :%s' % (catId,cat)

        # 如果该类名在我们需要的物体种类列表中，将标注文件转换为YOLO需要的格式
        if cat in classes:
            objCount = objCount + 1
            out_file = open('%s/%s/labels/%s.txt' % (dataDir, dataType, filename[:-4]), 'a')
            cls_id = classes[cat]  # 获取该类物体在yolo训练中的id
            box = anns['bbox']
            size = [width, height]
            bb = convert(size, box)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            out_file.close()

    if objCount > 0:
        list_file.write('%s/%s/JPEGImages/%s\n' % (dataDir, dataType, filename))

list_file.close()
