# coding=utf-8
# 使用说明
# 需要先安装coco tools
# git clone https://github.com/pdollar/coco.git
# cd coco/PythonAPI
# make install(可能会缺少相关依赖，根据提示安装依赖即可)
# 执行脚本前需在train2014和val2014目录下分别创建JPEGImages和labels目录，并将原来train2014和val2014目录下的图片移到JPEGImages下


from pycocotools.coco import COCO

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = box[0] + box[2]/2.0
    y = box[1] + box[3]/2.0
    w = box[2]
    h = box[3]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def get_classes_and_index(path):
    D = {}
    f = open(path)
    for line in f:
        temp = line.rstrip().split(',', 2)
        print("temp[0]:"+temp[0]+"\n")
        print("temp[1]:" + temp[1]+"\n")
        D[temp[1]] = temp[0]
    return D

dataDir = '/mnt/large4t/pengchong_data/Data/COCO'
dataType = 'train2014'
annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)
classes = get_classes_and_index('/mnt/large4t/pengchong_data/Tools/Yolo_paul/darknet/data/coco_list.txt')

coco = COCO(annFile)
list_file = open('%s/filelist/%s.txt' % (dataDir,dataType), 'w')

imgIds = coco.getImgIds()
catIds = coco.getCatIds()

for imgId in imgIds:
    objCount = 0
    print('imgId :%s'%imgId)
    Img = coco.loadImgs(imgId)[0]
    print('Img :%s' % Img)
    filename = Img['file_name']
    width = Img['width']
    height = Img['height']
    print('filename :%s, width :%s ,height :%s' % (filename,width,height))
    annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)
    print('annIds :%s' % annIds)
    for annId in annIds:
        anns = coco.loadAnns(annId)[0]
        catId = anns['category_id']
        cat = coco.loadCats(catId)[0]['name']
        #print 'anns :%s' % anns
        #print 'catId :%s , cat :%s' % (catId,cat)
        if cat in classes:
            objCount = objCount + 1
            out_file = open('%s/%s/labels/%s.txt' % (dataDir, dataType, filename[:-4]), 'a')
            cls_id = classes[cat]
            box = anns['bbox']
            size = [width,height]
            bb=convert(size, box)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            out_file.close()

    if objCount > 0:
        list_file.write('%s/%s/JPEGImages/%s\n' % (dataDir, dataType, filename))

list_file.close()
