# coding=utf-8
# 使用说明
# 需要先安装coco tools
# git clone https://github.com/pdollar/coco.git
# cd coco/PythonAPI
# make install


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


dataDir = '/mnt/large4t/pengchong_data/Data/COCO'
dataType = 'train2014'
annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'fork', 'knife', 'spoon', 'bowl', 'banana',
           'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']

coco = COCO(annFile)
list_file = open('%s/filelist/%s.txt' % (dataDir,dataType), 'w')

imgIds = coco.getImgIds()
catIds = coco.getCatIds()

for imgId in imgIds:
    objCount = 0
    print 'imgId :%s'%imgId
    Img = coco.loadImgs(imgId)[0]
    print 'Img :%s' % Img
    filename = Img['file_name']
    width = Img['width']
    height = Img['height']
    print 'filename :%s, width :%s ,height :%s' % (filename,width,height)
    annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)
    print 'annIds :%s' % annIds
    for annId in annIds:
        anns = coco.loadAnns(annId)[0]
        catId = anns['category_id']
        cat = coco.loadCats(catId)[0]['name']
        #print 'anns :%s' % anns
        #print 'catId :%s , cat :%s' % (catId,cat)
        if cat in classes:
            objCount = objCount + 1
            out_file = open('%s/%s/test/%s.txt' % (dataDir, dataType, filename[:-4]), 'a')
            cls_id = classes.index(cat)
            box = anns['bbox']
            size = [width,height]
            bb=convert(size, box)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            out_file.close()
        #break

    if objCount > 0:
        list_file.write('%s/%s/JPEGImages/%s\n' % (dataDir, dataType, filename))
    #break

list_file.close()
