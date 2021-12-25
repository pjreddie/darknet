import cv2
import sys
import time
from tiler import *
from matplotlib import pyplot as plt

def sliding_box(boxes):
    #plt.ion()
    for i in range(len(boxes)):
        im = cv2.cvtColor(boxes[i]['img'], cv2.COLOR_BGR2RGB)
        if im.shape[0]==416 and im.shape[1]==416:
            #print(im.shape)
            #print(boxes[i]['pos'])
            plt.figure(figsize=(5,5))
            plt.imshow(im)
            plt.title('res: '+str(im.shape)+'\n'+'pos: '+str(boxes[i]['pos']))
            plt.xticks([])
            plt.yticks([])
            plt.show()
            #_ = input("Press [enter] to continue.")


if __name__ == "__main__":
    img = read_image(sys.argv[1])
    boxes = image_boxes(img, [416,416])
    sliding_box(boxes)
