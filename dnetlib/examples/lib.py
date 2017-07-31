
import dnetlib as dnet
import threading
import time
net = dnet.loadNetwork("yolo.cfg", "yolo.weights", "coco.names")

def functiont(n):
	im = dnet.loadImage("dog.jpg")
        print(im.w)
	print(im.h)
	print(im.c)
	r = dnet.classify(n,im)
	print dnet.getDict(r)
	dnet.drawDetections(im,r)
	dnet.saveImage(im,"out")


#functiont(net)
#functiont(net)
#functiont(net)

