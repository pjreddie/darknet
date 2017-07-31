
import dnetlib as dnet
import threading
import time
import base64

net = dnet.loadNetwork("/var/trains/generico/net.cfg", "/var/trains/generico/net.weights", "/var/trains/generico/net.names")

def functiont(n):
	with open("dog.jpg", "rb") as image_file:
	    encoded_string = base64.b64encode(image_file.read())
	im = dnet.loadImageBase64(encoded_string)
	r = dnet.classify(n,im)
	print dnet.getDict(r)
	dnet.drawDetections(im,r)
	#print dnet.getImageBase64(im)


functiont(net)
#functiont(net)
#functiont(net)

