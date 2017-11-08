# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb

net = dn.load_net("cfg/yolo-tag.cfg", "yolo-tag_final.weights", 0)
meta = dn.load_meta("cfg/openimages.data")
pdb.set_trace()
rr = dn.detect(net, meta, 'data/dog.jpg')
print rr
pdb.set_trace()

# And then down here you could detect a lot more images like:
rr = dn.detect(net, meta, "data/eagle.jpg")
print rr
rr = dn.detect(net, meta, "data/giraffe.jpg")
print rr
rr = dn.detect(net, meta, "data/horses.jpg")
print rr
rr = dn.detect(net, meta, "data/person.jpg")
print rr

