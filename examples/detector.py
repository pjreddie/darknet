# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn

net = dn.load_net("cfg/tiny-yolo.cfg", "tiny-yolo.weights", 0)
meta = dn.load_meta("cfg/coco.data")
r = dn.detect(net, meta, "data/dog.jpg")
print r

# And then down here you could detect a lot more images like:
r = dn.detect(net, meta, "data/eagle.jpg")
print r
r = dn.detect(net, meta, "data/giraffe.jpg")
print r
r = dn.detect(net, meta, "data/horses.jpg")
print r
r = dn.detect(net, meta, "data/person.jpg")
print r

