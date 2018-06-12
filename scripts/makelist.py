#!/usr/bin/env python
import os
train = "./coco/train/"
valid = "./coco/valid/"

traintxt = []
validtxt = []

for image_name in os.listdir(os.path.join(train, "images")):
	image_path = os.path.join(train, "images", image_name)
	label_name = image_name[:-4] + ".txt"
	label_path = os.path.join(train, "labels", label_name)
	if os.path.exists(label_path):
		traintxt.append(image_path)

for image_name in os.listdir(os.path.join(valid, "images")):
        image_path = os.path.join(valid, "images", image_name)
        label_name = image_name[:-4] + ".txt"
        label_path = os.path.join(valid, "labels", label_name)
        if os.path.exists(label_path):
                validtxt.append(image_path)

with open("./cfg/train.txt", "w") as f:
	f.write("\n".join(traintxt))
with open("./cfg/valid.txt", "w") as f:
	f.write("\n".join(validtxt))


