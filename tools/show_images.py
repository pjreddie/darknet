# This is for the show of the images

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import listdir
from os.path import isfile, join
from PIL import Image
import glob
import random
import os

# read the folder names
rootpath = '/media/elab/sdd/data/WildHog/wildhog/'
mypath = []
'''
for dir in os.listdir(rootpath):
    if os.path.isdir(rootpath + dir):
        mypath.append(rootpath + dir)
mypath.sort()
print(mypath)
'''
mypath.append(rootpath)
print(mypath)

# read the images and show
w = 5
h = 5
image_list = []

fig = plt.figure(figsize=(20, 20))
for file in range(0, len(mypath)+1):
    fig.canvas.set_window_title(mypath[file])
    print(mypath[file])
    for filename in glob.glob(mypath[file] + '/*.jpg'):
        image_list.append(filename)
    for i in range(1, w*h+1):
        img = mpimg.imread(image_list[random.randrange(len(image_list))])
        fig.add_subplot(w, h, i)
        plt.imshow(img)
    plt.show(block=False)
    _ = raw_input("Press [enter] to continue.")
