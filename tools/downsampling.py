# downsampling of the dataset from rootpath to despath with downsample_rate

import glob
import random
import os
import shutil


rootpath = "/media/elab/sdd/data/WildHog/wildhog/"
despath = "/media/elab/sdd/data/WildHog/wildhog_downsampling/"
downsample_rate = 6

# get the files name in the rootpath
myfiles = []
for filename in glob.glob(rootpath + '/*.txt'):
    myfiles.append(filename)

# remove train.txt and test.txt
print(len(myfiles))
myfiles.remove(rootpath + "train.txt")
myfiles.remove(rootpath + "test.txt")
print(len(myfiles))

# downsampling
myfiles_downsampling = []
for i in range(len(myfiles)):
    myfiles_downsampling = random.sample(myfiles, len(myfiles) / downsample_rate)

#print(myfiles_downsampling)
print(len(myfiles_downsampling))
#print(myfiles_downsampling[0])
#exit(0)

# copy the images and txt files to the despath
if os.path.exists(despath):
    shutil.rmtree(despath, ignore_errors=True)

if not os.path.exists(despath):
    os.makedirs(despath)

for i in range(len(myfiles_downsampling)):
    shutil.copyfile(myfiles_downsampling[i], despath + os.path.basename(myfiles_downsampling[i])) #copy txt
    shutil.copyfile(rootpath + os.path.splitext(os.path.basename(myfiles_downsampling[i]))[0] + '.jpg', despath + os.path.splitext(os.path.basename(myfiles_downsampling[i]))[0] + '.jpg') #copy jpg

