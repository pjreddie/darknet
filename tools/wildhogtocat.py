#this file used to replace the txt of wildhog folder's 0 to 15 - cat, in case of training a model to recognize 0-person as wildhog
mypath = "/media/elab/sdd/data/WildHog/wildhogtext"

from os import listdir
from os.path import isfile, join

files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

print(len(files))

for file in files:
    content = []
    with open(mypath + '/' + file) as f:
        content = f.readlines()
        for i in range(len(content)):
            content[i] = content[i].replace('0','15', 1)
        #print(content)

    with open(mypath + '/' + file, 'w+') as f:
        for i in range(len(content)):
            f.write(str(content[i]))
        f.close()
