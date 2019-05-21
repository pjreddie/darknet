import glob
import os

pathDarknet    = os.path.dirname(os.path.abspath(__file__))
pathCrowdHuman = os.path.join(pathDarknet, 'crowdhuman_train')

fileTrain = open('train.txt', 'w')
count = 0
for pathAndFilename in glob.iglob(os.path.join(pathCrowdHuman, '*.jpg')):
    
    fileTrain.write(pathAndFilename + '\n')

fileTrain.close()
    