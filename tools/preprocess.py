# This is the preprocess of seperate train and test from the data before training.

import glob, os

# Current directory
# current_dir = os.path.dirname(os.path.abspath(__file__))

# Directory where the data will reside, relative to 'darknet.exe'
path_data = '/media/elab/sdd/data/WildHog/hogmodel_Elab/'

# Percentage of images to be used for the test set
percentage_test = 10

# Create and/or truncate train.txt and test.txt
file_train = open(path_data + 'train.txt', 'w')
file_test = open(path_data + 'test.txt', 'w')

# Populate train.txt and test.txt
counter = 1
index_test = round(100 / percentage_test)
for pathAndFilename in glob.iglob(os.path.join(path_data, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    #file = open(title + '.txt', 'w')
    #file.write('0 0.5 0.5 1 1')
    #file.close()

    if counter == index_test:
        counter = 1
        file_test.write(path_data + title + '.jpg' + "\n")
    else:
        file_train.write(path_data + title + '.jpg' + "\n")
        counter = counter + 1
