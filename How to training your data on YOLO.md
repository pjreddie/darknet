![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

#Darknet#
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).


----------

#How to training your own data on YOLO#

## **Steps:** ##
1. first, prepare your data, including images and lables. Note the image extension name and label format.  
    (1). The darknet support images: jpeg jpg. I had add some code to support other image format the same as OpenCV. Please see the [darknet pull 13](https://github.com/pjreddie/darknet/pull/13) or my github.  
    (2). The label must be unix format. If you generate labels in Windows, you can use dos2unix tools ion Ubuntu to convert the format.  
    (3). Please note the code in Data.c line 231：  

	`char *labelpath = find_replace(path, "images", "images");`	

	`labelpath = find_replace(labelpath, "JPEGImages", "labels");`	 
	

	**the darknet will replace images to images in trainging data's txt file.**

2. Change the train.txt file's path in tolo.c line 17 18 and line 147：  
	`char *train_images = "/data/voc/train.txt";`	

    `char *backup_directory = "/home/pjreddie/backup/";`	

	`list *plist = get_paths("/home/pjreddie/data/voc/2007_test.txt");`	
	


