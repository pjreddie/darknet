![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

#Darknet#
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

#About This Fork#
1. This fork repository adds some additional niche in addition to the current darkenet from pjreddie. e.g.

   (1). Read a video file, process it, and output a video with boundingboxes.
   
   (2). Some util functions like image_to_Ipl, converting the image from darknet back to Ipl image format from OpenCV(C).
   
   (3). Adds some python scripts to label our own data, and preprocess annotations to the required format by darknet.  
   
   ...More to be added

2. This fork repository illustrates how to train a customized neural network with our own data, with our own classes.

   The procedure is documented in README.md.

#How to Train With Customized Data and Class Numbers/Labels#

1. Collect Data and Annotation
   
   (1). For Videos, we can use video summary, shot boundary detection or camera take detection, to create static images.
   
   (2). For Images, we can use [BBox-Label-Tool](https://github.com/puzzledqs/BBox-Label-Tool) to label objects.

2. Create Annotation in Darknet Format 
   
   (1). If we choose to use VOC data to train, use [scripts/voc_label.py](https://github.com/Guanghan/darknet/blob/master/scripts/voc_label.py) to convert existing VOC annotations to darknet format.
   
   (2). If we choose to use our own collected data, use [scripts/] to convert the annotations.

   At this step, we should have darknet annotations(.txt) and a training list(.txt).
   
3. Modify Some Code

   (1)
   
   (2)
   
   (3)
