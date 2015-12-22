![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

#Darknet#
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).
------------------------------------------------------------------------------------------------------------------

#About This Fork#
1. This fork repository adds some additional niche in addition to the current darkenet from pjreddie. e.g.
   (1). Read a video file, process it, and output a video with boundingboxes.
   (2). Some util functions like image_to_Ipl, converting the image from darknet back to Ipl image format from OpenCV(C).
   (3). Adds some python scripts to label our own data, and preprocess annotations to the required format by darknet.  
   ...More to be added
2. This fork repository illustrates how to train a customized neural network with our own data, with our own classes.
   The procedure is documented in README.md.


#How to Train With Customized Data and Class Numbers/Labels#

