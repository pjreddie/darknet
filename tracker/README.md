**Why most of the trackers are written by matlab? I hate that! Cpp is fast and clear! I even doubt that the FPS measured by using matlab is not really meaningful, especially for actual and embedded system use! So I will re-implement those trackers by cpp day by day, hope you like it!**

# Run Opencv trackers
Change the path of your test images in **kcf/opencvtrackers.cpp**.
```
cd opencvtrackers
make 
./opencvtrackers.bin
```

# Run KCF
Change the path of your test images in **kcf/runkcftracker.cpp**.
```
cd kcf
make -j12
./runkcftracker.bin
```

# Run GOTURN
Change the path of your test images in **goturn/rungoturntracker.cpp**.

Need to install [[caffe](https://github.com/rockkingjy/caffe)], and change the goturn/makefile according to your installation.
### Pretrained model
You can download a pretrained [[tracker model (434 MB)](https://drive.google.com/file/d/1uc9k8sTqug_EY9kv1v_QnrDxjkrTJejR/view?usp=sharing)], put it into folder: **goturn/nets**

```
cd goturn
make -j12
./rungoturntracker.bin
```

### Run caffe classification for simple test
```
./classification.bin   /media/elab/sdd/caffe/models/bvlc_reference_caffenet/deploy.prototxt   /media/elab/sdd/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel   /media/elab/sdd/caffe/data/ilsvrc12/imagenet_mean.binaryproto   /media/elab/sdd/caffe/data/ilsvrc12/synset_words.txt   /media/elab/sdd/caffe/examples/images/cat.jpg
```

# Run to compare all the trackers
```
make all
./trackerscompare.bin
```

--------------------------------
## KCF Tracker

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,   
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,   
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

Authors: Joao Faro, Christian Bailer, Joao F. Henriques   
Contacts: joaopfaro@gmail.com, Christian.Bailer@dfki.de, henriques@isr.uc.pt   
Institute of Systems and Robotics - University of Coimbra / Department of Augmented Vision DFKI   

## GOTURN Tracker

**[Learning to Track at 100 FPS with Deep Regression Networks](http://davheld.github.io/GOTURN/GOTURN.html)**,
<br>
[David Held](http://davheld.github.io/),
[Sebastian Thrun](http://robots.stanford.edu/),
[Silvio Savarese](http://cvgl.stanford.edu/silvio/),
<br>
European Conference on Computer Vision (ECCV), 2016 (In press)
