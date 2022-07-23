![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

# Darknet #
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

**Discord** invite link for for communication and questions: https://discord.gg/zSq8rtW

## YOLOv7: 

* **paper** - YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors: https://arxiv.org/abs/2207.02696

* **source code - Pytorch (use to reproduce results):** https://github.com/WongKinYiu/yolov7

----

Official YOLOv7 is more accurate and faster than YOLOv5 by **120%** FPS, than YOLOX by **180%** FPS, than Dual-Swin-T by **1200%** FPS, than ConvNext by **550%** FPS, than SWIN-L by **500%** FPS.

YOLOv7 surpasses all known object detectors in both speed and accuracy in the range from 5 FPS to 160 FPS and has the highest accuracy 56.8% AP among all known real-time object detectors with 30 FPS or higher on GPU V100, batch=1.

* YOLOv7-e6 (55.9% AP, 56 FPS V100 b=1) by `+500%` FPS faster than SWIN-L Cascade-Mask R-CNN (53.9% AP, 9.2 FPS A100 b=1)
* YOLOv7-e6 (55.9% AP, 56 FPS V100 b=1) by `+550%` FPS faster than ConvNeXt-XL C-M-RCNN (55.2% AP, 8.6 FPS A100 b=1)
* YOLOv7-w6 (54.6% AP, 84 FPS V100 b=1) by `+120%` FPS faster than YOLOv5-X6-r6.1 (55.0% AP, 38 FPS V100 b=1)
* YOLOv7-w6 (54.6% AP, 84 FPS V100 b=1) by `+1200%` FPS faster than Dual-Swin-T C-M-RCNN (53.6% AP, 6.5 FPS V100 b=1)
* YOLOv7x (52.9% AP, 114 FPS V100 b=1) by `+150%` FPS faster than PPYOLOE-X (51.9% AP, 45 FPS V100 b=1)
* YOLOv7 (51.2% AP, 161 FPS V100 b=1) by `+180%` FPS faster than YOLOX-X (51.1% AP, 58 FPS V100 b=1)

----

![more5](https://user-images.githubusercontent.com/4096485/179425274-f55a36d4-8450-4471-816b-8c105841effd.jpg)

----

![image](https://user-images.githubusercontent.com/4096485/177675030-a929ee00-0eba-4d93-95c2-225231d0fd61.png)


----

![yolov7_640_1280](https://user-images.githubusercontent.com/4096485/177688869-d75e0c36-63af-46ec-bdbd-81dbb281f257.png)

----

## Scaled-YOLOv4: 

* **paper (CVPR 2021)**: https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Scaled-YOLOv4_Scaling_Cross_Stage_Partial_Network_CVPR_2021_paper.html

* **source code - Pytorch (use to reproduce results):** https://github.com/WongKinYiu/ScaledYOLOv4

* **source code - Darknet:** https://github.com/AlexeyAB/darknet

* **Medium:** https://alexeyab84.medium.com/scaled-yolo-v4-is-the-best-neural-network-for-object-detection-on-ms-coco-dataset-39dfa22fa982?source=friends_link&sk=c8553bfed861b1a7932f739d26f487c8

## YOLOv4:

* **paper:** https://arxiv.org/abs/2004.10934

* **source code:** https://github.com/AlexeyAB/darknet

* **Wiki:** https://github.com/AlexeyAB/darknet/wiki

* **useful links:** https://medium.com/@alexeyab84/yolov4-the-most-accurate-real-time-neural-network-on-ms-coco-dataset-73adfd3602fe?source=friends_link&sk=6039748846bbcf1d960c3061542591d7

For more information see the [Darknet project website](http://pjreddie.com/darknet).


<details><summary> <b>Expand</b> </summary>

![yolo_progress](https://user-images.githubusercontent.com/4096485/146988929-1ed0cbec-1e01-4ad0-b42c-808dcef32994.png) https://paperswithcode.com/sota/object-detection-on-coco

----

![scaled_yolov4](https://user-images.githubusercontent.com/4096485/112776361-281d8380-9048-11eb-8083-8728b12dcd55.png) AP50:95 - FPS (Tesla V100) Paper: https://arxiv.org/abs/2011.08036

----

![YOLOv4Tiny](https://user-images.githubusercontent.com/4096485/101363015-e5c21200-38b1-11eb-986f-b3e516e05977.png)

----

![YOLOv4](https://user-images.githubusercontent.com/4096485/90338826-06114c80-dff5-11ea-9ba2-8eb63a7409b3.png)

</details>

----

![OpenCV_TRT](https://user-images.githubusercontent.com/4096485/90338805-e5e18d80-dff4-11ea-8a68-5710956256ff.png)


## Citation


```
@misc{https://doi.org/10.48550/arxiv.2207.02696,
  doi = {10.48550/ARXIV.2207.02696},
  url = {https://arxiv.org/abs/2207.02696},
  author = {Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  publisher = {arXiv},
  year = {2022}, 
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

```
@misc{bochkovskiy2020yolov4,
      title={YOLOv4: Optimal Speed and Accuracy of Object Detection}, 
      author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
      year={2020},
      eprint={2004.10934},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```
@InProceedings{Wang_2021_CVPR,
    author    = {Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
    title     = {{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {13029-13038}
}
```
