# Low Precision Extension to Darknet

This extension to the Darknet framework aims at providing low precision inference for Darknet models. Currently the underlaying arithmetic is still in floating point. However, the model file can be stored using 8b or 16b format which helps to reduce the model size by a factor of 4 or 2 respectively. Initial goal is to save some space on SD card when using on embedded platforms and mobile devices.



# TODO

1. On the fly weight conversion from 8b to float during the floating point inference so that RAM usage can be reduced.
2. Memory and runtime measurement hooks.
3. Analysis of effect of quantization on network performance such as mAP, classification error.
4. Computations using 8b/16bit.