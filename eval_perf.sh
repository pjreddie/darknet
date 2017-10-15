#!/bin/bash
tool=./extra/tools/eval_accuracy.py
#det_files=./results/f32-results/*.txt
#det_files=./results/16b-results/*.txt
det_files=./results/8b-results/*.txt
python $tool\
	--det $det_files\
	--ann_dir /opt/voc/VOCdevkit/VOC2007/Annotations/\
	--imlist /opt/voc/2007_test.txt
