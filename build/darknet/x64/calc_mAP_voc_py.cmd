rem C:\Users\Alex\AppData\Local\Programs\Python\Python36\Scripts\pip install numpy
rem C:\Users\Alex\AppData\Local\Programs\Python\Python36\Scripts\pip install cPickle
rem C:\Users\Alex\AppData\Local\Programs\Python\Python36\Scripts\pip install _pickle


rem darknet.exe detector valid cfg/voc.data cfg/yolov2-tiny-voc.cfg yolov2-tiny-voc.weights

darknet.exe detector valid cfg/voc.data cfg/yolov2-voc.cfg yolo-voc.weights


reval_voc_py3.py --year 2007 --classes data\voc.names --image_set test --voc_dir E:\VOC2007_2012\VOCtrainval_11-May-2012\VOCdevkit results




pause
