model test:
  ./darknet detect cfg/yolov3.cfg ./model_pretrained/yolov3.weights data/dog.jpg
  ./darknet detect cfg/yolov3.cfg ./model_pretrained/yolov3.weights data/giraffe.jpg
  ./darknet detect cfg/yolov3.cfg ../darknet_official/model_pretrained/yolov3.weights data/giraffe.jpg


train model on voc:
  data preprocess:
    1.download voc data, 2007 and 2012, then copy file 'scripts/voc_label.py' to voc folder
    2.run voc_label.py, and change path of the cfg/voc.data
    *3.change parameters in 'cfg/yolov3.cfg' if need

  train YOLOv3:
    ./darknet detector train cfg/voc.data  cfg/yolov3.cfg ./model_pretrained/yolov3.weights

  test after train:
    ./darknet detect cfg/yolov3.cfg ./backup/yolov3_final.weights data/dog.jpg
