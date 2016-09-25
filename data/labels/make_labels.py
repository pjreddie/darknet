import os
import string
import pipes

#l = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

l = string.printable

for word in l:
    #os.system("convert -fill black -background white -bordercolor white -border 4 -font futura-normal -pointsize 18 label:\"%s\" \"%s.png\""%(word, word))
    if word == ' ':
        os.system('convert -fill black -background white -bordercolor white -font futura-normal -pointsize 64 label:"\ " 32.png')
    elif word == '\\':
        os.system('convert -fill black -background white -bordercolor white -font futura-normal -pointsize 64 label:"\\\\\\\\" 92.png')
    elif ord(word) in [9,10,11,12,13,14]:
        pass
    else:
        os.system("convert -fill black -background white -bordercolor white -font futura-normal -pointsize 64 label:%s \"%d.png\""%(pipes.quote(word), ord(word)))

