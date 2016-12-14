# import the necessary packages
from __future__ import print_function
from imutils import paths
import argparse
from subprocess import call
from timeit import timeit
def test(): 
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True, help="path to images directory")
    args = vars(ap.parse_args())

    detect_cmd = ["./darknet", "detector", "test", "cfg/coco.data", "cfg/yolo.cfg", "yolo.weights"] 

    #loop over the image paths
    for imagePath in paths.list_images(args["images"]):
        call(detect_cmd + [imagePath])

if __name__=="__main__":
    print(timeit("test()", setup="from __main__ import test", number=1))
