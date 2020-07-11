#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
#
# This is a utility for converting ground truth data from the kitti format
# to the YOLO format.
#
#
#
# YOLO FORMAT
# .txt for each .jpg - in the same directory and with the same name
# <object-class> <x> <y> <width> <height>
#
# Where:
#
# <object-class> - integer number of object from 0 to (classes-1)
# <x> <y> <width> <height> - floats relative to image width/height 0.0 to 1.0
# eg. <x> = <absolute_x> / <image_width>
# Note: <x> <y> - are center of rectangle (not top-left corner)
#
# For example for img1.jpg you will be created img1.txt containing:
#
#                        1 0.716797 0.395833 0.216406 0.147222
#                        0 0.687109 0.379167 0.255469 0.158333
#                        1 0.420312 0.395833 0.140625 0.166667
#
# KITTI FORMAT
#
# All images as .png in a separate folder to the .txt labels of the same name
# One label line is as follows:
#
#    1 type Describes the type of object: Car, Van, Truck,
#    Pedestrian, Person_sitting, Cyclist, Tram,
#    Misc or DontCare
#    1 truncated Float from 0 (non-truncated) to 1 (truncated), where
#    truncated refers to the object leaving image boundaries
#    1 occluded Integer (0,1,2,3) indicating occlusion state:
#    0 = fully visible, 1 = partly occluded
#    2 = largely occluded, 3 = unknown
#    1 alpha Observation angle of object, ranging [-pi..pi]
#    4 bbox 2D bounding box of object in the image (0-based index):
#    contains left, top, right, bottom pixel coordinates
#    3 dimensions 3D object dimensions: height, width, length (in meters)
#    3 location 3D object location x,y,z in camera coordinates (in meters)
#    1 rotation_y Rotation ry around Y-axis in camera coordinates [-pi..pi]
#    1 score Only for results: Float, indicating confidence in
#    detection, needed for p/r curves, higher is better.
#
# Car 0.0 0 -1.5 57.0 17.3 614.1 200.12 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59
# Cyclist 0.0 0 -2.46 665.45 160.00 717.9 217.9 1.7 0.4 1.6 2.4 1.3 22.1 -2.35
# Pedestrian 0.00 2 0.2 42.1 17.6 433.1 24.0 1.6 0.38 0.30 -5.8 1.6 23.1 -0.03
# DontCare -1 -1 -10 650.19 175.02 668.98 210.48 -1 -1 -1 -1000 -1000 -1000 -10

# core imports
import argparse
import sys
import os
import shutil
import cv2


kitti2yolotype_dict = {'Car': '0',
                       'Van': '0',
                       'Pedestrian': '1',
                       'Person_sitting': '1',
                       'Cyclist': '2',
                       'Truck': '3',
                       'Tram': '6',
                       'Misc': '6',
                       'DontCare': '6'}


def kitti2yolo(kitti_label, img_height, img_width):

    kitti_label_arr = kitti_label.split(' ')
    x1 = float(kitti_label_arr[4])
    y1 = float(kitti_label_arr[5])
    x2 = float(kitti_label_arr[6])
    y2 = float(kitti_label_arr[7])

    bb_width = x2 - x1
    bb_height = y2 - y1
    yolo_x = (x1 + 0.5*bb_width) / img_width
    yolo_y = (y1 + 0.5*bb_height) / img_height
    yolo_bb_width = bb_width / img_width
    yolo_bb_height = bb_height / img_height
    yolo_label = kitti2yolotype_dict[kitti_label_arr[0]]

    return (yolo_label + ' '
            + str(yolo_x) + ' '
            + str(yolo_y) + ' '
            + str(yolo_bb_width) + ' '
            + str(yolo_bb_height))


def main(args):

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti",
                        help="path to kitti-format images and labels, images\
                        should be under images_path/images and labels should\
                        be under images_path/labels")
    parser.add_argument("--yolo",
                        help="path to output yolo-ready training data")
    # kitti paths
    args = parser.parse_args()
    root_path = args.kitti
    yolo_path = args.yolo
    if root_path is None:
        root_path = os.getcwd()
    if (root_path[-1] != os.sep):
        root_path += os.sep
    kitti_images_path = root_path + 'image_2' + os.sep
    kitti_labels_path = root_path + 'label_2' + os.sep

    # yolo paths
    if yolo_path is None:
        yolo_path = root_path + 'yolo_labels' + os.sep

    if not os.path.exists(yolo_path):
        os.makedirs(yolo_path)

    # load each kitti label, convert to yolo and save
    for labelfilename in os.listdir(kitti_labels_path):
        yolo_labels = []
        with open(kitti_labels_path + labelfilename, 'r') as kittilabelfile:
            cvimage = cv2.imread(kitti_images_path
                                 + labelfilename.split('.txt')[0] + '.png')
            height, width, frame_depth = cvimage.shape
            for kitti_label in kittilabelfile:
                yolo_labels.append(kitti2yolo(kitti_label,
                                              img_height=height,
                                              img_width=width))
        with open(yolo_path + labelfilename, 'w+') as yololabelfile:
            for label in yolo_labels:
                yololabelfile.write(label + '\n')


if __name__ == '__main__':
    main(sys.argv)
