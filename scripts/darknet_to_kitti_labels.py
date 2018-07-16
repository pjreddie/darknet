#!/usr/bin/python

import os
import sys
import numpy as np
from argparse import ArgumentParser

from os import listdir
from os.path import isfile, join


# width = 960
# height = 540

label_tuples = (
    'DontCare', 'Car', 'SUV', 'SmallTruck', 'MediumTruck', 'LargeTruck', 'Pedestrian', 'Bus', 'Van', 'GroupOfPeople',
    'Bicycle', 'Motorcycle', 'TrafficSignal-Green', 'TrafficSignal-Yellow', 'TrafficSignal-Red')

vehicles = ('Car', 'SUV', 'SmallTruck', 'MediumTruck', 'Van')


# labels = {
#     'DontCare': 0,
#     'Car': 1,
#     'SUV': 2,
#     'SmallTruck': 3,
#     'MediumTruck': 4,
#     'LargeTruck': 5,
#     'Pedestrian': 6,
#     'Bus': 7,
#     'Van': 8,
#     'GroupOfPeople': 9,
#     'Bicycle': 10,
#     'Motorcycle': 11,
#     'TrafficSignal-Green': 12,
#     'TrafficSignal-Yellow': 13,
#     'TrafficSignal-Red': 14,
#     # 'Crossing' : 15,  # ignore Crossing annotations
# }


def readAnnotations(lbf):
    """ Read annotations for a given image """
    # lbf = "../labels/" + f[: f.rfind('.')] + ".txt"

    b = []
    with open(lbf, "r") as fh:
        for l in fh:
            p = l.strip().split()
            b.append((p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4])))

    # print b
    return b


def writeKitty(b, of):
    """
    Transform annoatations to KITTI format.
    KITTY format info: https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md
    Bounding box encoded as minx miny maxx maxy
    """
    with open(of, "w") as fh:
        for r in b:
            fh.write("%s 0 0 0 %d %d %d %d 0 0 0 0 0 0 0\n" % (
                r[0], int(r[1] * dw), int(r[2] * dh), int(r[3] * dw), int(r[4] * dh)))


def writeKittyFromDarkNet(b, of):
    """
    Transform annoatations to KITTI format.
    KITTY format info: https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md
    Bounding box encoded as minx miny maxx maxy
    """

    # fh.write("%d %f %f %f %f\n" % (labels[r[0]], cx * dw, cy * dh, w * dw, h * dh))
    # cx = (r[1] + r[3]) * 0.5 * iw
    # cy = (r[2] + r[4]) * 0.5 * ih
    # w = (r[3] - r[1]) * iw
    # h = (r[4] - r[2]) * ih

    with open(of, "w") as fh:
        for r in b:
            left = r[1] * 1 / iw - r[3] * 1 / iw * 0.5
            top = r[2] * 1 / ih - r[4] * 1 / ih * 0.5
            right = r[1] * 1 / iw + r[3] * 1 / iw * 0.5
            bottom = r[2] * 1 / ih + r[4] * 1 / ih * 0.5

            text_label = label_tuples[int(r[0])]
            fh.write(
                "%s 0 0 0 %d %d %d %d 0 0 0 0 0 0 0\n" % (text_label, int(left), int(top), int(right), int(bottom)))


def writeKittyFromDarkNet2(b, of):
    with open(of, "w") as fh:
        for r in b:
            left = r[1] * 1 / iw - r[3] * 1 / iw * 0.5
            top = r[2] * 1 / ih - r[4] * 1 / ih * 0.5
            right = r[1] * 1 / iw + r[3] * 1 / iw * 0.5
            bottom = r[2] * 1 / ih + r[4] * 1 / ih * 0.5


            text_label = label_tuples[int(r[0])]

            if text_label in vehicles:
                text_label = 'Car'
                fh.write("%s 0 0 0 %.2f %.2f %.2f %.2f 0 0 0 0 0 0 0\n" % (text_label, left, top, right, bottom))
            else:
                text_label = 'DontCare'
                fh.write("%s -1 -1 -10 %.2f %.2f %.2f %.2f -1 -1 -1 -1000 -1000 -1000 -10\n" % (text_label, left, top, right, bottom))


def create_subset(size, input_images, input_label_dir, output_label_dir, output_image_dir):

    label_list = list()
    images = [f for f in listdir(input_images) if isfile(join(input_images, f))]

    np.random.shuffle(images)

    images = images[:size] # take first N

    ensureDir(output_label_dir)
    ensureDir(output_image_dir)

    for img in images:
        assoc_label = os.path.splitext(img)[0] + '.txt'
        jpg_ext = os.path.splitext(img)[0] + '.jpg'

        label_list.append(assoc_label)

        in_img_path = input_images + '/' + img
        out_img_path = output_image_dir + '/' + jpg_ext

        os.link(in_img_path, out_img_path)


        i=0
    for label in label_list:
        assoc_label_path = input_label_dir + '/' + label
        out_label_path = out_label_dir + '/' + label

        annos = readAnnotations(assoc_label_path)

        writeKittyFromDarkNet2(annos, out_label_path)
        print('%d done with %s' % (i, out_label_path))
        i += 1


def ensureDir(d):
    if not d[-1] == '/':
        d += '/'
    if not os.path.exists(d):
        os.makedirs(d)


def usage():
    print "Usage: python2 create_dataset.py -d <dataset_path> [-f <format> -h <image_height> -w <image_width>] <output_dir>"
    exit()


def get_args():
    parser = ArgumentParser(add_help=False)

    parser.add_argument("dimensions", nargs=2, help="dimensions")
    parser.add_argument('--help', action='help', help='Show this help message and exit')


    parser.add_argument('-ii', '--in_image_dir', type=str, default=None, help="in images directory path")
    parser.add_argument('-il', '--in_label_dir', type=str, default=None, help="in labels directory path")
    parser.add_argument('-oi', '--out_image_dir', type=str, default=None, help="out images directory path")
    parser.add_argument('-ol', '--out_label_dir', type=str, default=None, help="out labels directory path")

    parser.add_argument('-if', '--in_format', type=str, default="darknet", help="labels format")
    parser.add_argument('-of', '--out_format', type=str, default="kitti", help="output label format")

    parser.add_argument('-s', '--subset', type=int, default=None, help="output label format")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if not args.dimensions:
        print "Please specify dimensions!"
        usage()
    height = int(args.dimensions[0])
    width = int(args.dimensions[1])

    if not args.in_label_dir:
        print "Need input label dir"
        usage()
    in_label_dir = args.in_label_dir

    if not args.out_label_dir:
        print "Need output label dir"
        usage()
    out_label_dir = args.out_label_dir

    print(in_label_dir)
    print(out_label_dir)

    dw = 1
    dh = 1
    iw = 1. / width
    ih = 1. / height


    if args.subset:
        size = int(args.subset)
        if not args.in_image_dir:
            print "Need input img dir"
            usage()
        in_image_dir = args.in_image_dir

        if not args.out_image_dir:
            print "Need out_image_dir"
            usage()
        out_image_dir = args.out_image_dir

        create_subset(size,in_image_dir,in_label_dir,out_label_dir,out_image_dir)


    if out_label_dir[0] != '/':
        out_label_dir = os.path.join(os.getcwd(), out_label_dir)
    ensureDir(out_label_dir)

    label_files = [f for f in listdir(in_label_dir) if isfile(join(in_label_dir, f))]

    convert = False
    if (convert):
        i = 0
        os.chdir(in_label_dir)
        if args.inputformat == 'darknet':
            for label in label_files:
                outfile_path = os.path.join(os.getcwd(), out_label_dir, label)
                # print outfile_path
                annos = readAnnotations(label)
                writeKittyFromDarkNet2(annos, outfile_path)
                print('%d done with %s' % (i, label))
                i += 1
        elif args.inputformat == 'original':
            for label in label_files:
                outfile_path = os.path.join(os.getcwd(), out_label_dir, label)
                # print outfile_path
                annos = readAnnotations(label)
                writeKitty(annos, outfile_path)
                print('%d done with %s' % (i, label))
                i += 1

