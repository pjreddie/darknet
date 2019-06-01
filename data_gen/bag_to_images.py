#!/usr/bin/python
import rosbag

import sys
import argparse

import os
import shutil

import progressbar

import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

def zip_images(dir_name, output_filename):
    shutil.make_archive(output_filename, 'zip', dir_name)

def decompress_images(bag_filename):

    bridge = CvBridge()
    bag = rosbag.Bag(bag_filename)

    if not os.path.exists("images"):
        os.mkdir("images")

    start_time = bag.get_start_time()
    end_time = bag.get_end_time()
    run_time = end_time - start_time

    print("Bag is %.2f seconds" % run_time)

    type_topic_info = bag.get_type_and_topic_info()
    topics = type_topic_info.topics
    print("Bag contains topics: ")

    for topic in topics.keys():
        print("\t%s %s %d" % (topic, topics[topic][0], topics[topic][1]))

    toolbar_width = 70
    bar = progressbar.ProgressBar(maxval=toolbar_width,
                                  widgets=[progressbar.Bar('#', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()

    num_images = 0
    for topic, msg, t in bag.read_messages():

        bar.update((t.to_sec() - start_time) / run_time * toolbar_width)

        if msg._type == Image._type:
            try:
                img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                cv2.imwrite("images/%s_%s.jpg" % (topic.split("/")[-1], str(t))
                            , img)
                num_images += 1

            except CvBridgeError as e:
                print(e)

    bag.close()

    print("")  # move down a line from the progress bar
    print("Extracted %s images. Creating zip ..." % num_images)
    zip_images("images", "zipped_images/%s" % bag_filename.split(".")[0])
    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', default=None)

    args = parser.parse_args(sys.argv[1:])

    if args.bag is None:
        print("Bag file is required!")
        return

    decompress_images(args.bag)


if __name__ == "__main__":
    main()
