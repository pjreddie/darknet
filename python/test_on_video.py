#! /usr/bin/env python

import math
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import skvideo.io

import keras
from keras.applications.mobilenet import DepthwiseConv2D


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-m',
    '--model',
    help='path to converted model (Keras h5 file)')

argparser.add_argument(
    '-i',
    '--input',
    help='path to an image or an video (mp4 format)')


img_w, img_h = 512, 960
grid_w, grid_h = 16, 30
num_anchors = 10
num_classes = 3
detect_threshold = 0.3
nms_threshold = 0.1

classes = ['complementary_signs', 'white_signs', 'cars']

biases = np.array([
    [0.298, 0.256], [0.602, 0.566], [0.932, 0.772], [1.225, 1.190], [1.858, 1.681],
    [1.895, 0.854], [2.713, 2.490], [3.381, 1.460], [4.488, 3.066], [7.450, 5.975]
])


class BBox(object):

    def __init__(self, row, col, n, y, x, h, w, obj, classes):
        self.row = row
        self.col = col
        self.n = n
        self.y = y
        self.x = x
        self.h = h
        self.w = w
        self.obj = obj
        self.classes = classes

    def absolute_values(self, img_h, img_w):
        left = int(round((self.x - self.w / 2) * img_w))
        top = int(round((self.y - self.h / 2) * img_h))
        height = int(round(self.h * img_h))
        width = int(round(self.w * img_w))

        return left, top, height, width

    def get_score(self):
        return self.classes[self.get_label()]

    def get_label(self):
        return np.argmax(self.classes)


def bbox_iou(box1, box2):
    x1_min = box1.x - box1.w/2
    x1_max = box1.x + box1.w/2
    y1_min = box1.y - box1.h/2
    y1_max = box1.y + box1.h/2

    x2_min = box2.x - box2.w/2
    x2_max = box2.x + box2.w/2
    y2_min = box2.y - box2.h/2
    y2_max = box2.y + box2.h/2

    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])

    intersect = intersect_w * intersect_h

    union = box1.w * box1.h + box2.w * box2.h - intersect

    return float(intersect) / union


def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def predict_model(model, frame):

    image = cv2.resize(frame, dsize=(img_h, img_w), interpolation=cv2.INTER_LINEAR)
    img = np.expand_dims(np.array(image, dtype=np.float32), axis=0) / 255.

    predicted_tensor = model.predict(img)[0]
    output_grid = predicted_tensor.reshape(grid_w, grid_h, num_anchors, 4 + 1 + num_classes)

    return image, output_grid


def parse_predictions(model_output):
    bboxes = []
    for row in range(grid_w):
        for col in range(grid_h):
            for n in range(num_anchors):
                anchor = biases[n]
                vals = model_output[row, col, n, :]

                ty, tx = vals[0], vals[1]
                th, tw = vals[2], vals[3]
                to, logits = vals[4], vals[5:]

                # by = (sigmoid(ty) + col + 1) / grid_h
                # bx = (sigmoid(tx) + row + 1) / grid_w

                by = (sigmoid(ty) + col) / grid_h
                bx = (sigmoid(tx) + row) / grid_w

                bw = math.exp(tw) * anchor[1] / grid_w
                bh = math.exp(th) * anchor[0] / grid_h

                scale = sigmoid(to)
                classses_proba = scale * softmax(logits)

                bbox = BBox(row, col, n, by, bx, bh, bw, scale, classses_proba)
                bboxes.append(bbox)

    return bboxes


def nms_bboxes(boxes):
    for c in range(num_classes):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] < detect_threshold:
                boxes[index_i].classes[c] = 0
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > detect_threshold]

    return boxes


def draw_boxes(image, boxes, labels):
    for box in boxes:
        y, x, w, h = box.absolute_values(img_h, img_w)

        xmin, ymin = x, y
        xmax, ymax = x + w, y + h

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        cv2.putText(image,
                    labels[box.get_label()] + ' ' + str(box.get_score()),
                    (xmin, ymin - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * image.shape[0],
                    (0, 255, 0), 2)


def main(model_path, input_path):
    model = keras.models.load_model(model_path, compile=False, custom_objects={'DepthwiseConv2D': DepthwiseConv2D})

    if input_path[-4:] == '.mp4':
        video_out = input_path[:-4] + '_detected' + input_path[-4:]
        video_reader = skvideo.io.FFmpegReader(input_path)
        video_writer = skvideo.io.FFmpegWriter(video_out)

        for frame in tqdm(video_reader.nextFrame()):
            frame, model_output = predict_model(model, frame)

            parsed_bboxes = parse_predictions(model_output)
            detected_bboxes = nms_bboxes(parsed_bboxes)

            draw_boxes(frame, detected_bboxes, classes)

            video_writer.writeFrame(np.uint8(frame))

        video_reader.close()
        video_writer.close()


if __name__ == '__main__':
    args = argparser.parse_args()
    main(args.model, args.input)
