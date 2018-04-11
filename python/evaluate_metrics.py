import os
import argparse
import numpy as np
import tqdm

import keras
from keras.preprocessing.image import load_img
from keras.applications.mobilenet import DepthwiseConv2D

from test_on_video import parse_predictions, predict_model, absolute_bbox_cords, nms_bboxes, bbox_iou_absolute
from gen_anchors import parse_annotation

argparser = argparse.ArgumentParser()

argparser.add_argument(
    '--dataset', help='path to test subset'
)

argparser.add_argument(
    '--min_w', type=int, default=None, help='minimum bbox width'
)

argparser.add_argument(
    '--min_h', type=int, default=None, help='minimum bbox height'
)

argparser.add_argument(
    '--model', help='path to converted model (Keras h5 file)'
)


def get_minus_sets(A, B):
    aset = set([tuple(x) for x in A])
    bset = set([tuple(x) for x in B])
    return aset - bset


def is_valid_bbox(xmin, ymin, xmax, ymax, min_w, min_h):
    width = xmax - xmin
    height = ymax - ymin

    return width >= min_w and height >= min_h


def main(model_path, dataset_path):
    TP, FP, FN = 0, 0, 0
    ignored_true, ignored_pred = 0, 0

    ann_dir = os.path.join(dataset_path, 'ann')
    img_dir = os.path.join(dataset_path, 'images')

    min_w, min_h = args.min_w, args.min_h
    ignore_small_boxes = min_w is not None and min_h is not None

    model = keras.models.load_model(
        model_path, compile=False, custom_objects={'DepthwiseConv2D': DepthwiseConv2D}
    )

    test_images, seen_labels = parse_annotation(ann_dir, img_dir)

    for img_obj in tqdm.tqdm(test_images):
        image_path = img_obj['filename']
        width, height = img_obj['width'], img_obj['height']

        x = np.array(load_img(image_path), dtype=np.float32)
        _, pred_grid = predict_model(model, x)

        parsed_bboxes = parse_predictions(pred_grid)
        detected_bboxes = nms_bboxes(parsed_bboxes)

        # Step 1: process true and predicted bboxes (convert to image coordinates)
        true_boxes = []
        for real_box in img_obj['object']:
            real_label = real_box['label']
            bbox_coords = real_box['xmin'], real_box['ymin'], real_box['xmax'], real_box['ymax']

            if ignore_small_boxes and not is_valid_bbox(*bbox_coords, min_w, min_h):
                ignored_true += 1
                continue

            true_boxes.append((bbox_coords, real_label))

        pred_boxes = []
        for pred_box in detected_bboxes:
            pred_label = pred_box.get_label()
            bbox_coords = absolute_bbox_cords(pred_box, width, height)

            # We should ignore predicted bboxes too
            if ignore_small_boxes and not is_valid_bbox(*bbox_coords, min_w, min_h):
                ignored_pred += 1
                continue

            pred_boxes.append((bbox_coords, pred_label))

        # Step 2: calculate Precision and Recall
        # Note that predicted bboxes already has only bboxes with confidence >= detect_threshold
        # We only need to check that real IOU is greater or equals predicted IOU
        true_boxes_used, pred_boxes_used = [], []
        for true_box, true_label in true_boxes:
            for pred_box, pred_label in pred_boxes:
                if bbox_iou_absolute(true_box, pred_box) >= 0.3 and true_label == pred_label:
                    true_boxes_used.append(true_box)
                    pred_boxes_used.append(pred_box)

        true_boxes_used = np.array(true_boxes_used)
        pred_boxes_used = np.array(pred_boxes_used)

        if len(true_boxes_used) > 0:
            true_boxes_used = np.unique(true_boxes_used, axis=0)

        if len(pred_boxes_used) > 0:
            pred_boxes_used = np.unique(pred_boxes_used, axis=0)

        # Take only coordinates
        y_true_left = get_minus_sets([bbox_coords for bbox_coords, bbox_class in true_boxes], true_boxes_used)
        y_pred_left = get_minus_sets([bbox_coords for bbox_coords, bbox_class in pred_boxes], pred_boxes_used)

        FN += len(y_true_left)
        FP += len(y_pred_left)
        TP += len(pred_boxes_used)

    AP = TP / (TP + FP + 1e-5)
    Recall = TP / (TP + FN + 1e-5)

    print("AP = %.2f, Recall = %.2f" % (AP, Recall))
    print("Ignored true / predicted = %d / %d" % (ignored_true, ignored_pred))


if __name__ == '__main__':
    args = argparser.parse_args()
    main(args.model, args.dataset)
