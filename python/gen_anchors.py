import os
import random
import argparse
import numpy as np

import xml.etree.ElementTree as ET

argparser = argparse.ArgumentParser()

argparser.add_argument(
    '--dataset',
    help='path to train subset')

argparser.add_argument(
    '--anchors',
    default=10,
    help='number of anchors to use')

argparser.add_argument(
    '--grid_w',
    default=30,
    help='output grid width')

argparser.add_argument(
    '--grid_h',
    default=17,
    help='output grid height')


def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape

    return np.array(similarities)


def avg_IOU(anns, centroids):
    n, d = anns.shape
    sum_value = 0.

    for i in range(anns.shape[0]):
        sum_value += max(IOU(anns[i], centroids))

    return sum_value / n


def print_anchors(centroids):
    res_str = ''

    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    for i in sorted_indices[:-1]:
        res_str += '%0.3f,%0.3f, ' % (anchors[i,0], anchors[i,1])

    # There should not be comma after last anchor, that's why
    res_str += '%0.3f,%0.3f' % (anchors[sorted_indices[-1:], 0], anchors[sorted_indices[-1:], 1])

    print(res_str)


def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    prev_assignments = np.ones(ann_num)*(-1)
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for _ in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    iteration = 0
    while True:
        distances = []
        iteration += 1

        for i in range(ann_num):
            d = 1.0 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances)

        # distances.shape = (ann_num, anchor_num)
        print("iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances-distances))))

        # assign samples to centroids
        assignments = np.argmin(distances, axis=1)

        if (assignments == prev_assignments).all():
            return centroids

        # calculate new centroids
        centroid_sums = np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]] += ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments == j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()


def parse_annotation(ann_dir, img_dir, labels=()):
    all_imgs = []
    seen_labels = {}
    
    for ann in sorted(os.listdir(ann_dir)):
        annotation_file = os.path.join(ann_dir, ann)

        try:
            img = {'object': []}

            tree = ET.parse(annotation_file)

            for elem in tree.iter():
                if 'filename' in elem.tag:
                    img['filename'] = img_dir + elem.text
                if 'width' in elem.tag:
                    img['width'] = int(elem.text)
                if 'height' in elem.tag:
                    img['height'] = int(elem.text)
                if 'object' in elem.tag or 'part' in elem.tag:
                    obj = {}

                    for attr in list(elem):
                        if 'name' in attr.tag:
                            obj['name'] = attr.text

                            if obj['name'] in seen_labels:
                                seen_labels[obj['name']] += 1
                            else:
                                seen_labels[obj['name']] = 1

                            if len(labels) > 0 and obj['name'] not in labels:
                                break
                            else:
                                img['object'] += [obj]

                        if 'bndbox' in attr.tag:
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    obj['xmin'] = int(round(float(dim.text)))
                                if 'ymin' in dim.tag:
                                    obj['ymin'] = int(round(float(dim.text)))
                                if 'xmax' in dim.tag:
                                    obj['xmax'] = int(round(float(dim.text)))
                                if 'ymax' in dim.tag:
                                    obj['ymax'] = int(round(float(dim.text)))

            if len(img['object']) > 0:
                all_imgs += [img]
        except Exception:
            print('Cannot parse file: {}'.format(annotation_file))
                        
    return all_imgs, seen_labels


def main(args):
    ann_dir = os.path.join(args.dataset, 'ann')
    img_dir = os.path.join(args.dataset, 'images')
    num_anchors = args.anchors

    train_imgs, train_labels = parse_annotation(ann_dir, img_dir, labels=[])

    # run k_mean to find the anchors
    annotation_dims = []
    for image in train_imgs:
        cell_w = float(image['width']) / args.grid_w
        cell_h = float(image['height']) / args.grid_h

        for obj in image['object']:
            relative_w = (float(obj['xmax']) - float(obj['xmin'])) / cell_w
            relatice_h = (float(obj["ymax"]) - float(obj['ymin'])) / cell_h
            annotation_dims.append((relative_w, relatice_h))

    annotation_dims = np.array(annotation_dims)

    centroids = run_kmeans(annotation_dims, num_anchors)

    print('Average IOU for', num_anchors, 'anchors:', '%0.3f' % avg_IOU(annotation_dims, centroids))
    print_anchors(centroids)


if __name__ == '__main__':
    argv = argparser.parse_args()
    main(argv)
