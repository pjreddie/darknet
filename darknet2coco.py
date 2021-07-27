
import numpy as np
import darknet
from darknet_images import (image_detection, load_images)
import cv2
import os
import imagesize
import json
from argparse import ArgumentParser


def main():

    def parse_args():
        parser = ArgumentParser()
        parser.add_argument("--config-file", default="./cfg/yolov4.cfg", help="path to config file")
        parser.add_argument("--data-file", default="./cfg/coco.data", help="path to data file")
        parser.add_argument('--weights', type=str, default='', help='Weights')
        parser.add_argument('--img-root', type=str, default='', help='Image root')
        parser.add_argument('--json-path', type=str, default='', help='json path')
        parser.add_argument('--thresh', type=float, default=0.3, help='bbox score threshold')
        parser.add_argument('--batch-size', type=int, default=1, help='batch size')
        args = parser.parse_args()
        return args

    args = parse_args()
    
    def get_cat(data_file):
        
        cats = []

        with open(data_file) as f:
            for i in f.readlines():
                if i.endswith('.names'):
                    names = i.split(' ')[-1]

        with open(names) as f:
            for idx, i in enumerate(f.readlines()):
                cat_dict = {'id':idx, 'name': i.strip()}
                cats.append(cat_dict)
            
        return cats


    def get_data(detections, cats):
        class_id_list = []
        bbox_list = []
        area_list = []
        for i in detections:
            class_name = i[0]
            left = int(i[2][0])
            top = int(i[2][1])
            bbox_width = int(i[2][2])
            bbox_height = int(i[2][3])
            bbox = [left, top, bbox_width, bbox_height]
            area = bbox_width*bbox_height
            area_list.append(area)
            bbox_list.append(bbox)
            for i in cats:
                if i['name'] == class_name:
                    class_id = i['id']
            class_id_list.append(class_id)
        return bbox_list, class_id_list, area_list

    img_path = load_images(args.img_root)

    categories = get_cat(args.data_file)

    img_anno_dict = {'images':[], 'annotations': [], 'categories': categories}

    #load network
    network, class_names, class_colors = darknet.load_network(args.config_file, args.data_file, args.weights, args.batch_size)

    ann_id = 0
    
    for img_id, image in enumerate(img_path):

        img_width, img_height = imagesize.get(image)

        img_name = image.split('/')[-1]


        # get result bboxes
        image, detections = image_detection(image, network, class_names, class_colors, args.thresh)


        bbox_list, class_id_list, area_list = get_data(detections, categories)

        
        images = {
        'file_name': img_name,
        'height': img_height,
        'width': img_width,
        'id': img_id
        }
        
        for bl, cil, al in zip(bbox_list, class_id_list, area_list):
            annotations = {
            'id': ann_id,
            'image_id': img_id,
            'category_id': cil,
            'area': al,
            'bbox': bl,
            'iscrowd': 0
            }
            ann_id = ann_id+1
            img_anno_dict['annotations'].append(annotations)
            

        img_anno_dict['images'].append(images)

    with open(args.json_path, "w") as outfile:
        json.dump(img_anno_dict, outfile, indent=2)

if __name__ == '__main__':
    main()


