#!/bin/python
import argparse
import glob
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from lxml import etree
import xml.etree.cElementTree as ET


DELAY = 20 # keyboard delay (in milliseconds)
WITH_QT = False
try:
    cv2.namedWindow('Test')
    cv2.displayOverlay('Test', 'Test QT', 500)
    WITH_QT = True
except cv2.error:
    print('-> Please ignore this error message\n')
cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description='Open-source image labeling tool')
parser.add_argument('-i', '--input_dir', default='input', type=str, help='Path to input directory')
parser.add_argument('-o', '--output_dir', default='output', type=str, help='Path to output directory')
parser.add_argument('-t', '--thickness', default='1', type=int, help='Bounding box and cross line thickness')
args = parser.parse_args()

class_index = 0
img_index = 0
img = None
img_objects = []

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir

WINDOW_NAME = 'OpenLabeling'
TRACKBAR_IMG = 'Image'
TRACKBAR_CLASS = 'Class'

annotation_formats = {'PASCAL_VOC' : '.xml', 'YOLO_darknet' : '.txt'}
TRACKER_DIR = os.path.join(OUTPUT_DIR, '.tracker')

# selected bounding box
prev_was_double_click = False
is_bbox_selected = False
selected_bbox = -1
line_thickness = args.thickness

mouse_x = 0
mouse_y = 0
point_1 = (-1, -1)
point_2 = (-1, -1)


def display_text(text, time):
    if WITH_QT:
        cv2.displayOverlay(WINDOW_NAME, text, time)
    else:
        print(text)

def set_img_index(x):
    global img_index, img
    img_index = x
    img_path = IMAGE_PATH_LIST[img_index]
    img = cv2.imread(img_path)
    text = 'Showing image {}/{}, path: {}'.format(str(img_index), str(last_img_index), img_path)
    display_text(text, 1000)


def set_class_index(x):
    global class_index
    class_index = x
    text = 'Selected class {}/{} -> {}'.format(str(class_index), str(last_class_index), CLASS_LIST[class_index])
    display_text(text, 3000)


def draw_edges(tmp_img):
    blur = cv2.bilateralFilter(tmp_img, 3, 75, 75)
    edges = cv2.Canny(blur, 150, 250, 3)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Overlap image and edges together
    tmp_img = np.bitwise_or(tmp_img, edges)
    #tmp_img = cv2.addWeighted(tmp_img, 1 - edges_val, edges, edges_val, 0)
    return tmp_img


def decrease_index(current_index, last_index):
    current_index -= 1
    if current_index < 0:
        current_index = last_index
    return current_index


def increase_index(current_index, last_index):
    current_index += 1
    if current_index > last_index:
        current_index = 0
    return current_index


def draw_line(img, x, y, height, width, color):
    cv2.line(img, (x, 0), (x, height), color, line_thickness)
    cv2.line(img, (0, y), (width, y), color, line_thickness)


def yolo_format(class_index, point_1, point_2, width, height):
    # YOLO wants everything normalized
    # Order: class x_center y_center x_width y_height
    x_center = (point_1[0] + point_2[0]) / float(2.0 * width)
    y_center = (point_1[1] + point_2[1]) / float(2.0 * height)
    x_width = float(abs(point_2[0] - point_1[0])) / width
    y_height = float(abs(point_2[1] - point_1[1])) / height
    items = map(str, [class_index, x_center, y_center, x_width, y_height])
    return ' '.join(items)


def voc_format(class_name, point_1, point_2):
    # Order: class_name xmin ymin xmax ymax
    xmin, ymin = min(point_1[0], point_2[0]), min(point_1[1], point_2[1])
    xmax, ymax = max(point_1[0], point_2[0]), max(point_1[1], point_2[1])
    items = map(str, [class_name, xmin, ymin, xmax, ymax])
    return items


def write_xml(xml_str, xml_path):
    # remove blank text before prettifying the xml
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(xml_str, parser)
    # prettify
    xml_str = etree.tostring(root, pretty_print=True)
    # save to file
    with open(xml_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)


def append_bb(ann_path, line, extension):
    if '.txt' in extension:
        with open(ann_path, 'a') as myfile:
            myfile.write(line + '\n') # append line
    elif '.xml' in extension:
        class_name, xmin, ymin, xmax, ymax = line

        tree = ET.parse(ann_path)
        annotation = tree.getroot()

        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = class_name
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        bbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = xmin
        ET.SubElement(bbox, 'ymin').text = ymin
        ET.SubElement(bbox, 'xmax').text = xmax
        ET.SubElement(bbox, 'ymax').text = ymax

        xml_str = ET.tostring(annotation)
        write_xml(xml_str, ann_path)


def yolo_to_voc(x_center, y_center, x_width, y_height, width, height):
    x_center *= float(width)
    y_center *= float(height)
    x_width *= float(width)
    y_height *= float(height)
    x_width /= 2.0
    y_height /= 2.0
    xmin = int(round(x_center - x_width))
    ymin = int(round(y_center - y_height))
    xmax = int(round(x_center + x_width))
    ymax = int(round(y_center + y_height))
    return xmin, ymin, xmax, ymax


def draw_text(tmp_img, text, center, color, size):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(tmp_img, text, center, font, 0.6, color, size, cv2.LINE_AA)
    return tmp_img


def get_xml_object_data(obj):
    class_name = obj.find('name').text
    class_index = CLASS_LIST.index(class_name)
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    xmax = int(bndbox.find('xmax').text)
    ymin = int(bndbox.find('ymin').text)
    ymax = int(bndbox.find('ymax').text)
    return [class_name, class_index, xmin, ymin, xmax, ymax]


def draw_bboxes_from_file(tmp_img, annotation_paths, width, height):
    global img_objects
    img_objects = []
    ann_path = next(path for path in annotation_paths if 'PASCAL_VOC' in path)
    if os.path.isfile(ann_path):
        tree = ET.parse(ann_path)
        annotation = tree.getroot()
        for obj in annotation.findall('object'):
            class_name, class_index, xmin, ymin, xmax, ymax = get_xml_object_data(obj)
            #print('{} {} {} {} {}'.format(class_index, xmin, ymin, xmax, ymax))
            img_objects.append([class_index, xmin, ymin, xmax, ymax])
            color = class_rgb[class_index].tolist()
            cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), color, line_thickness)
            tmp_img = draw_text(tmp_img, class_name, (xmin, ymin - 5), color, line_thickness)
    return tmp_img


def get_bbox_area(x1, y1, x2, y2):
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return width*height


def set_selected_bbox(set_class):
    global is_bbox_selected, selected_bbox
    smallest_area = -1
    # if clicked inside multiple bboxes selects the smallest one
    for idx, obj in enumerate(img_objects):
        ind, x1, y1, x2, y2 = obj
        if is_mouse_inside_points(x1, y1, x2, y2):
            is_bbox_selected = True
            tmp_area = get_bbox_area(x1, y1, x2, y2)
            if tmp_area < smallest_area or smallest_area == -1:
                smallest_area = tmp_area
                selected_bbox = idx
                if set_class:
                    # set class to the one of the selected bounding box
                    cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, ind)


def is_mouse_inside_delete_button():
    for idx, obj in enumerate(img_objects):
        if idx == selected_bbox:
            _ind, x1, y1, x2, y2 = obj
            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            if is_mouse_inside_points(x1_c, y1_c, x2_c, y2_c):
                return True
    return False


def edit_bbox(action):
    ''' action = (delete) or (change_class:[class_index]) '''
    if 'change_class' in action:
        new_class_index = int(action.split(':')[1])

    # 1. initialize bboxes_to_edit_dict
    bboxes_to_edit_dict = {}
    current_img_path = IMAGE_PATH_LIST[img_index]
    obj_to_edit = img_objects[selected_bbox]
    bboxes_to_edit_dict[current_img_path] = obj_to_edit

    # 2. add elements to bboxes_to_edit_dict
    '''
        If the bbox is in the json file then it was used by the video Tracker, hence,
        we must also edit the next predicted bboxes associated to the same `anchor_id`.
    '''
    # if `current_img_path` is a frame from a video
    is_from_video, video_name = is_frame_from_video(current_img_path)
    if is_from_video:
        # get json file corresponding to that video
        json_file_path = '{}.json'.format(os.path.join(TRACKER_DIR, video_name))
        file_exists, json_file_data = get_json_file_data(json_file_path)
        # if json file exists
        if file_exists:
            # match obj_to_edit with the corresponding json object
            frame_data_dict = json_file_data['frame_data_dict']
            json_object_list = get_json_file_object_list(current_img_path, frame_data_dict)
            obj_to_edit = img_objects[selected_bbox]
            obj_matched = get_json_object_dict(obj_to_edit, json_object_list)
            # if match found
            if obj_matched is not None:
                # get this object's anchor_id
                anchor_id = obj_matched['anchor_id']

                frame_path_list = get_next_frame_path_list(video_name, current_img_path)
                frame_path_list.insert(0, current_img_path)

                if 'change_class' in action:
                    # add also the previous frames
                    prev_path_list = get_prev_frame_path_list(video_name, current_img_path)
                    frame_path_list = prev_path_list + frame_path_list

                # update json file if contain the same anchor_id
                for frame_path in frame_path_list:
                    json_object_list = get_json_file_object_list(frame_path, frame_data_dict)
                    json_obj = get_json_file_object_by_id(json_object_list, anchor_id)
                    if json_obj is not None:
                        bboxes_to_edit_dict[frame_path] = [
                            json_obj['class_index'],
                            json_obj['bbox']['xmin'],
                            json_obj['bbox']['ymin'],
                            json_obj['bbox']['xmax'],
                            json_obj['bbox']['ymax']
                        ]
                        # edit json file
                        if 'delete' in action:
                            json_object_list.remove(json_obj)
                        elif 'change_class' in action:
                            json_obj['class_index'] = new_class_index
                    else:
                        break

                # save the edited data
                with open(json_file_path, 'w') as outfile:
                    json.dump(json_file_data, outfile, sort_keys=True, indent=4)

    # 3. loop through bboxes_to_edit_dict and edit the corresponding annotation files
    for path in bboxes_to_edit_dict:
        obj_to_edit = bboxes_to_edit_dict[path]
        class_index, xmin, ymin, xmax, ymax = map(int, obj_to_edit)

        for ann_path in get_annotation_paths(path, annotation_formats):
            if '.txt' in ann_path:
                with open(ann_path, 'r') as old_file:
                    lines = old_file.readlines()

                yolo_line = yolo_format(class_index, (xmin, ymin), (xmax, ymax), width, height) # TODO: height and width ought to be stored

                with open(ann_path, 'w') as new_file:
                    for line in lines:
                        if line != yolo_line + '\n':
                            new_file.write(line)
                        elif 'change_class' in action:
                            items = line.split()
                            items[0] = str(new_class_index)
                            new_file.write(' '.join(items) + '\n')
            elif '.xml' in ann_path:
                tree = ET.parse(ann_path)
                annotation = tree.getroot()
                for obj in annotation.findall('object'):
                    class_name_xml, class_index_xml, xmin_xml, ymin_xml, xmax_xml, ymax_xml = get_xml_object_data(obj)
                    if ( class_index == class_index_xml and
                                     xmin == xmin_xml and
                                     ymin == ymin_xml and
                                     xmax == xmax_xml and
                                     ymax == ymax_xml ) :
                        if 'delete' in action:
                            annotation.remove(obj)
                        else:
                            # edit object class name
                            object_class = obj.find('name')
                            object_class.text = CLASS_LIST[new_class_index]
                        break

                xml_str = ET.tostring(annotation)
                write_xml(xml_str, ann_path)


def mouse_listener(event, x, y, flags, param):
    # mouse callback function
    global is_bbox_selected, prev_was_double_click, mouse_x, mouse_y, point_1, point_2

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        prev_was_double_click = True
        #print('Double click')
        point_1 = (-1, -1)
        # if clicked inside a bounding box we set that bbox
        set_class = True
        set_selected_bbox(set_class)
    # By AlexeyGy: delete via right-click
    elif event == cv2.EVENT_RBUTTONDOWN:
        set_class = False
        set_selected_bbox(set_class)
        if is_bbox_selected:
            edit_bbox('delete')
            is_bbox_selected = False
    elif event == cv2.EVENT_LBUTTONDOWN:
        if prev_was_double_click:
            #print('Finish double click')
            prev_was_double_click = False

        #print('Normal left click')
        if point_1[0] is -1:
            if is_bbox_selected:
                if is_mouse_inside_delete_button():
                    edit_bbox('delete')
                is_bbox_selected = False
            else:
                # first click (start drawing a bounding box or delete an item)
                point_1 = (x, y)
        else:
            # minimal size for bounding box to avoid errors
            threshold = 20
            if abs(x - point_1[0]) > threshold or abs(y - point_1[1]) > threshold:
                # second click
                point_2 = (x, y)


def is_mouse_inside_points(x1, y1, x2, y2):
    return mouse_x > x1 and mouse_x < x2 and mouse_y > y1 and mouse_y < y2


def get_close_icon(x1, y1, x2, y2):
    percentage = 0.05
    height = -1
    while height < 15 and percentage < 1.0:
        height = int((y2 - y1) * percentage)
        percentage += 0.1
    return (x2 - height), y1, x2, (y1 + height)


def draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c):
    red = (0,0,255)
    cv2.rectangle(tmp_img, (x1_c + 1, y1_c - 1), (x2_c, y2_c), red, -1)
    white = (255, 255, 255)
    cv2.line(tmp_img, (x1_c, y1_c), (x2_c, y2_c), white, 2)
    cv2.line(tmp_img, (x1_c, y2_c), (x2_c, y1_c), white, 2)
    return tmp_img


def draw_info_bb_selected(tmp_img):
    for idx, obj in enumerate(img_objects):
        ind, x1, y1, x2, y2 = obj
        if idx == selected_bbox:
            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c)
    return tmp_img


def sort_video_frames(x):
    # format: [video_name_ext]_[frame number].[image format]
    # ex: video_sample_0.jpg, video_sample_1.jpg, ...
    start = x.rfind('_') + 1
    end = x.rfind('.', start)
    return int(x[start:end]) # we want to sort by the [frame number]


def convert_video_to_images(video_path, n_frames, desired_img_format):
    # create folder to store images (if video was not converted to images already)
    file_path, file_extension = os.path.splitext(video_path)
    # append extension to avoid collision of videos with same name
    # e.g.: `video.mp4`, `video.avi` -> `video_mp4/`, `video_avi/`
    file_extension = file_extension.replace('.', '_')
    file_path += file_extension
    video_name_ext = os.path.basename(file_path)
    if not os.path.exists(file_path):
        print(' Converting video to individual frames...')
        cap = cv2.VideoCapture(video_path)
        os.makedirs(file_path)
        # read the video
        for i in tqdm(range(n_frames)):
            if not cap.isOpened():
                break
            # capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # save each frame (we use this format to avoid repetitions)
                frame_name =  '{}_{}{}'.format(video_name_ext, i, desired_img_format)
                frame_path = os.path.join(file_path, frame_name)
                cv2.imwrite(frame_path, frame)
        # release the video capture object
        cap.release()
    return file_path, video_name_ext


def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line


def get_annotation_paths(img_path, annotation_formats):
    annotation_paths = []
    for ann_dir, ann_ext in annotation_formats.items():
        new_path = os.path.join(OUTPUT_DIR, ann_dir)
        new_path = img_path.replace(INPUT_DIR, new_path, 1)
        pre_path, img_ext = os.path.splitext(new_path)
        new_path = new_path.replace(img_ext, ann_ext, 1)
        annotation_paths.append(new_path)
    return annotation_paths


def create_PASCAL_VOC_xml(xml_path, abs_path, folder_name, image_name, img_height, img_width, depth):
    # By: Jatin Kumar Mandav
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder_name
    ET.SubElement(annotation, 'filename').text = image_name
    ET.SubElement(annotation, 'path').text = abs_path
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = img_width
    ET.SubElement(size, 'height').text = img_height
    ET.SubElement(size, 'depth').text = depth
    ET.SubElement(annotation, 'segmented').text = '0'

    xml_str = ET.tostring(annotation)
    write_xml(xml_str, xml_path)


def save_bounding_box(annotation_paths, class_index, point_1, point_2, width, height):
    for ann_path in annotation_paths:
        if '.txt' in ann_path:
            line = yolo_format(class_index, point_1, point_2, width, height)
            append_bb(ann_path, line, '.txt')
        elif '.xml' in ann_path:
            line = voc_format(CLASS_LIST[class_index], point_1, point_2)
            append_bb(ann_path, line, '.xml')

def is_frame_from_video(img_path):
    for video_name in VIDEO_NAME_DICT:
        video_dir = os.path.join(INPUT_DIR, video_name)
        if os.path.dirname(img_path) == video_dir:
            # image belongs to a video
            return True, video_name
    return False, None


def get_json_file_data(json_file_path):
    if os.path.isfile(json_file_path):
        with open(json_file_path) as f:
            data = json.load(f)
            return True, data
    else:
        return False, {'n_anchor_ids':0, 'frame_data_dict':{}}


def get_prev_frame_path_list(video_name, img_path):
    first_index = VIDEO_NAME_DICT[video_name]['first_index']
    last_index = VIDEO_NAME_DICT[video_name]['last_index']
    img_index = IMAGE_PATH_LIST.index(img_path)
    return IMAGE_PATH_LIST[first_index:img_index]


def get_next_frame_path_list(video_name, img_path):
    first_index = VIDEO_NAME_DICT[video_name]['first_index']
    last_index = VIDEO_NAME_DICT[video_name]['last_index']
    img_index = IMAGE_PATH_LIST.index(img_path)
    return IMAGE_PATH_LIST[(img_index + 1):last_index]


def get_json_object_dict(obj, json_object_list):
    if len(json_object_list) > 0:
        class_index, xmin, ymin, xmax, ymax = map(int, obj)
        for d in json_object_list:
                    if ( d['class_index'] == class_index and
                         d['bbox']['xmin'] == xmin and
                         d['bbox']['ymin'] == ymin and
                         d['bbox']['xmax'] == xmax and
                         d['bbox']['ymax'] == ymax ) :
                        return d
    return None


def remove_already_tracked_objects(object_list, img_path, json_file_data):
    frame_data_dict = json_file_data['frame_data_dict']
    json_object_list = get_json_file_object_list(img_path, frame_data_dict)
    # copy the list since we will be deleting elements without restarting the loop
    temp_object_list = object_list[:]
    for obj in temp_object_list:
        obj_dict = get_json_object_dict(obj, json_object_list)
        if obj_dict is not None:
            object_list.remove(obj)
            json_object_list.remove(obj_dict)
    return object_list


def get_json_file_object_by_id(json_object_list, anchor_id):
    for obj_dict in json_object_list:
        if obj_dict['anchor_id'] == anchor_id:
            return obj_dict
    return None


def get_json_file_object_list(img_path, frame_data_dict):
    object_list = []
    if img_path in frame_data_dict:
        object_list = frame_data_dict[img_path]
    return object_list


def json_file_add_object(frame_data_dict, img_path, anchor_id, pred_counter, obj):
    object_list = get_json_file_object_list(img_path, frame_data_dict)
    class_index, xmin, ymin, xmax, ymax = obj

    bbox = {
      'xmin': xmin,
      'ymin': ymin,
      'xmax': xmax,
      'ymax': ymax
    }

    temp_obj = {
      'anchor_id': anchor_id,
      'prediction_index': pred_counter,
      'class_index': class_index,
      'bbox': bbox
    }

    object_list.append(temp_obj)
    frame_data_dict[img_path] = object_list

    return frame_data_dict


class LabelTracker():
    ''' Special thanks to Rafael Caballero Gonzalez '''
    # extract the OpenCV version info, e.g.: 
    # OpenCV 3.3.4 -> [major_ver].[minor_ver].[subminor_ver]
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # TODO: press ESC to stop the tracking process

    def __init__(self, tracker_type, init_frame, next_frame_path_list):
        tracker_types = ['CSRT', 'KCF','MOSSE', 'MIL', 'BOOSTING', 'MEDIANFLOW', 'TLD', 'GOTURN']
        ''' Recomended tracker_type:
              KCF -> KCF is usually very good (minimum OpenCV 3.1.0)
              CSRT -> More accurate than KCF but slightly slower (minimum OpenCV 3.4.2)
              MOSSE -> Less accurate than KCF but very fast (minimum OpenCV 3.4.1)
        '''
        self.tracker_type = tracker_type
        # -- TODO: remove this if I assume OpenCV version > 3.4.0
        if tracker_type == tracker_types[0] or tracker_type == tracker_types[2]:
            if int(self.major_ver == 3) and int(self.minor_ver) < 4:
                self.tracker_type = tracker_types[1] # Use KCF instead of CSRT or MOSSE
        # --
        self.init_frame = init_frame
        self.next_frame_path_list = next_frame_path_list

        self.img_h, self.img_w = init_frame.shape[:2]


    def call_tracker_constructor(self, tracker_type):
        # -- TODO: remove this if I assume OpenCV version > 3.4.0
        if int(self.major_ver == 3) and int(self.minor_ver) < 3:
            tracker = cv2.Tracker_create(tracker_type)
        # --
        else:
            if tracker_type == 'CSRT':
                tracker = cv2.TrackerCSRT_create()
            elif tracker_type == 'KCF':
                tracker = cv2.TrackerKCF_create()
            elif tracker_type == 'MOSSE':
                tracker = cv2.TrackerMOSSE_create()
            elif tracker_type == 'MIL':
                tracker = cv2.TrackerMIL_create()
            elif tracker_type == 'BOOSTING':
                tracker = cv2.TrackerBoosting_create()
            elif tracker_type == 'MEDIANFLOW':
                tracker = cv2.TrackerMedianFlow_create()
            elif tracker_type == 'TLD':
                tracker = cv2.TrackerTLD_create()
            elif tracker_type == 'GOTURN':
                tracker = cv2.TrackerGOTURN_create()
        return tracker


    def start_tracker(self, json_file_data, json_file_path, img_path, obj, color, annotation_formats):
        tracker = self.call_tracker_constructor(self.tracker_type)
        anchor_id = json_file_data['n_anchor_ids']
        frame_data_dict = json_file_data['frame_data_dict']

        pred_counter = 0
        frame_data_dict = json_file_add_object(frame_data_dict, img_path, anchor_id, pred_counter, obj)
        # tracker bbox format: xmin, xmax, w, h
        xmin, ymin, xmax, ymax = obj[1:5]
        initial_bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
        tracker.init(self.init_frame, initial_bbox)
        for frame_path in self.next_frame_path_list:
            next_image = cv2.imread(frame_path)
            # get the new bbox prediction of the object
            success, bbox = tracker.update(next_image.copy())
            if success:
                pred_counter += 1
                xmin, ymin, w, h = map(int, bbox)
                xmax = xmin + w
                ymax = ymin + h
                obj = [class_index, xmin, ymin, xmax, ymax]
                frame_data_dict = json_file_add_object(frame_data_dict, frame_path, anchor_id, pred_counter, obj)
                cv2.rectangle(next_image, (xmin, ymin), (xmax, ymax), color, line_thickness)
                # save prediction
                annotation_paths = get_annotation_paths(frame_path, annotation_formats)
                save_bounding_box(annotation_paths, class_index, (xmin, ymin), (xmax, ymax), self.img_w, self.img_h)
                # show prediction
                cv2.imshow(WINDOW_NAME, next_image)
                pressed_key = cv2.waitKey(DELAY)
            else:
                break

        json_file_data.update({'n_anchor_ids': (anchor_id + 1)})
        # save the updated data
        with open(json_file_path, 'w') as outfile:
            json.dump(json_file_data, outfile, sort_keys=True, indent=4)


# change to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# load all images and videos (with multiple extensions) from a directory using OpenCV
IMAGE_PATH_LIST = []
VIDEO_NAME_DICT = {}
for f in sorted(os.listdir(INPUT_DIR)):
    f_path = os.path.join(INPUT_DIR, f)
    if os.path.isdir(f_path):
        # skip directories
        continue
    # check if it is an image
    test_img = cv2.imread(f_path)
    if test_img is not None:
        IMAGE_PATH_LIST.append(f_path)
    else:
        # test if it is a video
        test_video_cap = cv2.VideoCapture(f_path)
        n_frames = int(test_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        test_video_cap.release()
        if n_frames > 0:
            # it is a video
            desired_img_format = '.jpg'
            video_frames_path, video_name_ext = convert_video_to_images(f_path, n_frames, desired_img_format)
            # add video frames to image list
            frame_list = sorted(os.listdir(video_frames_path), key = sort_video_frames)
            ## store information about those frames
            first_index = len(IMAGE_PATH_LIST)
            last_index = first_index + len(frame_list) # exclusive
            indexes_dict = {}
            indexes_dict['first_index'] = first_index
            indexes_dict['last_index'] = last_index
            VIDEO_NAME_DICT[video_name_ext] = indexes_dict
            IMAGE_PATH_LIST.extend((os.path.join(video_frames_path, frame) for frame in frame_list))
last_img_index = len(IMAGE_PATH_LIST) - 1

# create output directories
if len(VIDEO_NAME_DICT) > 0:
    if not os.path.exists(TRACKER_DIR):
        os.makedirs(TRACKER_DIR)
for ann_dir in annotation_formats:
    new_dir = os.path.join(OUTPUT_DIR, ann_dir)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for video_name_ext in VIDEO_NAME_DICT:
        new_video_dir = os.path.join(new_dir, video_name_ext)
        if not os.path.exists(new_video_dir):
            os.makedirs(new_video_dir)

# create empty annotation files for each image, if it doesn't exist already
for img_path in IMAGE_PATH_LIST:
    # image info for the .xml file
    test_img = cv2.imread(img_path)
    abs_path = os.path.abspath(img_path)
    folder_name = os.path.dirname(img_path)
    image_name = os.path.basename(img_path)
    img_height, img_width, depth = (str(number) for number in test_img.shape)

    for ann_path in get_annotation_paths(img_path, annotation_formats):
        if not os.path.isfile(ann_path):
            if '.txt' in ann_path:
                open(ann_path, 'a').close()
            elif '.xml' in ann_path:
                create_PASCAL_VOC_xml(ann_path, abs_path, folder_name, image_name, img_height, img_width, depth)

# load class list
with open('class_list.txt') as f:
    CLASS_LIST = list(nonblank_lines(f))
#print(CLASS_LIST)
last_class_index = len(CLASS_LIST) - 1

# Make the class colors the same each session
# The colors are in BGR order because we're using OpenCV
class_rgb = [
    (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0),
    (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128)]
class_rgb = np.array(class_rgb)
# If there are still more classes, add new colors randomly
num_colors_missing = len(CLASS_LIST) - len(class_rgb)
if num_colors_missing > 0:
    more_colors = np.random.randint(0, 255+1, size=(num_colors_missing, 3))
    class_rgb = np.vstack([class_rgb, more_colors])

# create window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_NAME, 1000, 700)
cv2.setMouseCallback(WINDOW_NAME, mouse_listener)

# selected image
cv2.createTrackbar(TRACKBAR_IMG, WINDOW_NAME, 0, last_img_index, set_img_index)

# selected class
if last_class_index != 0:
    cv2.createTrackbar(TRACKBAR_CLASS, WINDOW_NAME, 0, last_class_index, set_class_index)

# initialize
set_img_index(0)
edges_on = False

display_text('Welcome!\n Press [h] for help.', 4000)

# loop
while True:
    color = class_rgb[class_index].tolist()
    # clone the img
    tmp_img = img.copy()
    height, width = tmp_img.shape[:2]
    if edges_on == True:
        # draw edges
        tmp_img = draw_edges(tmp_img)
    # draw vertical and horizontal guide lines
    draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
    # write selected class
    tmp_img = draw_text(tmp_img, CLASS_LIST[class_index], (mouse_x + 5, mouse_y - 5), color, line_thickness)
    img_path = IMAGE_PATH_LIST[img_index]
    annotation_paths = get_annotation_paths(img_path, annotation_formats)
    # draw already done bounding boxes
    tmp_img = draw_bboxes_from_file(tmp_img, annotation_paths, width, height)
    # if bounding box is selected add extra info
    if is_bbox_selected:
        tmp_img = draw_info_bb_selected(tmp_img)
    # if first click
    if point_1[0] is not -1:
        # draw partial bbox
        cv2.rectangle(tmp_img, point_1, (mouse_x, mouse_y), color, line_thickness)
        # if second click
        if point_2[0] is not -1:
            # save the bounding box
            save_bounding_box(annotation_paths, class_index, point_1, point_2, width, height)
            # reset the points
            point_1 = (-1, -1)
            point_2 = (-1, -1)

    cv2.imshow(WINDOW_NAME, tmp_img)
    pressed_key = cv2.waitKey(DELAY)

    ''' Key Listeners START '''
    if pressed_key == ord('a') or pressed_key == ord('d'):
        # show previous image key listener
        if pressed_key == ord('a'):
            img_index = decrease_index(img_index, last_img_index)
        # show next image key listener
        elif pressed_key == ord('d'):
            img_index = increase_index(img_index, last_img_index)
        cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)
    elif pressed_key == ord('s') or pressed_key == ord('w'):
        # change down current class key listener
        if pressed_key == ord('s'):
            class_index = decrease_index(class_index, last_class_index)
        # change up current class key listener
        elif pressed_key == ord('w'):
            class_index = increase_index(class_index, last_class_index)
        draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
        cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_index)
        if is_bbox_selected:
            edit_bbox('change_class:{}'.format(class_index))
    # help key listener
    elif pressed_key == ord('h'):
        text = ('[e] to show edges;\n'
                '[q] to quit;\n'
                '[a] or [d] to change Image;\n'
                '[w] or [s] to change Class.\n'
                )
        display_text(text, 5000)
    # show edges key listener
    elif pressed_key == ord('e'):
        if edges_on == True:
            edges_on = False
            display_text('Edges turned OFF!', 1000)
        else:
            edges_on = True
            display_text('Edges turned ON!', 1000)
    elif pressed_key == ord('p'):
        # check if the image is a frame from a video
        is_from_video, video_name = is_frame_from_video(img_path)
        if is_from_video:
            # get list of objects associated to that frame
            object_list = img_objects[:]
            # remove the objects in that frame that are already in the `.json` file
            json_file_path = '{}.json'.format(os.path.join(TRACKER_DIR, video_name))
            file_exists, json_file_data = get_json_file_data(json_file_path)
            if file_exists:
                object_list = remove_already_tracked_objects(object_list, img_path, json_file_data)
            if len(object_list) > 0:
                # get list of frames following this image
                next_frame_path_list = get_next_frame_path_list(video_name, img_path)
                # initial frame
                init_frame = img.copy()
                label_tracker = LabelTracker('KCF', init_frame, next_frame_path_list) # TODO: replace 'KCF' by 'CSRT'
                for obj in object_list:
                    class_index = obj[0]
                    color = class_rgb[class_index].tolist()
                    label_tracker.start_tracker(json_file_data, json_file_path, img_path, obj, color, annotation_formats)
    # quit key listener
    elif pressed_key == ord('q'):
        break
    ''' Key Listeners END '''

    if WITH_QT:
        # if window gets closed then quit
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

cv2.destroyAllWindows()
