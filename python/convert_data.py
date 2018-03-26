import sys
import os
import xml.etree.ElementTree as ET


CLASSES = ('complementary_signs', 'white_signs', 'cars')
SETS = ('train', 'val')

OUTPUT_TEMPLATE = '''
classes = {}
train  = {}
valid  = {}
names = classes.names
backup = backup
'''


def convert_label(name):
    if "complementary" in name or "warning" in name or "yellow" in name or "winding--ramp" in name:
        return CLASSES[0]
    elif "speed_limit" in name or "minimum" in name or "trucks-" in name or "end" in name or "other_white" in name or "speed-limit" in name or "night" in name:
        return CLASSES[1]
    elif "car_part" in name or "truck" in name or "truk" in name or "car" in name or "minivan" in name or "van" in name or "bus" in name or "trailer" in name:
        return CLASSES[2]


def convert_bbox(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return x, y, w, h


def convert_annotation(in_file_path, out_file_path):
    in_file = open(in_file_path, 'r')
    out_file = open(out_file_path, 'w')

    tree = ET.parse(in_file)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = convert_label(obj.find('name').text)

        if not cls:
            continue

        cls_id = CLASSES.index(cls)
        xmlbox = obj.find('bndbox')

        b = (
            float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
            float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)
        )

        bb = convert_bbox((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    out_file.close()
    in_file.close()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Dataset path must be specified!')

    dataset_path = sys.argv[1]

    outputs = {}
    for set_type in SETS:
        subset_path = os.path.join(dataset_path, set_type)

        list_file_path = os.path.join(subset_path, '{}.txt'.format(set_type))
        print('Generating {}'.format(list_file_path))

        images_path = os.path.join(subset_path, 'images')
        annotations_path = os.path.join(subset_path, 'ann')
        labels_path = os.path.join(subset_path, 'labels')

        if not os.path.exists(labels_path):
            os.mkdir(labels_path)

        with open(list_file_path, 'w') as list_file:
            for image_file in os.listdir(images_path):
                try:
                    image_name = os.path.splitext(image_file)[0]
                    annotation_file = '{}.xml'.format(image_name)
                    label_file = '{}.txt'.format(image_name)

                    image_path = os.path.join(images_path, image_file)
                    annotation_path = os.path.join(annotations_path, annotation_file)
                    label_path = os.path.join(labels_path, label_file)

                    assert os.path.exists(annotation_path), 'Annotation does not exists!'

                    convert_annotation(annotation_path, label_path)
                    list_file.write(image_path + '\n')
                except Exception as exc:
                    print('Cannot process {}: {}'.format(image_file, exc))

        outputs[set_type] = list_file_path

    print('Generating classes.names file')
    with open(os.path.join(dataset_path, 'classes.names'), 'w') as classes_file:
        for class_name in CLASSES:
            classes_file.write(class_name + '\n')

    print('Generating dataset.data file')
    with open(os.path.join(dataset_path, 'dataset.data'), 'w') as dataset_file:
        dataset_file.write(
            OUTPUT_TEMPLATE.format(len(CLASSES), outputs['train'], outputs['val'])
        )

    print('Done!')
