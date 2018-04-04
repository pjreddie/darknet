CLASSES = ('complementary_signs', 'white_signs', 'cars')
SETS = ('train', 'val')

OUTPUT_TEMPLATE = '''
classes = {}
train  = {}
valid  = {}
names = {}/classes.names
backup = backup
'''


def convert_label(name):
    if "complementary" in name or "warning" in name or "yellow" in name or "winding--ramp" in name:
        return CLASSES[0], 0
    elif "speed_limit" in name or "minimum" in name or "trucks-" in name or "end" in name or "other_white" in name or "speed-limit" in name or "night" in name:
        return CLASSES[1], 1
    elif "car_part" in name or "truck" in name or "truk" in name or "car" in name or "minivan" in name or "van" in name or "bus" in name or "trailer" in name:
        return CLASSES[2], 2

    return None, None


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
