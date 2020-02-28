import cv2 
import numpy as np
import os
import sys
from tqdm import tqdm
import math
import conf

# DEPTH = 4 -> 4 * 4 * 4 = 64 colors
DEPTH = conf.DEPTH
# list of rotations, in degrees, to apply over the original image
ROTATIONS = conf.ROTATIONS

img_path = sys.argv[1]
img_dir = os.path.dirname(img_path)
img_name, ext = os.path.basename(img_path).rsplit('.', 1)
out_folder = img_dir + '/gen_' + img_name

if not os.path.exists(out_folder):
    os.mkdir(out_folder)

img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
img = img.astype('float')

height, width, channels = img.shape
center = (width/2, height/2)

for b in tqdm(np.arange(0, 1.01, 1 / DEPTH)):
    for g in np.arange(0, 1.01, 1 / DEPTH):
        for r in np.arange(0, 1.01, 1 / DEPTH):
            mult_vector = [b, g, r]
            if channels == 4:
                mult_vector.append(1)
            new_img = img * mult_vector
            new_img = new_img.astype('uint8')
            for rotation in ROTATIONS:
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1)
                abs_cos = abs(rotation_matrix[0,0])
                abs_sin = abs(rotation_matrix[0,1])
                new_w = int(height * abs_sin + width * abs_cos)
                new_h = int(height * abs_cos + width * abs_sin)
                rotation_matrix[0, 2] += new_w/2 - center[0]
                rotation_matrix[1, 2] += new_h/2 - center[1]
                cv2.imwrite(
                    f'{out_folder}/{img_name}_{round(r,1)}_{round(g,1)}_{round(b,1)}_r{rotation}.{ext}',
                    cv2.warpAffine(new_img, rotation_matrix, (new_w, new_h)),
                    # compress image
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
