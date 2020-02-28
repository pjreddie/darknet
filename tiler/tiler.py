import cv2
import numpy as np
import os
import sys
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
import math
import pickle
import conf
from time import sleep


# number of colors per image
COLOR_DEPTH = conf.COLOR_DEPTH
# tiles scales
RESIZING_SCALES = conf.RESIZING_SCALES
# number of pixels shifted to create each box (x,y)
PIXEL_SHIFT = conf.PIXEL_SHIFT
# multiprocessing pool size
POOL_SIZE = conf.POOL_SIZE
# if tiles can overlap
OVERLAP_TILES = conf.OVERLAP_TILES


# reduces the number of colors in an image
def color_quantization(img, n_colors):
    return np.round(img / 255 * n_colors) / n_colors * 255


# returns an image given its path
def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = color_quantization(img.astype('float'), COLOR_DEPTH)
    return img.astype('uint8')


# scales an image
def resize_image(img, ratio):
    img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)))
    return img


# the most frequent color in an image and its relative frequency
def mode_color(img, ingore_alpha=False):
    counter = defaultdict(int)
    total = 0
    for y in img:
        for x in y:
            if len(x) < 4 or ingore_alpha or x[3] != 0:
                counter[tuple(x[:3])] += 1
            else:
                counter[(-1,-1,-1)] += 1
            total += 1

    if total > 0:
        mode_color = max(counter, key=counter.get)
        if mode_color == (-1,-1,-1):
            return None, None
        else:
            return mode_color, counter[mode_color] / total
    else:
        return None, None


# displays an image
def show_image(img, wait=True):
    cv2.imshow('img', img)
    if wait:
        cv2.waitKey(0)
    else:
        cv2.waitKey(1)


# load and process the tiles
def load_tiles(paths):
    print('Loading tiles')
    tiles = defaultdict(list)

    for path in paths:
        if os.path.isdir(path):
            for tile_name in tqdm(os.listdir(path)):
                tile = read_image(os.path.join(path, tile_name))
                mode, rel_freq = mode_color(tile, ingore_alpha=True)
                if mode != None:
                    for scale in RESIZING_SCALES:
                        t = resize_image(tile, scale)
                        res = tuple(t.shape[:2])
                        tiles[res].append({
                            'tile': t,
                            'mode': mode,
                            'rel_freq': rel_freq
                        })

            with open('tiles.pickle', 'wb') as f:
                pickle.dump(tiles, f)
        
        # load pickle with tiles (one file only)
        else:
            with open(path, 'rb') as f:
                tiles = pickle.load(f)

    return tiles


# returns the boxes (image and start pos) from an image, with 'res' resolution
def image_boxes(img, res):
    if not PIXEL_SHIFT:
        shift = np.flip(res)
    else:
        shift = PIXEL_SHIFT

    boxes = []
    for y in range(0, img.shape[0], shift[1]):
        for x in range(0, img.shape[1], shift[0]):
            boxes.append({
                'img': img[y:y+res[0], x:x+res[1]],
                'pos': (x,y)
            })

    return boxes


# euclidean distance between two colors
def color_distance(c1, c2):
    c1_int = [int(x) for x in c1]
    c2_int = [int(x) for x in c2]
    return math.sqrt((c1_int[0] - c2_int[0])**2 + (c1_int[1] - c2_int[1])**2 + (c1_int[2] - c2_int[2])**2)


# returns the most similar tile to a box (in terms of color)
def most_similar_tile(box_mode_freq, tiles):
    if not box_mode_freq[0]:
        return (0, np.zeros(shape=tiles[0]['tile'].shape))
    else:
        min_distance = None
        min_tile_img = None
        for t in tiles:
            dist = (1 + color_distance(box_mode_freq[0], t['mode'])) / box_mode_freq[1]
            if min_distance == None or dist < min_distance:
                min_distance = dist
                min_tile_img = t['tile']
        return (min_distance, min_tile_img)


# builds the boxes and finds the best tile for each one
def get_processed_image_boxes(image_path, tiles):
    print('Gettting and processing boxes')
    img = read_image(image_path)
    pool = Pool(POOL_SIZE)
    all_boxes = []

    for res, ts in tqdm(sorted(tiles.items(), reverse=True)):
        boxes = image_boxes(img, res)
        modes = pool.map(mode_color, [x['img'] for x in boxes])
        most_similar_tiles = pool.starmap(most_similar_tile, zip(modes, [ts for x in range(len(modes))]))
        
        i = 0
        for min_dist, tile in most_similar_tiles:
            boxes[i]['min_dist'] = min_dist
            boxes[i]['tile'] = tile
            i += 1

        all_boxes += boxes

    return all_boxes, img.shape 


# places a tile in the image
def place_tile(img, box):
    p1 = np.flip(box['pos'])
    p2 = p1 + box['img'].shape[:2]
    img_box = img[p1[0]:p2[0], p1[1]:p2[1]]
    mask = box['tile'][:, :, 3] != 0
    mask = mask[:img_box.shape[0], :img_box.shape[1]]
    if OVERLAP_TILES or not np.any(img_box[mask]):
        img_box[mask] = box['tile'][:img_box.shape[0], :img_box.shape[1], :][mask]


# tiles the image
def create_tiled_image(boxes, res, render=False):
    print('Creating tiled image')
    img = np.zeros(shape=(res[0], res[1], 4), dtype=np.uint8)
    
    for box in tqdm(sorted(boxes, key=lambda x: x['min_dist'], reverse=OVERLAP_TILES)):
        place_tile(img, box)
        if render:
            show_image(img, wait=False)
            sleep(0.025)

    return img


# main
def main():
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = conf.IMAGE_TO_TILE
    
    if len(sys.argv) > 2:
        tiles_paths = sys.argv[2:]
    else:
        tiles_paths = conf.TILES_FOLDER.split(' ')

    if not os.path.exists(image_path):
        print('Image not found')
        exit(-1)
    for path in tiles_paths:
        if not os.path.exists(path):
            print('Tiles folder not found')
            exit(-1)

    tiles = load_tiles(tiles_paths)
    boxes, original_res = get_processed_image_boxes(image_path, tiles)
    img = create_tiled_image(boxes, original_res, render=conf.RENDER)
    cv2.imwrite(conf.OUT, img)


if __name__ == "__main__":
    main()
