# GEN TILES CONFS

# number of divisions per color (R, G and B)
# DEPTH = 4 -> 4 * 4 * 4 = 64 colors
DEPTH = 4
# list of rotations, in degrees, to apply over the original image
ROTATIONS = [0]


#############################


# TILER CONFS

# number of colors per image
COLOR_DEPTH = 32
# tiles scales (1 = default resolution)
RESIZING_SCALES = [0.5, 0.4, 0.3, 0.2, 0.1]
# number of pixels shifted to create each box (tuple with (x,y))
# if value is None, shift will be done accordingly to tiles dimensions
PIXEL_SHIFT = (100, 100)
# if tiles can overlap
OVERLAP_TILES = True
# render image as its being built
RENDER = False
# multiprocessing pool size
POOL_SIZE = 8

# out file name
OUT = 'out.png'
# image to tile (ignored if passed as the 1st arg)
IMAGE_TO_TILE = None
# folder with tiles (ignored if passed as the 2nd arg)
TILES_FOLDER = None
