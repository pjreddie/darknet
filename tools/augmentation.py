# augmentation using imgaug
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import glob
from skimage import data
import cv2

#path = "/media/elab/sdd/data/WildHog/wildhog_downsampling/"
path = "/media/elab/sdd/data/WildHog/augmentation/"

myfiles = []
for filename in glob.glob(path + '/*.jpg'):
    myfiles.append(filename)
print(myfiles)

def imshow(image):
    """
    Shows an image in a window.

    Parameters
    ----------
    image : (H,W,3) ndarray
        Image to show.
    """
    image_bgr = image
    if image.ndim == 3 and image.shape[2] in [3, 4]:
        image_bgr = image[..., 0:3][..., ::-1]

    win_name = "imgaug-default-window"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, image_bgr)
    cv2.waitKey(0)
    cv2.destroyWindow(win_name)

# Number of batches and batch size for this example
nb_batches = 10
batch_size = 32

# Example augmentation sequence to run in the background
augseq = iaa.Sequential([
    iaa.Fliplr(0.5),
    #iaa.CoarseDropout(p=0.1, size_percent=0.1),
    #iaa.GaussianBlur(sigma=(0, 3.0))
    iaa.Multiply((0.5, 1.5), per_channel=0.5),
    iaa.Scale((0.5, 1.0))
])

# For simplicity, we use the same image here many times
# images = data.load(myfiles[0]) # data.astronaut()
#astronaut = ia.imresize_single_image(astronaut, (64, 64))

# Make batches out of the example image (here: 10 batches, each 32 times
# the example image)
'''
batches = []
for _ in range(nb_batches):
    batches.append(
        np.array(
            [astronaut for _ in range(batch_size)],
            dtype=np.uint8
        )
    )

# Show the augmented images.
# Note that augment_batches() returns a generator.
for images_aug in augseq.augment_batches(batches, background=True):
    imshow(ia.draw_grid(images_aug, cols=8))
'''
images = cv2.imread(myfiles[0]) # read images

images_aug = augseq.augment_image(images) # do the process of augmentation

# resize to show
print("Showing images:" + str(images.shape) + str(images_aug.shape))

images = cv2.resize(images, (400,400))
images_aug = cv2.resize(images_aug, (400,400))
# show origin and changed on together
imshow(np.hstack((images, images_aug)))
