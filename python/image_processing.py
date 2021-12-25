from PIL import Image
import cv2


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def display_obj_count(img=None, count=0, path=None):
    # Read the image
    if path!=None:
        img = cv2.imread(path)
    img_h, img_w, _ = img.shape

    thickness = 3
    fontScale = 1
    color = (255, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    location = (int(img_w*0.95), int(img_h*0.95))
    cv2.putText(img, str(count), location, font, fontScale, color, thickness, cv2.LINE_AA)
    return img
