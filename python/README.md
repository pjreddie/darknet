# Darknet Python Wrapper
## predict.py
#### - predict a given image and display output.
    python3 predict.py image <output_file_path>


## predict_one.py
#### - split an image into 4 slices and predict on each image, then merge back to one. output will be store in output and 4 slices will be store in temp
    python3 predict_one.py image

## predict_silding_windows.py
    python3 predict_silding_windows.py image
#### - image tiling (Avoid resizing)
#### - when dealing with big images, darknet will resize the image into the network input size; this process downsample the image and might even change the aspect ratio which will destroy features.
#### - For image tiling, we simply have a small window that have the same size as the network input size, slice through the entire image and predict on every single slice.
