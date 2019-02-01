# Yolo Train System

This repo assists the training of YOLO neural networks from the data generation stage to the training stage.

## Data Generation (data_gen)

This directory contains all the needed scripts and tools to quickly go from a bag file to a zip file containing labeled jpg images and descriptors of the dataset. See the data_gen/README for more info on the process.

## Model Generation (model_gen)

This directory contains all the needed scripts to extract and combine zip files containing labeled jpg images and descriptors and feed them into darknet to train a model. See model_gen/README for more info on the process.
