
darknet.exe detector test cfg/openimages.data cfg/yolov3-openimages.cfg yolov3-openimages.weights -i 0 -thresh 0.25 dog.jpg -ext_output



darknet.exe detector demo cfg/openimages.data cfg/yolov3-openimages.cfg yolov3-openimages.weights -i 0 -thresh 0.25 street1k.mp4 -ext_output

pause