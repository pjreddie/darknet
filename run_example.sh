#!/bin/sh
echo "this is example how to run validation on multiple images with loading the network only once"
echo "we use -out argument to set the destination directory for putting results"
echo "check additional field in .data file called detector_log - this will be used as name for validation log"
echo "you need to point the darknet detector test to valid paths to cfg and weights files"
cat example_validation_files.txt | ./darknet detector test run_example.data ../aeolus/reco-Nov21-TL-yolo.cfg ../aeolus/reco-Nov21-TL-yolo_final.weights -thresh 0.5 -out "results/run_example" 

