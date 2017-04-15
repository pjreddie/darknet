## results
ret=$(./darknet detector test cfg/combine9k.data cfg/yolo9000.cfg yolo9000.weights $1)
echo $ret
