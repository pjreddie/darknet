## images directory
path_folder="$1"

## declare an array variable
declare -a arr=( $(ls $path_folder*$2) )

## results
result=""

## now loop through the above array
for i in "${arr[@]}"
do
   ret=$(./darknet detector test cfg/combine9k.data cfg/yolo9000.cfg yolo9000.weights $i)
   result="$result$ret;"
done

echo $result
