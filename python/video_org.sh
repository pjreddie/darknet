i=0;
for file in $1/*
do
 # do something on $file
 echo "$file"
 python3 predict.py $file $2/"$i.jpg"
 i=$((i+1));
done

echo "input dir: $1"
echo "output dir: $2"
