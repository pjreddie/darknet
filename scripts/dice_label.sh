mkdir -p images
mkdir -p images/orig
mkdir -p images/train
mkdir -p images/val

ffmpeg -i Face1.mp4 images/orig/face1_%6d.jpg
ffmpeg -i Face2.mp4 images/orig/face2_%6d.jpg
ffmpeg -i Face3.mp4 images/orig/face3_%6d.jpg
ffmpeg -i Face4.mp4 images/orig/face4_%6d.jpg
ffmpeg -i Face5.mp4 images/orig/face5_%6d.jpg
ffmpeg -i Face6.mp4 images/orig/face6_%6d.jpg

mogrify -resize 100x100^ -gravity center -crop 100x100+0+0 +repage images/orig/*

ls images/orig/* | shuf | head -n 1000 | xargs mv -t images/val
mv images/orig/* images/train

find `pwd`/images/train > dice.train.list -name \*.jpg
find `pwd`/images/val > dice.val.list -name \*.jpg

