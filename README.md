![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

#Darknet#
Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

For more information see the [Darknet project website](http://pjreddie.com/darknet).

For questions or issues please use the [Google Group](https://groups.google.com/forum/#!forum/darknet).

# Added command

* Supports a new command: `darknet stream` for batch processing. Default for `-in` is `stdin`, default for `-out` is `stdout`

Running `darknet stream cfg/yolo.cfg yolo.weights -thresh 0.15 -in images.txt -out images.result.txt`

where `images.txt` has

```data/dog.jpg
data/eagle.jpg
data/giraffe.jpg
data/horses.jpg
data/person.jpg
data/scream.jpg
```

produces `images.result.txt` with

```image,category,prob,xmin,ymin,xmax,ymax
"data/dog.jpg",car,0.542268,0.594182,0.135254,0.876864,0.302098
"data/dog.jpg",truck,0.261111,0.594182,0.135254,0.876864,0.302098
"data/dog.jpg",bicycle,0.218625,0.066853,0.160006,0.348404,0.675097
"data/dog.jpg",bicycle,0.509336,0.129278,0.215466,0.725198,0.763334
"data/dog.jpg",cat,0.346415,0.151465,0.367006,0.417390,0.938185
"data/dog.jpg",dog,0.558766,0.151465,0.367006,0.417390,0.938185
"data/eagle.jpg",bird,0.893585,0.147595,0.153328,0.807179,0.892081
"data/giraffe.jpg",giraffe,0.961540,0.355689,0.034908,0.836137,0.919772
"data/giraffe.jpg",zebra,0.865405,0.599002,0.396530,0.836604,0.923422
"data/horses.jpg",horse,0.493722,0.007934,0.366037,0.197422,0.505652
"data/horses.jpg",horse,0.231258,0.000000,0.389534,0.125733,0.602529
"data/horses.jpg",horse,0.813255,0.287581,0.349805,0.592668,0.700121
"data/horses.jpg",horse,0.885061,0.000000,0.398561,0.414195,0.800371
"data/horses.jpg",horse,0.818568,0.550553,0.414785,0.769648,0.693348
"data/person.jpg",person,0.857074,0.289393,0.227537,0.429284,0.883788
"data/person.jpg",horse,0.886999,0.631595,0.315156,0.946103,0.804331
"data/person.jpg",dog,0.746660,0.105581,0.609787,0.318202,0.831716
"data/scream.jpg",person,0.357746,0.354842,0.480332,0.646520,1.005375
```

