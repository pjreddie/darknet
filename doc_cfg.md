# Ficheiro .cfg

Contem a estrutura da rede (número de camadas, tipo de camadas, parametros hyperparameters)

## Parametros da rede

```
[net]
batch=1
subdivisions=1
```

```
width=416   | Resizing da imagens para estas dimensões
height=416  |
channels=3  | 3 para RGB
```

```
momentum=0.9 
decay=0.0005
angle=`
```

```
saturation = 1.5    | Alteração de cor e luz das imagens
exposure = 1.5      |
hue=.1              |
```

```
learning_rate=0.001     | The learning rate is a parameter that determines how much an updating step influences the current value of the weights.
max_batches = 120000    | Número max de iterações
policy=steps            |
steps=-1,100,80000,100000   |
scales=.1,10,.1,.1      | Escala das imagens ??
``` 

## Tipos de camadas

A rede é essencialmente constituida por dois tipos de camadas, Convulutional e Max Pool.

As Conculutional podem ser fully conected ou não. 

### Convolução

```
[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[route]
layers=-9

[reorg]
stride=2

[route]
layers=-1,-3
```

### Max Pool
Camada para reduzir a informação. Faz um pooling aos valores mais altos da matriz.
```
[maxpool]
size=2      | Matrix
stride=2    | A matriz move-se de 2 me dois
```

## Outros parametros

```
[region]
anchors = 0.738768,0.874946,  2.42204,2.65704,  4.30971,7.04493,  10.246,4.59428,  12.6868,11.8741
bias_match=1
classes=80  | Numero de classes a detetar
coords=4    |
num=5       | means each cell predict 5 objects
softmax=1   | It seems means use softmax.
jitter=.2   | means in load picture random cut 0.2*width 0.2*height.
rescore=1

object_scale=5
noobject_scale=1
class_scale=1
coord_scale=1

absolute=1
thresh = .6
random=0
```
