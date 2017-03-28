# Darknet em Português 

   ![Darknet Logo](http://pjreddie.com/media/files/darknet-black-small.png)

> Este é um fork independente. O trabalho original está em [github/pjreddie](https://github.com/pjreddie/darknet)

Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

Para mais informações, consulte o site oficial [Darknet website](http://pjreddie.com/darknet).

Para questões ou erro use o [Google Group](https://groups.google.com/forum/#!forum/darknet).


## Instalação

#### Requisitos

* **CUDA**: https://developer.nvidia.com/cuda-downloads
   - Funciona sem CUDA, mas o desempenho é muito mais lento, porque só utiliza 1 core do CPU. 
   - Para verificar qual a versão do CUDA instalado `nvcc --version`
   - Para verificar qual a gráfica instalada: `nvidia-smi`
* **OpenCV 2.4.9**: http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html
   - Não testado com OpenCV 3
* **Cmake** 


1) Instalar Darknet ([Instruções](https://pjreddie.com/darknet/install/))

`git clone https://github.com/pjreddie/darknet`

> NOTA: Deve-se compilar o depois de fazer as alterações dos directórios nos ficheiros.

2) Fazer train.txt

3) Criar o obj.name

4) Criar o obj.data

5) Alterar a Makefile.
```
GPU=1
CUDNN=0
OPENCV=1
DEBUG=0
```
Ver também se a arquitetura da gráfica é a correta. ([Compute Capability](https://developer.nvidia.com/cuda-gpus))

`LN.6 ARCH=  -gencode arch=compute_52,code=compute_52`

Para o caso da NVIDIA Quadro K2000 GK107GL (LSA/ISEP) - Compute Capability 3.0

`ARCH=  -gencode arch=compute_30,code=compute_35`

6) Alterar o detector.c / yolo.c - Diretoriio do backup e train.txt

alterar as linhas de nº x de backups

7) Compilar o Darknet já com os directórios corretos.
```
cd darknet
make
```

8 ) alterar o yolo.cfg

## How to improve object detection:

 Before training:

set flag `random=1` in your `.cfg-file` - it will increase precision by training Yolo for different resolutions

 After training - for detection:

Increase network-resolution by set in your `.cfg-file` (height=608 and width=608) or (height=832 and width=832) or (any value multiple of 32) - this increases the precision and makes it possible to detect small objects.

you do not need to train the network again, just use `.weights-file` already trained for 416x416 resolution

if `error Out of memory` occurs then in `.cfg-file` you should increase `subdivisions=16, 32 or 64`.


## Treinar

## Testar

`./darknet detector test dataObjTeste_D/obj.data cfg/yolo-voc.cfg backup/tiny-yolo-voc_1730.weights dataObjTeste_D/images/1.jpg -thresh 0`

## Fontes
- [Darknet website](http://pjreddie.com/darknet) - Página oficial do projeto
- [Google Group](https://groups.google.com/forum/#!forum/darknet) - Forum de dúvidas
- [Github/AlexeyAB](https://github.com/AlexeyAB/darknet) - Versão para Windows
