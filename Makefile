CC=gcc
COMMON=-Wall `pkg-config --cflags opencv` -isystem /usr/local/Cellar/opencv/2.4.6.1/include/opencv -isystem /usr/local/Cellar/opencv/2.4.6.1/include
CFLAGS= $(COMMON) -O3 -ffast-math -flto
#CFLAGS= $(COMMON) -O0 -g 
LDFLAGS=`pkg-config --libs opencv` -lm
VPATH=./src/

OBJ=network.o image.o tests.o convolutional_layer.o connected_layer.o maxpool_layer.o activations.o list.o option_list.o parser.o utils.o data.o matrix.o softmax_layer.o

all: cnn

cnn: $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

%.o: %.c 
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -rf $(OBJ) cnn

