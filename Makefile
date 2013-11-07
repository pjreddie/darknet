CC=gcc
CFLAGS=-Wall `pkg-config --cflags opencv` -O3 -flto -ffast-math
CFLAGS=-Wall `pkg-config --cflags opencv` -O0 -g
LDFLAGS=`pkg-config --libs opencv` -lm
VPATH=./src/

OBJ=network.o image.o tests.o convolutional_layer.o connected_layer.o maxpool_layer.o activations.o list.o option_list.o parser.o utils.o

all: cnn

cnn: $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

%.o: %.c 
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -rf $(OBJ) cnn

