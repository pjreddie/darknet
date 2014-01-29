CC=gcc
COMMON=-Wall `pkg-config --cflags opencv`
CFLAGS= $(COMMON) -O3 -ffast-math -flto
UNAME = $(shell uname)
ifeq ($(UNAME), Darwin)
COMMON += -isystem /usr/local/Cellar/opencv/2.4.6.1/include/opencv -isystem /usr/local/Cellar/opencv/2.4.6.1/include
else
CFLAGS += -march=native
endif
#CFLAGS= $(COMMON) -O0 -g 
LDFLAGS=`pkg-config --libs opencv` -lm
VPATH=./src/
EXEC=cnn

OBJ=network.o image.o tests.o connected_layer.o maxpool_layer.o activations.o list.o option_list.o parser.o utils.o data.o matrix.o softmax_layer.o mini_blas.o convolutional_layer.o

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

%.o: %.c 
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -rf $(OBJ) $(EXEC)

