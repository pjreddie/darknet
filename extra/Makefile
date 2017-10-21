DEBUG=1
GPU=1

VPATH=./src
SLIB=libdarknet_lowp.so
ALIB=libdarknet_lowp.a
OBJDIR=./obj/

CC=gcc
AR=ar
ARFLAGS=rcs
OPTS=-Ofast
LDFLAGS= -lm -pthread 
COMMON= -I../include/ -I../src/ -I./src -I./include
CFLAGS=-Wall -Wno-unknown-pragmas -Wfatal-errors -fPIC
CFLAGS+= -Werror=implicit-function-declaration
 

ifeq ($(GPU), 1)
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

OBJ= mem_manager.o lowp_primitives.o lowp_model.o lowp_examples.o extra_utils.o lowp_gemm.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile ../include/darknet.h

all: obj  results $(SLIB) $(ALIB)

$(ALIB): $(OBJS)
	$(AR) $(ARFLAGS) $@ $^

$(SLIB): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir -p obj

.PHONY: clean

clean:
	rm -rf $(OBJS) $(SLIB) $(ALIB)

