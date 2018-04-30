

DEBUG=1

CC=gcc
LDFLAGS= `pkg-config --libs opencv` 
CFLAGS=-Wall -I. 
CXXFLAGS= `pkg-config --cflags opencv` -lstdc++ -lm -std=c++0x
DEPS = *.h *.hpp
OBJ = opencvtrackers.o

ifeq ($(DEBUG), 1) 
CFLAGS+= -DDEBUG
CXXFLAGS+= -DDEBUG
#OPTS=-O0 -g
endif

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

opencvtrackers: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) $(CXXFLAGS) 

.PHONY: clean

clean:
	rm -rf *.o opencvtrackers
