
CC=gcc
CFLAGS=-Wall -I. 
LDFLAGS= `pkg-config --libs opencv` 
CXXFLAGS= `pkg-config --cflags opencv` -lstdc++ -lm -std=c++0x
DEPS = *.h *.hpp
OBJ = fhog.o kcftracker.o runtracker.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

tracker: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(CXXFLAGS)

.PHONY: clean

clean:
	rm -rf *.o tracker
