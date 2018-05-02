
CC=gcc

LDFLAGS= `pkg-config --libs opencv` 

CXXFLAGS= -Wall `pkg-config --cflags opencv` -lstdc++ -lm -std=c++0x

DEPS = *.h *.hpp

OBJ = opencvtrackers.o


%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< 

opencvtrackers.bin: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) $(CXXFLAGS) 

.PHONY: clean

clean:
	rm -rf *.o *.bin
