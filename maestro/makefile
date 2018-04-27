
CC=gcc
CFLAGS=-Wall -I.  
DEPS = *.h *.hpp
OBJ = test.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

maestro: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) 

.PHONY: clean

clean:
	rm -rf *.o
