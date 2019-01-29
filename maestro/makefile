
CC=gcc
CFLAGS=-Wall -I.  
DEPS = *.h
OBJ = maestro.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

maestro: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) 

.PHONY: clean

clean:
	rm -rf *.o maestro
