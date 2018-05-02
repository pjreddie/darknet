
CC=gcc

LDFLAGS= `pkg-config --libs opencv` -L/media/elab/sdd/caffe/build/install/lib -lcaffe-nv -lproto \
	-lboost_system -lboost_filesystem -lboost_regex -lglog

CXXFLAGS= -Wall `pkg-config --cflags opencv` -lstdc++ -lm -std=c++0x -I/media/elab/sdd/caffe/build/install/include/ \
	-I/usr/local/cuda/include/ 

DEPS = *.h *.hpp $(wildcard kcf/*.h) $(wildcard goturn/*/*.h) $(wildcard /media/elab/sdd/caffe/build/install/include/*/*.hpp)

OBJ = kcf/fhog.o kcf/kcftracker.o goturn/network/regressor_base.o goturn/network/regressor.o \
	goturn/helper/bounding_box.o goturn/helper/helper.o goturn/helper/image_proc.o \
	goturn/helper/high_res_timer.o goturn/tracker/tracker.o trackerscompare.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< 

all: makekcf makegoturn trackerscompare.bin

trackerscompare.bin: $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) $(CXXFLAGS) 

makekcf:
	cd kcf && make -j4

makegoturn:
	cd goturn && make -j8

.PHONY: clean

clean:
	rm -rf */*/*.o */*.o *.o */*.bin *.bin
