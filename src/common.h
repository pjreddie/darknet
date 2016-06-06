#ifndef AI2_COMMON_H 
#define AI2_COMMON_H

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <assert.h>
#include <limits.h>
#include <tgmath.h>
#include <unistd.h>
#include <stdint.h>
//#include <gperftools/profiler.h>
#include <sys/time.h>

typedef uint32_t BINARY_WORD;
#define BITS_PER_BINARY_WORD (sizeof(BINARY_WORD) * CHAR_BIT)

typedef struct{
    struct timespec requestStart;
    struct timespec requestEnd;
} Timer;

typedef struct {
	size_t x;
	size_t y;
	size_t z;
} dim3;

typedef struct {
	dim3 weights;
	dim3 input;
    dim3 output;
    dim3 alpha_plane;
    dim3 beta_plane;
    dim3 gamma_plane;
    dim3 zeta_plane;
} ConvolutionArgs;

// Timer stuff
double getElapsedTime(Timer *timer); // Returns the time in ms
void start_timer(Timer *timer);
void stop_timer(Timer *timer);

BINARY_WORD * mallocBinaryVolume(dim3 vol);
float * mallocFloatVolume(dim3 vol);
ConvolutionArgs initArgs(size_t ix, size_t iy, size_t iz, size_t wx, size_t wy, size_t wz);
double getSizeBytesBinaryArray(dim3 conv_args);

#endif
