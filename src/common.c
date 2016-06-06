#include "common.h" 

// Returns the time in ms
double getElapsedTime(Timer *timer) {
    // Calculate time it took in seconds
    double accum_ms = ( timer->requestEnd.tv_sec - timer->requestStart.tv_sec )
      + ( timer->requestEnd.tv_nsec - timer->requestStart.tv_nsec )
      / 1e6;
    return accum_ms;
}

void start_timer(Timer *timer) {
    clock_gettime(CLOCK_MONOTONIC_RAW, &(timer->requestStart));
}

void stop_timer(Timer *timer) {
    clock_gettime(CLOCK_MONOTONIC_RAW, &(timer->requestEnd));
}


BINARY_WORD * mallocBinaryVolume(dim3 vol) {
    return (BINARY_WORD *) malloc (vol.x * vol.y * vol.z / BITS_PER_BINARY_WORD * sizeof(BINARY_WORD));
}

float * mallocFloatVolume(dim3 vol) {
    return (float *) malloc (vol.x * vol.y * vol.z * sizeof(float));
}

// Returns the size (in bytes) of a binary array with dimensions stored in conv_args
double getSizeBytesBinaryArray(dim3 conv_args) {
    return conv_args.x * conv_args.y * conv_args.z * sizeof(BINARY_WORD) / (BITS_PER_BINARY_WORD);
}


ConvolutionArgs initArgs(size_t ix, size_t iy, size_t iz, size_t wx, size_t wy, size_t wz) {
    ConvolutionArgs conv_args;
    // Input Volume
    conv_args.input.x = ix;    // x == y for a square face
    conv_args.input.y = iy;
    conv_args.input.z = iz;
    conv_args.weights.x = wx; // x == y for square face
    conv_args.weights.y = wy;
    conv_args.weights.z = wz;

    // <!-- DO NOT MODIFY -->
    // Intermediate Volumes
    conv_args.alpha_plane.x = conv_args.weights.x;
    conv_args.alpha_plane.y = conv_args.weights.y;
    conv_args.alpha_plane.z = 1;

    conv_args.beta_plane.x = 1;
    conv_args.beta_plane.y = conv_args.input.y;
    conv_args.beta_plane.z = conv_args.input.z;

    conv_args.gamma_plane.x = conv_args.input.x * conv_args.weights.x;
    conv_args.gamma_plane.y = conv_args.input.y * conv_args.weights.y;
    conv_args.gamma_plane.z = 1;

    conv_args.zeta_plane.x = conv_args.gamma_plane.x;
    conv_args.zeta_plane.y = conv_args.gamma_plane.y;
    conv_args.zeta_plane.z = 1;

    // Output Volume
    conv_args.output.x = conv_args.input.x;
    conv_args.output.y = conv_args.input.y;
    conv_args.output.z = 1; // Output should be a 2D plane

    // Verify dimensions
    //assert(conv_args.weights.x % 32 == 0);  // must be divisble by 32 for efficient alignment to unsigned 32-bit ints
//    assert(conv_args.weights.y % 32 == 0);  // must be divisble by 32 for efficient alignment to unsigned 32-bit ints
	assert(conv_args.weights.z % 32 == 0);  // must be divisble by 32 for efficient alignment to unsigned 32-bit ints
    //assert(conv_args.input.x % 32 == 0);    // must be divisble by 32 for efficient alignment to unsigned 32-bit ints
//    assert(conv_args.input.y % 32 == 0);    // must be divisble by 32 for efficient alignment to unsigned 32-bit ints
    assert(conv_args.input.z % 32 == 0);    // must be divisble by 32 for efficient alignment to unsigned 32-bit ints
    assert(conv_args.weights.x <= conv_args.input.x);
    assert(conv_args.weights.y <= conv_args.input.y);
    assert(conv_args.weights.z <= conv_args.input.z);
    // <!-- DO NOT MODIFY -->

    return conv_args;
}
