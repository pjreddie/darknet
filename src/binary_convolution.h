#ifndef AI2_BINARY_CONVOLUTION_H
#define AI2_BINARY_CONVOLUTION_H

/** @file binary_convolution.h
 *  @brief Routines related for approximating convolutions using binary operations
 *      
 *  @author Carlo C. del Mundo (carlom)
 *  @date 05/23/2016
 */

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <assert.h>
#include <limits.h>
#include <tgmath.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include "common.h"

typedef struct {
    int batch;   // number of filter batches
    int c;       // channels, z
    int h;       // height, y
    int w;       // width, x
    int stride;
    int pad;

    int px;     // padded x (use this for striding in padded input and output arrays)
    int py;     // padded y (use this for striding in padded input and output arrays)
    int wx;
    int wy;

    float *input;       // input values
    BINARY_WORD *binary_input;

    float *weights;     // weight or filter values
    BINARY_WORD *binary_weights;

    float *output;      // output values
    BINARY_WORD *binary_output;

    float *alpha;       // we assume alpha is calculated at the beginning of initialization
    float *beta;        // we assume beta is given to us
    float *new_beta;    // we calculate the new beta for the next layer

    struct ai2_bin_conv_layer *next;
} ai2_bin_conv_layer;

/** @brief Performs a binary convolution using XNOR and POPCOUNT between input and weights
 *
 *  @param output A 2D real-valued plane to store the outputs
 *  @param input A 2D binary-valued plane that holds the inputs
 *  @param weights A 2D binary-valued plane that holds the weights 
 *  @param ix   the input's x dimension 
 *  @param iy   the input's y dimensions
 *  @param wx   the weight's x dimension
 *  @param wy   the weight's y dimension
 *  @param pad  the amount of padding applied to input. (ix+2*pad is the x dimension of the input
 *  @param stride NOP. TODO: implement stride. the stride between sliding windows
 *  @return the count of all overlapping set bits between the two volumes.
 */
void ai2_bin_conv2D(float *output, const BINARY_WORD *input, const BINARY_WORD *weights, int ix, int iy, int wx, int wy, int pad, int stride);

/** @brief Performs a binary dot product (XNOR and POPCOUNT) for two equal sized volumes.
 *
 *  @param a A 3D binary tensor
 *  @param b A 3D binary tensor 
 *  @param vdim the dimensionality of the data. Note: we pack 32 elements in the Z element.
 *  @return the count of all overlapping set bits between the two volumes.
 */
int ai2_bin_dp(BINARY_WORD *a, BINARY_WORD *b, dim3 vdim);

/** @brief Calculates the alpha plane given an alpha volume. 
 *
 *  Each point in the yz alpha plane
 *  is the average sum of the absolute value of all elements in the z-direction.
 *
 * Pre-conditions: 
 *                  alpha_volume is an array of size x*y*z.
 *                  alpha_plane is an array of size x*y.
 *                  alpha_volume (x,y,z) is transposed to (z,x,y).
 *
 *  @param alpha_plane The 2D real-valued output plane
 *  @param alpha_volume The 3D real-valued output volume
 *  @param vdim the dimensionality of alpha_volume.
 */
void ai2_calc_alpha(float *alpha_plane, float *alpha_volume, dim3 vdim);

/** @brief Wrapper function for generating the beta scaling factor */
void ai2_calc_beta(float *beta_plane, float *beta_volume, dim3 vdim); 

/** @brief Set the bit in a binary word */
void ai2_bitset(BINARY_WORD *bword, unsigned int position);

/** @brief Checks that the bit is set in a binary word */
int ai2_is_set(BINARY_WORD bword, unsigned int position) ;

/** @brief Converts a 3D float tensor into a 3D binary tensor.
 *
 *  The value of the ith element in the binary tensor is the sign
 *  of the ith element in the floating tensor.
 *
 *  @param binary_vol the binary tensor
 *  @param real_vol the real tensor
 *  @param vdim the size of the 3D tensor
 */
void ai2_flt_to_bin(BINARY_WORD *binary_vol, float *real_vol, dim3 vdim) ;

/** @brief Converts a 3D binary tensor into a 3D float tensor.
 *
 * The ith float element will be '1' if the ith binary element is '1'.
 * Otherwise, the float element will be '-1'.
 *
 *  @param real_vol the output real tensor
 *  @param binary_vol the input binary tensor
 *  @param vdim the dimension of both binary_vol and real_vol
 */
void ai2_bin_to_flt(float *real_vol, BINARY_WORD *binary_vol, dim3 vdim); 

/** @brief Performs a pointwise matrix multication between two 2D tensors
 *  @param output A 2D real-valued plane to store the outputs
 *  @param input A 2D binary-valued plane that holds the inputs
 *  @param N the number of elements between the arrays
 */
void ai2_pointwise_mul_mm(float *output, const float *input, int N);

/** @brief Performs a tiled pointwise matrix multiplication between two 2D tensors
 *  
 *  Pre-conditions: wx < ix, and wy < iy
 *
 *  @param output A 2D real-valued plane of size ix, iy
 *  @param alpha A 2D binary-valued plane of size wx, wy
 *  @param ix   the output's x dimension 
 *  @param iy   the output's y dimensions
 *  @param wx   the alpha's x dimension
 *  @param wy   the alpha's y dimension
 *  @param pad  how many cells are padded, adds 2*pad to the borders of the image 
 */
void ai2_pointwise_mul_mm_2d(float *output, const float *alpha, int ix, int iy, int wx, int wy, int pad);

// --------------------------------------
//  SETTER FUNCTIONS
// --------------------------------------
/** @brief Safe function to set the float input of a conv_layer
 */
void ai2_setFltInput(ai2_bin_conv_layer *layer, float *new_input);

/** @brief Safe function to set the binary input of a conv_layer
 */
void ai2_setBinInput(ai2_bin_conv_layer *layer, BINARY_WORD *new_input);

/** @brief Safe function to set the binary weights of a conv_layer
 */
void ai2_setFltWeights(ai2_bin_conv_layer *layer, float *new_weights);

/** @brief Safe function to set the binary weights of a conv_layer
 */
void ai2_setBinWeights(ai2_bin_conv_layer *layer, BINARY_WORD *new_weights);

/** @brief Safe function to set the binary outputs of a conv_layer
 */
void ai2_setFltOutput(ai2_bin_conv_layer *layer, float *new_output);

/** @brief Safe function to set the binary outputs of a conv_layer
 */
void ai2_setBinOutput(ai2_bin_conv_layer *layer, BINARY_WORD *new_output);

/** @brief Safe function to set the alpha of a conv_layer
 */
void ai2_setFltAlpha(ai2_bin_conv_layer *layer, float *new_alpha);

/** @brief Safe function to set the beta of a conv_layer
 */
void ai2_setFltBeta(ai2_bin_conv_layer *layer, float *new_beta);

/** @brief Safe function to set the new_beta of a conv_layer
 */
void ai2_setFltNewBeta(ai2_bin_conv_layer *layer, float *new_new_beta);

// --------------------------------------
//  GETTER FUNCTIONS
// --------------------------------------
/** @brief Safe function to get the float outputs of a conv_layer
 */
float * ai2_getFltOutput(ai2_bin_conv_layer *layer);

/** @brief 3D tranpose from (x,y,z) to (z,y,x)
 *  @return a new pointer with the transposed matrix
 */
void ai2_transpose3D(float *data, dim3 d);

/** @brief Checks if a float is a whole number (e.g., an int)
 */
int ai2_isFloatWhole(float f);

/* @brief Allocates all memory objects in an ai2_bin_conv_layer
 * b - batches (number of filter batches)
 * c - input channels
 * ix - input width
 * iy - input height
 * wx - weight/filter width
 * wy - weight/filter height
 * s - stride between sliding windows
 * pad - the amount of padding
 */
ai2_bin_conv_layer ai2_make_bin_conv_layer(int b, int c, int ix, int iy, int wx, int wy, int s, int pad);

/* @brief Safe deallocation of  all memory objects in an ai2_bin_conv_layer
 */
void ai2_free_bin_conv_layer(ai2_bin_conv_layer *layer);

/* @brief Given real-valued filter data and a conv layer, performs a forward pass
 */
void ai2_bin_forward(ai2_bin_conv_layer *layer);

#endif
