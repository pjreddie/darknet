#include "binary_convolution.h"

int ai2_bin_dp(BINARY_WORD *a, BINARY_WORD *b, dim3 vdim) {     // TODO unroll
    int accumulator = 0;
    for (int z = 0; z < vdim.z / BITS_PER_BINARY_WORD; z++) {
        for (int y = 0; y < vdim.y; y++) {
            for (int x = 0; x < vdim.x; x++) {
                int idx = z*vdim.y*vdim.x + y*vdim.x + x;
                accumulator += __builtin_popcount(~(a[idx] ^ b[idx]));   // count the XNOR of the two bit vectors
            }
        }
    }

    return accumulator;
}

/** 
 * Pre-conditions: 
 *                  alpha_volume is an array of size x*y*z.
 *                  alpha_plane is an array of size x*y.
 *                  alpha_volume (x,y,z) is transposed to (z,x,y).
 */
void ai2_calc_alpha(float *alpha_plane, float *alpha_volume, dim3 vdim) {
    for (int y = 0; y < vdim.y; ++y) {
        for (int x = 0; x < vdim.x; ++x) {
            int out = y * vdim.x + x;
            double accum = 0.0;
            for (int z = 0; z < vdim.z; ++z) {
                accum += alpha_volume[out * vdim.z + z];
            }

            alpha_plane[out] = accum / vdim.z;
        }
    }
}

/** @brief Wrapper function for generating the beta scaling factor */
void ai2_calc_beta(float *beta_plane, float *beta_volume, dim3 vdim) {
    ai2_calc_alpha(beta_plane, beta_volume, vdim);
}

/** @brief Set the bit in a binary word */
void ai2_bitset(BINARY_WORD *bword, unsigned int position) {
    BINARY_WORD mask = (1 << position);
    *bword = *bword | mask;
}

/** @brief Checks that the bit is set in a binary word */
int ai2_is_set(BINARY_WORD bword, unsigned int position) {
    unsigned int position_complement = (BITS_PER_BINARY_WORD - 1) - position;   // number of leading bits before the bit position of interest
    bword = (bword << position_complement);                                     // zero out leading bits
    bword = (bword >> (BITS_PER_BINARY_WORD - 1));                              // shift bit position of interest to the 0th position
    return (bword & 0x1);                                                       // test if bit position of interest is set
}

void ai2_flt_to_bin(BINARY_WORD *binary_vol, float *real_vol, dim3 dim) {
    ai2_transpose3D(real_vol, dim); // (x,y,z) -> (z,x,y)

    int sz = dim.x * dim.y * dim.z;
    for (int i = 0; i < sz; i += BITS_PER_BINARY_WORD) {
        BINARY_WORD tmp = 0x00000000;
        for (int x = 0; x < BITS_PER_BINARY_WORD; ++x) {
            int waddr = x + i;
            if (signbit(real_vol[waddr]) == 0)
                ai2_bitset(&tmp, (BITS_PER_BINARY_WORD - 1) - x);
        }
        binary_vol[i / BITS_PER_BINARY_WORD] = tmp;
    }
}

void ai2_bin_to_flt(float *real_vol, BINARY_WORD *binary_vol, dim3 dim) {   // TODO unit tests
    for (int z = 0; z < dim.z; z++) {
        for (int y = 0; y < dim.y; y++) {
            for (int x = 0; x < dim.x / BITS_PER_BINARY_WORD; x++) {    // TODO boundary checks, for uneven input
                BINARY_WORD word = binary_vol[z*dim.y*dim.x + y*dim.x + x];
                for (int t = 0; t < BITS_PER_BINARY_WORD; ++t) {
                    int oidx = z*dim.y*dim.x + y*dim.x + x * BITS_PER_BINARY_WORD + t;
                    if (ai2_is_set(word, t))
                        real_vol[oidx] = 1.f;
                    else
                        real_vol[oidx] = -1.f;
                }
            }
        }
    }

    // Transpose channels back to output
    ai2_transpose3D(real_vol, dim); // (z,y,x) -> (x,y,z)
}

/* @brief: input is padded.
 */
void ai2_bin_conv2D(float *output, const BINARY_WORD *input, const BINARY_WORD *weights, int ix, int iy, int wx, int wy, int pad, int stride) {

    int r, rd, c, cd;
    int wx_2 = wx / 2;
    int wy_2 = wy / 2;

    // Indexing for output pixels. x = [wx_2, ix + wx_2 - 1], y = [wy_2, iy + wy_2 - 1]
    int sx = pad;               // start x
    int ex = ix + pad - 1;      // end x
    int sy = pad;               // start y
    int ey = iy + pad - 1;      // end y

    // Indexing for weights
    int wsx, wex, wsy, wey;
    if (wx % 2 == 1) {  // odd weights
        wsx = -wx_2; wex = wx_2 + 1;
        wsy = -wy_2; wey = wy_2 + 1;    
    }
    else {
        wsx = -wx_2; wex = wx_2;
        wsy = -wy_2; wey = wy_2;    
    }

    int px = ix + 2*pad;
    //int py = iy + 2*pad;

    for (r = sy; r <= ey; ++r) {
        for (c = sx; c <= ex; ++c) {
            int accumulator = 0;
            for (rd = wsy; rd < wey; ++rd) {
                for (cd = wsx; cd < wex; ++cd) {
                    int iidx = (r+rd)*px + (c+cd);
                    BINARY_WORD pixel = input[iidx];
                    //BINARY_WORD pixel = 0xFFFFFFFF;
                    //BINARY_WORD weight = 0xFFFFFFFF;
                    int widx = (rd + wy_2)*wx + (cd+wx_2);
                    BINARY_WORD weight = weights[widx];
                    accumulator += __builtin_popcount(~(pixel ^ weight));
                }
            }

            // Padded space
            int oidx = r*px + c;
            output[oidx] += (float) accumulator;
        }
    }

    //for (r = sy; r <= ey; ++r) {
    //  for (c = sx; c <= ex; ++c) {
    //      int accumulator = 0;
    //      for (rd = -wy_2; rd < wy_2; ++rd) {
    //          for (cd = -wx_2; cd < wx_2; ++cd) {
    //              int iidx = (r+rd)*px + (c+cd);
    //              BINARY_WORD pixel = input[iidx];
    //              //BINARY_WORD pixel = 0xFFFFFFFF;
    //              //BINARY_WORD weight = 0xFFFFFFFF;
    //              int widx = (rd + wy_2)*wx + (cd+wx_2);
    //              BINARY_WORD weight = weights[widx];
    //              accumulator += __builtin_popcount(~(pixel ^ weight));
    //          }
    //      }

    //      // Padded space
    //      int oidx = r*px + c;
    //      output[oidx] += (float) accumulator;
    //  }
    //}
    
    //ai2_bin_conv_within_boundary(output, input, weights, ix, iy, wx, wy, stride);
    //ai2_bin_conv_borders(output, input, weights, ix, iy, wx, wy, stride);
}

void ai2_pointwise_mul_mm(float *output, const float *input, int N) {
    int i = 0;

    while (i + 8 <= N) {
        output[i+0] *= input[i+0];
        output[i+1] *= input[i+1];
        output[i+2] *= input[i+2];
        output[i+3] *= input[i+3];
        output[i+4] *= input[i+4];
        output[i+5] *= input[i+5];
        output[i+6] *= input[i+6];
        output[i+7] *= input[i+7];

        i += 8;
    }

    while (++i < N) // Finish iteration that's leftover (e.g., last batch not divisible by 8 exactly)
         output[i] *= input[i];
}

/** @brief Performs a tiled pointwise matrix multiplication between two 2D tensors
 *  Pre-conditions: wx < ix, and wy < iy
 */
void ai2_pointwise_mul_mm_2d(float *output, const float *alpha, int ix, int iy, int wx, int wy, int pad) {
    // Slower version
//      for (int y = 0; y < iy; ++y) 
//          for (int x = 0; x < ix; x++)
//              output[y*ix+x] *= input[(y % wy)*wx + (x % wx)];

    // Stride prefetch optimized
    for (int s = 0; s < wy; ++s) {  // for each strip
        const float *strip_ptr = &alpha[s*wx];
        for (int y = pad; y < pad + (iy / wy); ++y) {   //
            int stride = y*((ix+2*pad)*wy) + s*(ix+2*pad);
            float *output_ptr = &output[stride];

            for (int x = 0; x < ix; ++x) {
                output_ptr[x] *= strip_ptr[x % wx];
            }
        }
    }
}

void ai2_setFltInput(ai2_bin_conv_layer *layer, float *new_input) {
    if (new_input != NULL) {
        if (layer->input != NULL)
            free(layer->input);
        layer->input = new_input;

        dim3 dim;
        dim.x = layer->px;
        dim.y = layer->py;
        dim.z = layer->c;

        // Binarize input
        ai2_flt_to_bin(layer->binary_input, layer->input, dim);

        float *new_beta = (float *) calloc (dim.x * dim.y, sizeof(float));
        ai2_setFltBeta(layer, new_beta);

        // layer->input is transposed to (z,x,y) already
        ai2_calc_beta(layer->beta, layer->input, dim);
    }
}

void ai2_setBinInput(ai2_bin_conv_layer *layer, BINARY_WORD *new_input) {
    if (new_input != NULL) {
        if (layer->binary_input != NULL)
            free(layer->binary_input);
        layer->binary_input = new_input;
    }
}

void ai2_setFltWeights(ai2_bin_conv_layer *layer, float *new_weights) {
    if (new_weights != NULL) {
        if (layer->weights != NULL)
            free(layer->weights);
        layer->weights = new_weights;

        dim3 dim;
        dim.x = layer->wx;
        dim.y = layer->wy;
        dim.z = layer->c;

        ai2_flt_to_bin(layer->binary_weights, layer->weights, dim);

        // Calculate alpha
        if (layer->alpha != NULL)
            free(layer->alpha);

        layer->alpha = (float *) calloc (dim.x * dim.y, sizeof(float));
        // layer->weights is already transposed to (z,x,y) from ai2_flt_to_bin()
        ai2_calc_alpha(layer->alpha, layer->weights, dim);
    }
}

void ai2_setBinWeights(ai2_bin_conv_layer *layer, BINARY_WORD *new_weights) {
    if (new_weights != NULL) {
        if (layer->binary_weights != NULL)
            free(layer->binary_weights);
        layer->binary_weights = new_weights;
    }
}

void ai2_setFltOutput(ai2_bin_conv_layer *layer, float *new_output) {
    if (new_output != NULL) {
        if (layer->output != NULL)
            free(layer->output);
        layer->output = new_output;
    }
}

void ai2_setBinOutput(ai2_bin_conv_layer *layer, BINARY_WORD *new_output) {
    if (new_output != NULL) {
        if (layer->binary_output != NULL)
            free(layer->binary_output);
        layer->binary_output = new_output;
    }
}

void ai2_setFltAlpha(ai2_bin_conv_layer *layer, float *new_alpha) {
    if (new_alpha != NULL) {
        if (layer->alpha != NULL)
            free(layer->alpha);
        layer->alpha = new_alpha;
    }
}

void ai2_setFltBeta(ai2_bin_conv_layer *layer, float *new_beta) {
    if (new_beta != NULL) {
        if (layer->beta != NULL)
            free(layer->beta);
        layer->beta = new_beta;
    }
}

void ai2_setFltNewBeta(ai2_bin_conv_layer *layer, float *new_new_beta) {
    if (new_new_beta != NULL) {
        if (layer->new_beta != NULL)
            free(layer->new_beta);
        layer->new_beta = new_new_beta;
    }
}

float* ai2_getFltOutput(ai2_bin_conv_layer *layer) {
    //if (layer->output != NULL && layer->binary_output != NULL) {
    if (layer->output != NULL) {

        // The idea here was that all intermediate states are stored in the binary output. 
        // Whenever the user needs the real-valued output, the conversion happens at this function call.
        //dim3 dim;
        //dim.x = layer->px;
        //dim.y = layer->py;
        //dim.z = layer->batch;
        //ai2_bin_to_flt(layer->output, layer->binary_output, dim);

        return layer->output;
    }
    else
        return NULL;
}

void ai2_transpose3D(float *data, dim3 d) {
    // Slow transpose for correctness

    // (x,y,z) becomes (z,x,y). Requires two transposes:
    //  (x,y,z) -> (x,z,y).
    //  (x,z,y) -> (z,x,y).

    // Intermediate buffer
    float *new_data = (float *) calloc (d.x * d.y * d.z, sizeof(float));

    // Transpose y and z axis.
    // (x,y,z) -> (x,z,y);
    for (int y = 0; y < d.y; ++y) {
        for (int z = 0; z < d.z; ++z) {
            for (int x = 0; x < d.x; ++x) {
                new_data[y*d.x*d.z + z*d.x + x] = data[z*d.x*d.y + y*d.x + x];
                //new_data[z*d.y*d.x + y*d.x + x] = data[y*d.x*d.z + z*d.x + x];
            }
        }
    }

    // Transpose x and z axis.
    //  (x,z,y) -> (z,x,y)
    for (int y = 0; y < d.y; ++y) {
        for (int x = 0; x < d.x; ++x) {
            for (int z = 0; z < d.z; ++z) {
                data[y*d.z*d.x + x*d.z + z] = new_data[y*d.x*d.z + x + z*d.x];
            }
        }
    }

    free(new_data);
}

int ai2_isFloatWhole(float f) { // TODO unit test
    return (ceilf(f) == f) ? 1 : 0;
}

/* @brief Initialize and create all memory arrays for this layer
 * b - batches (number of filter batches)
 * c - input channels
 * ix - input width
 * iy - input height
 * wx - weight/filter width
 * wy - weight/filter height
 * s - stride between sliding windows
 * pad - the amount of padding
 */
ai2_bin_conv_layer ai2_make_bin_conv_layer(int b, int c, int ix, int iy, int wx, int wy, int s, int pad) {
    // http://cs231n.github.io/convolutional-networks/
    //  See: spatial arrangement section for determining what the output size will be
    float output_size = ((ix - wx + 2 * pad) / s) + 1;
    if (ai2_isFloatWhole(output_size) == 0) {
        fprintf(stderr, "ERROR! conv layer of (b,c,ix,iy,s,pad) = (%d, %d, %d, %d, %d, %d) will give "
            " invalid output dimension: %fx%f\n", b, c, ix, iy, s, pad, output_size, output_size);
        exit(1);
    }

    // TODO: Support strided output
    if (s != 1) {
        fprintf(stderr, "ERROR! Only stride values of 1 is supported\n");
        exit(1);
    }

    // padded input size
    int px = (int) ix + 2*pad; 
    int py = (int) iy + 2*pad;

    ai2_bin_conv_layer l = {0}; // initialize all to 0
    l.input = (float *) calloc (c * px * py, sizeof(float));        // is padded
    l.binary_input =   (BINARY_WORD *) calloc (c * px * py / BITS_PER_BINARY_WORD, sizeof(BINARY_WORD));     // is padded

    dim3 dim;
    dim.x = px;
    dim.y = py;
    dim.z = c;
    ai2_flt_to_bin(l.binary_input, l.input, dim);

    l.weights = (float *) calloc (b * c * wx * wy, sizeof(float));  
    l.binary_weights = (BINARY_WORD *) calloc (b * c * wx * wy / BITS_PER_BINARY_WORD, sizeof(BINARY_WORD));

    l.output = (float *) calloc (c * px * py, sizeof(float));   // is padded
    l.new_beta = (float *) calloc(px * py, sizeof(float));      // is padded

    l.batch = b;
    l.c = c;
    l.h = iy;
    l.w = ix;
    l.stride = s;
    l.pad = pad;
    l.px = px;
    l.py = py;
    l.wx = wx;
    l.wy = wy;

    // The following parameters are uninitialized and should be set elsewhere:
    //  l.beta  - padded
    //  l.alpha - not padded

    return l;
}

void ai2_free_bin_conv_layer(ai2_bin_conv_layer *layer) {
    if (layer->input) free (layer->input);
    if (layer->binary_input) free(layer->binary_input);
    if (layer->weights) free (layer->weights);
    if (layer->binary_weights) free(layer->binary_weights);
    if (layer->output) free(layer->output);
    if (layer->binary_output) free (layer->binary_output);
    if (layer->alpha) free(layer->alpha);
    if (layer->beta) free(layer->beta);
    if (layer->new_beta) free(layer->new_beta);
}

void ai2_throw_error(char *str) {
    fprintf(stderr, "ERROR: %s\n", str);
    exit(1);
}

void ai2_bin_forward(ai2_bin_conv_layer *l) {
    if (l->input == NULL) ai2_throw_error("Input was not allocated and set in this layer");
    if (l->weights == NULL) ai2_throw_error("Weights was not allocated and set in this layer");
    if (l->output == NULL) ai2_throw_error("Output was not allocated and set in this layer");
    if (l->alpha == NULL) ai2_throw_error("Alpha was not allocated and set in this layer");
    if (l->beta == NULL) ai2_throw_error("Beta was not allocated and set in this layer");

    if (l->c % 32 != 0) ai2_throw_error("Channel is not divisible by 32. Need to implement mask "
                                        "before supporting arbitrary channel size. For now, "
                                        "set the channel size to the nearest multiple of 32 "
                                        "and ignore any ''extra'' channels unused.");

    l->c /= BITS_PER_BINARY_WORD;   // For compensating with doing more work per word

    float *output = l->output;
    float *alpha = l->alpha;
    float *beta = l->beta;
    int px = l->px;
    int py = l->py;
    BINARY_WORD *binary_weights = l->binary_weights;

    for (int z = 0; z < l->batch; ++z) {    // for each filter map
        BINARY_WORD *binary_input = l->binary_input;
        for (int c = 0; c < l->c; ++c) {    // for each input channel
            ai2_bin_conv2D(output, binary_input, binary_weights, l->w, l->h, l->wx, l->wy, l->pad, l->stride);
            binary_input += px*py;   // increment with next 2D plane
            binary_weights += l->wx*l->wy;       // increment with next 2D plane

            ai2_pointwise_mul_mm(output, beta, px*py);  
            ai2_pointwise_mul_mm_2d(output, alpha, l->w, l->h, l->wx, l->wy, l->pad);
        }
    }
}

// Deprecated
//double ai2_bin_conv_benchmark(ConvolutionArgs conv_args) {
//    printf("Running Binary Convolution test!\n");
//
//    size_t ix, iy, iz, wx, wy, wz, L, stride;
//    ix = conv_args.input.x;
//    iy = conv_args.input.y;
//    iz = conv_args.input.z;
//    wx = conv_args.weights.x;
//    wy = conv_args.weights.y;
//    wz = conv_args.weights.z;
//  L = BITS_PER_BINARY_WORD;
//  stride = 1;
//
//    printf("Input size (num elements, xyz): %zu %zu %zu\n", ix, iy, iz);
//    printf("Weights size (num elements. xyz): %zu %zu %zu\n", wx, wy, wz);
//
//    double sz_input_elements = ix * iy * iz;
//    double sz_input_bytes = getSizeBytesBinaryArray(conv_args.input);
//    double sz_weight_bytes = getSizeBytesBinaryArray(conv_args.weights);
//
//    printf("Input Size (MB): %f\n", sz_input_bytes / (1 << 20));
//    printf("Weight Size (MB): %f\n", sz_weight_bytes / (1 << 20));
//
//    BINARY_WORD *binary_input = mallocBinaryVolume(conv_args.input);
//    BINARY_WORD *binary_weights = mallocBinaryVolume(conv_args.weights);
//    BINARY_WORD *b_input = binary_input;    // alias
//    BINARY_WORD *b_weight = binary_weights; // alias
//    float *output = mallocFloatVolume(conv_args.output);
//  float *output_ptr = output;
//  float *beta =  (float *) malloc(sizeof(float) * ix * iy);   // we assume beta is given to us
//  float *alpha = (float *) malloc(sizeof(float) * wx * wy);   // we assume alpha is given to us
//  float *new_output = mallocFloatVolume(conv_args.output);
//  //float *new_output_ptr = new_output;
//  float *new_beta = (float *) malloc(sizeof(float) * ix * iy);
//  //float *new_beta_ptr = new_beta;
//
//    // Scale number of computations because we're packing.
//    // After this point, you should not have to reason about input dimensions for input and weights.
//    iz /= BITS_PER_BINARY_WORD;
//    wz /= BITS_PER_BINARY_WORD;
//
//    // Calculate time taken by a request
//    struct timeval start_time;
//    gettimeofday(&start_time, NULL);
//
//  // Preprocessing
//  int pad = wx/2;
//
//    for (int z = 0; z < iz; ++z) {    // number of channels
//        ai2_bin_conv2D(output_ptr, b_input, b_weight, ix, iy, wx, wy, pad, stride);
//        b_input += ix*iy;   // increment with next 2D plane
//        b_weight += wx*wy;  // increment with next 2D plane
//
//      ai2_pointwise_mul_mm(output_ptr, beta, ix*iy);
//      ai2_pointwise_mul_mm_2d(output_ptr, alpha, ix, iy, wx, wy, pad);
//    }
//
//  // copy to new array (need to wrap this around); TODO.
//    struct timeval end_time;
//    gettimeofday(&end_time, NULL);
//
//    struct timeval diff_time;
//    timersub(&end_time, &start_time, &diff_time);
//    double time_conv_s = diff_time.tv_sec + diff_time.tv_usec * 1e-6;
//    double time_conv_ms = time_conv_s * 1000.0;
//
//  double model_ops = (3*ix*iy*wx*wy*wz/L) + 2*ix*iy + ix*iy*iz;
//  double conv_ops_s = 1e-9 * model_ops / time_conv_s;
//    double conv_bandwidth_gb_s = 1e-9 * sz_input_bytes / (time_conv_ms / 1000.0);
//    double conv_bandwidth_gelement_s = 1e-9 * sz_input_elements / (time_conv_ms / 1000.0);
//
//    printf("Execution Time (ms): %f\n", time_conv_ms);
//    printf("Binary Convolution OPS/s (GOPS/s): %f\n", conv_ops_s);
//    printf("Binary Convolution Bandwidth (GB/s): %f\n", conv_bandwidth_gb_s);
//    printf("Binary Convolution Bandwidth (GElements/s): %f\n\n", conv_bandwidth_gelement_s);
//
//    free(binary_input);
//    free(binary_weights);
//    free(output);
//  free(beta);
//  free(alpha);
//  free(new_output);
//  free(new_beta);
//
//    return time_conv_ms;
//}

// double ai2_bin_conv_benchmark(ConvolutionArgs conv_args);

//void benchmark() {
//    int ix, iy, iz, wx, wy, wz;
//    iz = (1 << 9) * BITS_PER_BINARY_WORD;
//    ix = 227; // x == y for square face
//    iy = 227;
//    wx = 3;    // x == y for a square face
//    wy = 3;
//    wz = iz;
//
//  int runs = 1;
//  double accum_binary = 0;
//  double accum_real = 0;
//    ConvolutionArgs conv_args = initArgs(ix, iy, iz, wx, wy, wz);
//  for (int i = 0; i < runs; ++i) {
//      double t_binary_convolve = ai2_bin_conv_benchmark(conv_args);
//      double t_real_convolve = run_convolve2D_real(conv_args);
//      printf("t binary = %lf\n", t_binary_convolve);
//      printf("t real = %lf\n", t_real_convolve);
//      accum_binary += t_binary_convolve;
//      accum_real += t_real_convolve;
//  }
//
//  accum_binary /= runs;
//  accum_real /= runs;
//  printf("Average convolution pass binary (ms): %lf\n", accum_binary);
//  printf("Average convolution pass flt (ms): %lf\n", accum_real);
//  printf("Speedup (Binary over Real): %lfx\n", accum_real / accum_binary);    
//  exit(1);
//}
