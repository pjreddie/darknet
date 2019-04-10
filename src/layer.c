#include "layer.h"
#include "dark_cuda.h"
#include <stdlib.h>

void free_layer(layer l)
{
    // free layers: input_layer, self_layer, output_layer, ...
    if (l.type == CRNN) {
        if (l.input_layer) {
            free_layer(*l.input_layer);
            free(l.input_layer);
        }
        if (l.self_layer) {
            free_layer(*l.self_layer);
            free(l.self_layer);
        }
        if (l.output_layer) {
            free_layer(*l.output_layer);
            free(l.output_layer);
        }
        l.output = NULL;
        l.delta = NULL;
#ifdef GPU
        l.output_gpu = NULL;
        l.delta_gpu = NULL;
#endif // GPU
    }
    if (l.type == DROPOUT) {
        if (l.rand)           free(l.rand);
#ifdef GPU
        if (l.rand_gpu)             cuda_free(l.rand_gpu);
#endif
        return;
    }
    if (l.mask)               free(l.mask);
    if (l.cweights)           free(l.cweights);
    if (l.indexes)            free(l.indexes);
    if (l.input_layers)       free(l.input_layers);
    if (l.input_sizes)        free(l.input_sizes);
    if (l.map)                free(l.map);
    if (l.rand)               free(l.rand);
    if (l.cost)               free(l.cost);
    if (l.state)              free(l.state);
    if (l.prev_state)         free(l.prev_state);
    if (l.forgot_state)       free(l.forgot_state);
    if (l.forgot_delta)       free(l.forgot_delta);
    if (l.state_delta)        free(l.state_delta);
    if (l.concat)             free(l.concat);
    if (l.concat_delta)       free(l.concat_delta);
    if (l.binary_weights)     free(l.binary_weights);
    if (l.biases)             free(l.biases);
    if (l.bias_updates)       free(l.bias_updates);
    if (l.scales)             free(l.scales);
    if (l.scale_updates)      free(l.scale_updates);
    if (l.weights)            free(l.weights);
    if (l.weight_updates)     free(l.weight_updates);
    if (l.align_bit_weights)  free(l.align_bit_weights);
    if (l.mean_arr)           free(l.mean_arr);
#ifdef GPU
    if (l.delta && l.delta_pinned) {
        cudaFreeHost(l.delta);
        l.delta = NULL;
    }
    if (l.output && l.output_pinned) {
        cudaFreeHost(l.output);
        l.output = NULL;
    }
#endif  // GPU
    if (l.delta)              free(l.delta);
    if (l.output)             free(l.output);
    if (l.squared)            free(l.squared);
    if (l.norms)              free(l.norms);
    if (l.spatial_mean)       free(l.spatial_mean);
    if (l.mean)               free(l.mean);
    if (l.variance)           free(l.variance);
    if (l.mean_delta)         free(l.mean_delta);
    if (l.variance_delta)     free(l.variance_delta);
    if (l.rolling_mean)       free(l.rolling_mean);
    if (l.rolling_variance)   free(l.rolling_variance);
    if (l.x)                  free(l.x);
    if (l.x_norm)             free(l.x_norm);
    if (l.m)                  free(l.m);
    if (l.v)                  free(l.v);
    if (l.z_cpu)              free(l.z_cpu);
    if (l.r_cpu)              free(l.r_cpu);
    if (l.h_cpu)              free(l.h_cpu);
    if (l.binary_input)       free(l.binary_input);
    if (l.bin_re_packed_input) free(l.bin_re_packed_input);
    if (l.t_bit_input)        free(l.t_bit_input);
    if (l.loss)               free(l.loss);

#ifdef GPU
    if (l.indexes_gpu)           cuda_free((float *)l.indexes_gpu);

    if (l.z_gpu)                   cuda_free(l.z_gpu);
    if (l.r_gpu)                   cuda_free(l.r_gpu);
    if (l.h_gpu)                   cuda_free(l.h_gpu);
    if (l.m_gpu)                   cuda_free(l.m_gpu);
    if (l.v_gpu)                   cuda_free(l.v_gpu);
    if (l.prev_state_gpu)          cuda_free(l.prev_state_gpu);
    if (l.forgot_state_gpu)        cuda_free(l.forgot_state_gpu);
    if (l.forgot_delta_gpu)        cuda_free(l.forgot_delta_gpu);
    if (l.state_gpu)               cuda_free(l.state_gpu);
    if (l.state_delta_gpu)         cuda_free(l.state_delta_gpu);
    if (l.gate_gpu)                cuda_free(l.gate_gpu);
    if (l.gate_delta_gpu)          cuda_free(l.gate_delta_gpu);
    if (l.save_gpu)                cuda_free(l.save_gpu);
    if (l.save_delta_gpu)          cuda_free(l.save_delta_gpu);
    if (l.concat_gpu)              cuda_free(l.concat_gpu);
    if (l.concat_delta_gpu)        cuda_free(l.concat_delta_gpu);
    if (l.binary_input_gpu)        cuda_free(l.binary_input_gpu);
    if (l.binary_weights_gpu)      cuda_free(l.binary_weights_gpu);
    if (l.mean_gpu)                cuda_free(l.mean_gpu);
    if (l.variance_gpu)            cuda_free(l.variance_gpu);
    if (l.rolling_mean_gpu)        cuda_free(l.rolling_mean_gpu);
    if (l.rolling_variance_gpu)    cuda_free(l.rolling_variance_gpu);
    if (l.variance_delta_gpu)      cuda_free(l.variance_delta_gpu);
    if (l.mean_delta_gpu)          cuda_free(l.mean_delta_gpu);
    if (l.x_gpu)                   cuda_free(l.x_gpu);  // dont free
    if (l.x_norm_gpu)              cuda_free(l.x_norm_gpu);

    if (l.align_bit_weights_gpu)   cuda_free((float *)l.align_bit_weights_gpu);
    if (l.mean_arr_gpu)            cuda_free(l.mean_arr_gpu);
    if (l.align_workspace_gpu)     cuda_free(l.align_workspace_gpu);
    if (l.transposed_align_workspace_gpu) cuda_free(l.transposed_align_workspace_gpu);

    if (l.weights_gpu)             cuda_free(l.weights_gpu);
    if (l.weight_updates_gpu)      cuda_free(l.weight_updates_gpu);
    if (l.weights_gpu16)           cuda_free(l.weights_gpu16);
    if (l.weight_updates_gpu16)    cuda_free(l.weight_updates_gpu16);
    if (l.biases_gpu)              cuda_free(l.biases_gpu);
    if (l.bias_updates_gpu)        cuda_free(l.bias_updates_gpu);
    if (l.scales_gpu)              cuda_free(l.scales_gpu);
    if (l.scale_updates_gpu)       cuda_free(l.scale_updates_gpu);
    if (l.output_gpu)              cuda_free(l.output_gpu);
    if (l.delta_gpu)               cuda_free(l.delta_gpu);
    if (l.rand_gpu)                cuda_free(l.rand_gpu);
    if (l.squared_gpu)             cuda_free(l.squared_gpu);
    if (l.norms_gpu)               cuda_free(l.norms_gpu);
#endif
}
