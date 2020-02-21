#include "layer.h"
#include "dark_cuda.h"
#include <stdlib.h>

void free_sublayer(layer *l)
{
    if (l) {
        free_layer(*l);
        free(l);
    }
}

void free_layer(layer l)
{
    free_layer_custom(l, 0);
}

void free_layer_custom(layer l, int keep_cudnn_desc)
{
    if (l.share_layer != NULL) return;    // don't free shared layers
    if (l.antialiasing) {
        free_sublayer(l.input_layer);
    }
    if (l.type == CONV_LSTM) {
        if (l.peephole) {
            free_sublayer(l.vf);
            free_sublayer(l.vi);
            free_sublayer(l.vo);
        }
        else {
            free(l.vf);
            free(l.vi);
            free(l.vo);
        }
        free_sublayer(l.wf);
        free_sublayer(l.wi);
        free_sublayer(l.wg);
        free_sublayer(l.wo);
        free_sublayer(l.uf);
        free_sublayer(l.ui);
        free_sublayer(l.ug);
        free_sublayer(l.uo);
    }
    if (l.type == CRNN) {
        free_sublayer(l.input_layer);
        free_sublayer(l.self_layer);
        free_sublayer(l.output_layer);
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
        if (l.rand_gpu)              cuda_free(l.rand_gpu);
        if (l.drop_blocks_scale)     cuda_free_host(l.drop_blocks_scale);
        if (l.drop_blocks_scale_gpu) cuda_free(l.drop_blocks_scale_gpu);
#endif
        return;
    }
    if (l.mask)               free(l.mask);
    if (l.classes_multipliers)free(l.classes_multipliers);
    if (l.cweights)           free(l.cweights);
    if (l.indexes)            free(l.indexes);
    if (l.input_layers)       free(l.input_layers);
    if (l.input_sizes)        free(l.input_sizes);
    if (l.layers_output)      free(l.layers_output);
    if (l.layers_delta)       free(l.layers_delta);
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
    if (l.biases)             free(l.biases), l.biases = NULL;
    if (l.bias_updates)       free(l.bias_updates), l.bias_updates = NULL;
    if (l.scales)             free(l.scales), l.scales = NULL;
    if (l.scale_updates)      free(l.scale_updates), l.scale_updates = NULL;
    if (l.weights)            free(l.weights), l.weights = NULL;
    if (l.weight_updates)     free(l.weight_updates), l.weight_updates = NULL;
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
    if (l.delta)              free(l.delta), l.delta = NULL;
    if (l.output)             free(l.output), l.output = NULL;
    if (l.activation_input)   free(l.activation_input), l.activation_input = NULL;
    if (l.squared)            free(l.squared);
    if (l.norms)              free(l.norms);
    if (l.spatial_mean)       free(l.spatial_mean);
    if (l.mean)               free(l.mean), l.mean = NULL;
    if (l.variance)           free(l.variance), l.variance = NULL;
    if (l.mean_delta)         free(l.mean_delta), l.mean_delta = NULL;
    if (l.variance_delta)     free(l.variance_delta), l.variance_delta = NULL;
    if (l.rolling_mean)       free(l.rolling_mean), l.rolling_mean = NULL;
    if (l.rolling_variance)   free(l.rolling_variance), l.rolling_variance = NULL;
    if (l.x)                  free(l.x);
    if (l.x_norm)             free(l.x_norm);
    if (l.m)                  free(l.m);
    if (l.v)                  free(l.v);
    if (l.z_cpu)              free(l.z_cpu);
    if (l.r_cpu)              free(l.r_cpu);
    if (l.binary_input)       free(l.binary_input);
    if (l.bin_re_packed_input) free(l.bin_re_packed_input);
    if (l.t_bit_input)        free(l.t_bit_input);
    if (l.loss)               free(l.loss);

    // CONV-LSTM
    if (l.f_cpu)               free(l.f_cpu);
    if (l.i_cpu)               free(l.i_cpu);
    if (l.g_cpu)               free(l.g_cpu);
    if (l.o_cpu)               free(l.o_cpu);
    if (l.c_cpu)               free(l.c_cpu);
    if (l.h_cpu)               free(l.h_cpu);
    if (l.temp_cpu)            free(l.temp_cpu);
    if (l.temp2_cpu)           free(l.temp2_cpu);
    if (l.temp3_cpu)           free(l.temp3_cpu);
    if (l.dc_cpu)              free(l.dc_cpu);
    if (l.dh_cpu)              free(l.dh_cpu);
    if (l.prev_state_cpu)      free(l.prev_state_cpu);
    if (l.prev_cell_cpu)       free(l.prev_cell_cpu);
    if (l.stored_c_cpu)        free(l.stored_c_cpu);
    if (l.stored_h_cpu)        free(l.stored_h_cpu);
    if (l.cell_cpu)            free(l.cell_cpu);

#ifdef GPU
    if (l.indexes_gpu)           cuda_free((float *)l.indexes_gpu);

    if (l.z_gpu)                   cuda_free(l.z_gpu);
    if (l.r_gpu)                   cuda_free(l.r_gpu);
    if (l.m_gpu)                   cuda_free(l.m_gpu);
    if (l.v_gpu)                   cuda_free(l.v_gpu);
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
    if (l.mean_gpu)                cuda_free(l.mean_gpu), l.mean_gpu = NULL;
    if (l.variance_gpu)            cuda_free(l.variance_gpu), l.variance_gpu = NULL;
    if (l.m_cbn_avg_gpu)           cuda_free(l.m_cbn_avg_gpu), l.m_cbn_avg_gpu = NULL;
    if (l.v_cbn_avg_gpu)           cuda_free(l.v_cbn_avg_gpu), l.v_cbn_avg_gpu = NULL;
    if (l.rolling_mean_gpu)        cuda_free(l.rolling_mean_gpu), l.rolling_mean_gpu = NULL;
    if (l.rolling_variance_gpu)    cuda_free(l.rolling_variance_gpu), l.rolling_variance_gpu = NULL;
    if (l.variance_delta_gpu)      cuda_free(l.variance_delta_gpu), l.variance_delta_gpu = NULL;
    if (l.mean_delta_gpu)          cuda_free(l.mean_delta_gpu), l.mean_delta_gpu = NULL;
    if (l.x_norm_gpu)              cuda_free(l.x_norm_gpu);

    // assisted excitation
    if (l.gt_gpu)                  cuda_free(l.gt_gpu);
    if (l.a_avg_gpu)               cuda_free(l.a_avg_gpu);

    if (l.align_bit_weights_gpu)   cuda_free((float *)l.align_bit_weights_gpu);
    if (l.mean_arr_gpu)            cuda_free(l.mean_arr_gpu);
    if (l.align_workspace_gpu)     cuda_free(l.align_workspace_gpu);
    if (l.transposed_align_workspace_gpu) cuda_free(l.transposed_align_workspace_gpu);

    if (l.weights_gpu)             cuda_free(l.weights_gpu), l.weights_gpu = NULL;
    if (l.weight_updates_gpu)      cuda_free(l.weight_updates_gpu), l.weight_updates_gpu = NULL;
    if (l.weight_deform_gpu)       cuda_free(l.weight_deform_gpu), l.weight_deform_gpu = NULL;
    if (l.weights_gpu16)           cuda_free(l.weights_gpu16), l.weights_gpu16 = NULL;
    if (l.weight_updates_gpu16)    cuda_free(l.weight_updates_gpu16), l.weight_updates_gpu16 = NULL;
    if (l.biases_gpu)              cuda_free(l.biases_gpu), l.biases_gpu = NULL;
    if (l.bias_updates_gpu)        cuda_free(l.bias_updates_gpu), l.bias_updates_gpu = NULL;
    if (l.scales_gpu)              cuda_free(l.scales_gpu), l.scales_gpu = NULL;
    if (l.scale_updates_gpu)       cuda_free(l.scale_updates_gpu), l.scale_updates_gpu = NULL;
    if (l.input_antialiasing_gpu)  cuda_free(l.input_antialiasing_gpu), l.input_antialiasing_gpu = NULL;
    if (l.optimized_memory < 2) {
        if (l.x_gpu)                   cuda_free(l.x_gpu);  l.x_gpu = NULL;
        if (l.output_gpu)              cuda_free(l.output_gpu), l.output_gpu = NULL;
        if (l.activation_input_gpu)    cuda_free(l.activation_input_gpu), l.activation_input_gpu = NULL;
    }
    if (l.delta_gpu && l.keep_delta_gpu && l.optimized_memory < 3) cuda_free(l.delta_gpu), l.delta_gpu = NULL;
    if (l.rand_gpu)                cuda_free(l.rand_gpu);
    if (l.squared_gpu)             cuda_free(l.squared_gpu);
    if (l.norms_gpu)               cuda_free(l.norms_gpu);
    if (l.input_sizes_gpu)         cuda_free((float*)l.input_sizes_gpu);
    if (l.layers_output_gpu)       cuda_free((float*)l.layers_output_gpu);
    if (l.layers_delta_gpu)        cuda_free((float*)l.layers_delta_gpu);

    // CONV-LSTM
    if (l.f_gpu)                   cuda_free(l.f_gpu);
    if (l.i_gpu)                   cuda_free(l.i_gpu);
    if (l.g_gpu)                   cuda_free(l.g_gpu);
    if (l.o_gpu)                   cuda_free(l.o_gpu);
    if (l.c_gpu)                   cuda_free(l.c_gpu);
    if (l.h_gpu)                   cuda_free(l.h_gpu);
    if (l.temp_gpu)                cuda_free(l.temp_gpu);
    if (l.temp2_gpu)               cuda_free(l.temp2_gpu);
    if (l.temp3_gpu)               cuda_free(l.temp3_gpu);
    if (l.dc_gpu)                  cuda_free(l.dc_gpu);
    if (l.dh_gpu)                  cuda_free(l.dh_gpu);
    if (l.prev_state_gpu)          cuda_free(l.prev_state_gpu);
    if (l.prev_cell_gpu)           cuda_free(l.prev_cell_gpu);
    if (l.stored_c_gpu)            cuda_free(l.stored_c_gpu);
    if (l.stored_h_gpu)            cuda_free(l.stored_h_gpu);
    if (l.last_prev_state_gpu)     cuda_free(l.last_prev_state_gpu);
    if (l.last_prev_cell_gpu)      cuda_free(l.last_prev_cell_gpu);
    if (l.cell_gpu)                cuda_free(l.cell_gpu);
#ifdef CUDNN   // shouldn't be used for -map
    if (!keep_cudnn_desc) {
        if (l.srcTensorDesc) CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.srcTensorDesc));
        if (l.dstTensorDesc) CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.dstTensorDesc));
        if (l.srcTensorDesc16) CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.srcTensorDesc16));
        if (l.dstTensorDesc16) CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.dstTensorDesc16));
        if (l.dsrcTensorDesc) CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.dsrcTensorDesc));
        if (l.ddstTensorDesc) CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.ddstTensorDesc));
        if (l.dsrcTensorDesc16) CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.dsrcTensorDesc16));
        if (l.ddstTensorDesc16) CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.ddstTensorDesc16));
        if (l.normTensorDesc) CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.normTensorDesc));
        if (l.normDstTensorDesc) CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.normDstTensorDesc));
        if (l.normDstTensorDescF16) CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.normDstTensorDescF16));

        if (l.weightDesc) CHECK_CUDNN(cudnnDestroyFilterDescriptor(l.weightDesc));
        if (l.weightDesc16) CHECK_CUDNN(cudnnDestroyFilterDescriptor(l.weightDesc16));
        if (l.dweightDesc) CHECK_CUDNN(cudnnDestroyFilterDescriptor(l.dweightDesc));
        if (l.dweightDesc16) CHECK_CUDNN(cudnnDestroyFilterDescriptor(l.dweightDesc16));

        if (l.convDesc) CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(l.convDesc));

        if (l.poolingDesc) CHECK_CUDNN(cudnnDestroyPoolingDescriptor(l.poolingDesc));

        //cudnnConvolutionFwdAlgo_t fw_algo, fw_algo16;
        //cudnnConvolutionBwdDataAlgo_t bd_algo, bd_algo16;
        //cudnnConvolutionBwdFilterAlgo_t bf_algo, bf_algo16;
    }
#endif  // CUDNN

#endif  // GPU
}
