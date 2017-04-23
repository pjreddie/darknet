#include "layer.h"
#include "cuda.h"
#include <stdlib.h>

void free_layer(layer l)
{
    if(l.type == DROPOUT){
        if(l.rand)           free(l.rand);
#ifdef GPU
        if(l.rand_gpu)            { cuda_free(l.rand_gpu); l.rand_gpu = 0;}
#endif
        return;
    }
    if(l.cweights)           free(l.cweights);
    if(l.indexes)            free(l.indexes);
    if(l.input_layers)       free(l.input_layers);
    if(l.input_sizes)        free(l.input_sizes);
    if(l.map)                free(l.map);
    if(l.rand)               free(l.rand);
    if(l.cost)               free(l.cost);
    if(l.state)              free(l.state);
    if(l.prev_state)         free(l.prev_state);
    if(l.forgot_state)       free(l.forgot_state);
    if(l.forgot_delta)       free(l.forgot_delta);
    if(l.state_delta)        free(l.state_delta);
    if(l.concat)             free(l.concat);
    if(l.concat_delta)       free(l.concat_delta);
    if(l.binary_weights)     free(l.binary_weights);
    if(l.biases)             free(l.biases);
    if(l.bias_updates)       free(l.bias_updates);
    if(l.scales)             free(l.scales);
    if(l.scale_updates)      free(l.scale_updates);
    if(l.weights)            free(l.weights);
    if(l.weight_updates)     free(l.weight_updates);
    if(l.delta)              free(l.delta);
    if(l.output)             free(l.output);
    if(l.squared)            free(l.squared);
    if(l.norms)              free(l.norms);
    if(l.spatial_mean)       free(l.spatial_mean);
    if(l.mean)               free(l.mean);
    if(l.variance)           free(l.variance);
    if(l.mean_delta)         free(l.mean_delta);
    if(l.variance_delta)     free(l.variance_delta);
    if(l.rolling_mean)       free(l.rolling_mean);
    if(l.rolling_variance)   free(l.rolling_variance);
    if(l.x)                  free(l.x);
    if(l.x_norm)             free(l.x_norm);
    if(l.m)                  free(l.m);
    if(l.v)                  free(l.v);
    if(l.z_cpu)              free(l.z_cpu);
    if(l.r_cpu)              free(l.r_cpu);
    if(l.h_cpu)              free(l.h_cpu);
    if(l.binary_input)       free(l.binary_input);

#ifdef GPU

    if(l.indexes_gpu)           { cuda_free((float *)l.indexes_gpu); l.indexes_gpu = 0;}

    if(l.z_gpu)                  { cuda_free(l.z_gpu); l.z_gpu = 0;}
    if(l.r_gpu)                  {  cuda_free(l.r_gpu); l.r_gpu = 0;}
    if(l.h_gpu)                  {  cuda_free(l.h_gpu); l.h_gpu = 0;}
    if(l.m_gpu)                  {  cuda_free(l.m_gpu); l.m_gpu = 0;}
    if(l.v_gpu)                  {  cuda_free(l.v_gpu); l.v_gpu = 0;}
    if(l.prev_state_gpu)         {  cuda_free(l.prev_state_gpu); l.prev_state_gpu = 0;}
    if(l.forgot_state_gpu)       {  cuda_free(l.forgot_state_gpu); l.forgot_state_gpu = 0;}
    if(l.forgot_delta_gpu)       {  cuda_free(l.forgot_delta_gpu); l.forgot_delta_gpu = 0;}
    if(l.state_gpu)              {  cuda_free(l.state_gpu); l.state_gpu = 0;}
    if(l.state_delta_gpu)        {  cuda_free(l.state_delta_gpu); l.state_delta_gpu = 0;}
    if(l.gate_gpu)               {  cuda_free(l.gate_gpu); l.gate_gpu = 0;}
    if(l.gate_delta_gpu)         {  cuda_free(l.gate_delta_gpu); l.gate_delta_gpu = 0;}
    if(l.save_gpu)                { cuda_free(l.save_gpu); l.save_gpu = 0;}
    if(l.save_delta_gpu)          { cuda_free(l.save_delta_gpu); l.save_delta_gpu = 0;}
    if(l.concat_gpu)              { cuda_free(l.concat_gpu); l.concat_gpu = 0;}
    if(l.concat_delta_gpu)        { cuda_free(l.concat_delta_gpu); l.concat_delta_gpu = 0;}
    if(l.binary_input_gpu)        { cuda_free(l.binary_input_gpu); l.binary_input_gpu = 0;}
    if(l.binary_weights_gpu)      { cuda_free(l.binary_weights_gpu); l.binary_weights_gpu = 0;}
    if(l.mean_gpu)                { cuda_free(l.mean_gpu); l.mean_gpu = 0;}
    if(l.variance_gpu)            { cuda_free(l.variance_gpu); l.variance_gpu = 0;}
    if(l.rolling_mean_gpu)        { cuda_free(l.rolling_mean_gpu); l.rolling_mean_gpu = 0;}
    if(l.rolling_variance_gpu)    { cuda_free(l.rolling_variance_gpu); l.rolling_variance_gpu = 0;}
    if(l.variance_delta_gpu)      { cuda_free(l.variance_delta_gpu); l.variance_delta_gpu = 0;}
    if(l.mean_delta_gpu)          { cuda_free(l.mean_delta_gpu); l.mean_delta_gpu = 0;}
    if(l.x_gpu)                   { cuda_free(l.x_gpu); l.x_gpu = 0;}
    if(l.x_norm_gpu)              { cuda_free(l.x_norm_gpu); l.x_norm_gpu = 0;}
    if(l.weights_gpu)             { cuda_free(l.weights_gpu); l.weights_gpu = 0;}
    if(l.weight_updates_gpu)      { cuda_free(l.weight_updates_gpu); l.weight_updates_gpu = 0;}
    if(l.biases_gpu)              { cuda_free(l.biases_gpu); l.biases_gpu = 0;}
    if(l.bias_updates_gpu)        { cuda_free(l.bias_updates_gpu); l.bias_updates_gpu = 0;}
    if(l.scales_gpu)              { cuda_free(l.scales_gpu); l.scales_gpu = 0;}
    if(l.scale_updates_gpu)       { cuda_free(l.scale_updates_gpu); l.scale_updates_gpu = 0;}
    if(l.output_gpu)              { cuda_free(l.output_gpu); l.output_gpu = 0;}
    if(l.delta_gpu)               { cuda_free(l.delta_gpu); l.delta_gpu = 0;}
    if(l.rand_gpu)                { cuda_free(l.rand_gpu); l.rand_gpu = 0;}
    if(l.squared_gpu)             { cuda_free(l.squared_gpu); l.squared_gpu = 0;}
    if(l.norms_gpu)               { cuda_free(l.norms_gpu); l.norms_gpu = 0;}

#endif
}
