#include "layer.h"
#include "cuda.h"
#include <stdlib.h>

void free_layer(layer l)
{
    if(l.type == DROPOUT){
        if(l.rand)           { free(l.rand); l.rand = 0;}
#ifdef GPU
        if(l.rand_gpu)            { cuda_free(l.rand_gpu); l.rand_gpu = 0;}
#endif
        return;
    }
    if(l.cweights)           { free(l.cweights); l.cweights = 0;}
    if(l.indexes)            { free(l.indexes); l.indexes = 0;}
    if(l.input_layers)       { free(l.input_layers); l.input_layers = 0;}
    if(l.input_sizes)        { free(l.input_sizes); l.input_sizes = 0;}
    if(l.map)                { free(l.map); l.map = 0;}
    if(l.rand)               { free(l.rand); l.rand = 0;}
    if(l.cost)               { free(l.cost); l.cost = 0;}
    if(l.state)              { free(l.state); l.state = 0;}
    if(l.prev_state)         { free(l.prev_state); l.prev_state = 0;}
    if(l.forgot_state)       { free(l.forgot_state); l.forgot_state = 0;}
    if(l.forgot_delta)       { free(l.forgot_delta); l.forgot_delta = 0;}
    if(l.state_delta)        { free(l.state_delta); l.state_delta = 0;}
    if(l.concat)             { free(l.concat); l.concat = 0;}
    if(l.concat_delta)       { free(l.concat_delta); l.concat_delta = 0;}
    if(l.binary_weights)     { free(l.binary_weights); l.binary_weights = 0;}
    if(l.biases)             { free(l.biases); l.biases = 0;}
    if(l.bias_updates)       { free(l.bias_updates); l.bias_updates = 0;}
    if(l.scales)             { free(l.scales); l.scales = 0;}
    if(l.scale_updates)      { free(l.scale_updates); l.scale_updates = 0;}
    if(l.weights)            { free(l.weights); l.weights = 0;}
    if(l.weight_updates)     { free(l.weight_updates); l.weight_updates = 0;}
    if(l.delta)              { free(l.delta); l.delta = 0;}
    if(l.output)             { free(l.output); l.output = 0;}
    if(l.squared)            { free(l.squared); l.squared = 0;}
    if(l.norms)              { free(l.norms); l.norms = 0;}
    if(l.spatial_mean)       { free(l.spatial_mean); l.spatial_mean = 0;}
    if(l.mean)               { free(l.mean); l.mean = 0;}
    if(l.variance)           { free(l.variance); l.variance = 0;}
    if(l.mean_delta)         { free(l.mean_delta); l.mean_delta = 0;}
    if(l.variance_delta)     { free(l.variance_delta); l.variance_delta = 0;}
    if(l.rolling_mean)       { free(l.rolling_mean); l.rolling_mean = 0;}
    if(l.rolling_variance)   { free(l.rolling_variance); l.rolling_variance = 0;}
    if(l.x)                  { free(l.x); l.x = 0;}
    if(l.x_norm)             { free(l.x_norm); l.x_norm = 0;}
    if(l.m)                  { free(l.m); l.m = 0;}
    if(l.v)                  { free(l.v); l.v = 0;}
    if(l.z_cpu)              { free(l.z_cpu); l.z_cpu = 0;}
    if(l.r_cpu)              { free(l.r_cpu); l.r_cpu = 0;}
    if(l.h_cpu)              { free(l.h_cpu); l.h_cpu = 0;}
    if(l.binary_input)       { free(l.binary_input); l.binary_input = 0;}

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
