
__kernel void bias(int n, int size, __global float *biases, __global float *output)
{
    int id = get_global_id(0);
    int batch = get_global_id(1);
    int filter = id/size;
    //int position = id%size;

    output[batch*n*size + id] = biases[filter];
}

__kernel void learn_bias(int batch, int n, int size, __global float *delta, __global float *bias_updates)
{
    __local float part[BLOCK];
    int i,b;
    int filter = get_group_id(0);
    int p = get_local_id(0);
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (index < size) ? delta[index] : 0;
        }
    }
    part[p] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if(p == 0){
        for(i = 0; i < BLOCK; ++i) bias_updates[filter] += part[i];
    }
}

