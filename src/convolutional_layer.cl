
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
    int i,b;
    int filter = get_global_id(0);
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; ++i){
            int index = i + size*(filter + n*b);
            sum += delta[index];
        }
    }
    bias_updates[filter] += sum;
}

