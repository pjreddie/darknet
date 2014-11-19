__kernel void forward(__global float *input, __global float *rand, float prob)
{
    int id = get_global_id(0);
    input[id] = (rand[id] < prob) ? 0 : input[id]/(1.-prob);
}
