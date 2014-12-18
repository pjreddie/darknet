__kernel void yoloswag420blazeit360noscope(__global float *input, __global float *rand, float prob, float scale, __global float *output)
{
    int id = get_global_id(0);
    output[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}
