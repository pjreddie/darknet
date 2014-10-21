
__kernel void forward(int n, __global float *input, __global float *output)
{
    int b = get_global_id(0);

    int i;
    float sum = 0;
    float largest = -INFINITY;
    for(i = 0; i < n; ++i){
        int val = input[i+b*n];
        largest = (val>largest) ? val : largest;
    }
    for(i = 0; i < n; ++i){
        sum += exp(input[i+b*n]-largest);
    }
    sum = (sum != 0) ? largest+log(sum) : largest-100;
    for(i = 0; i < n; ++i){
        output[i+b*n] = exp(input[i+b*n]-sum);
    }
}

