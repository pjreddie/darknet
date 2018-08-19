#include "converter.h"

#include <math.h>

const char *get_precision_str(PRECISION precision)
{
    switch(precision) {
        case UINT8: return "UINT8";
        case INT8: return "INT8";
        case INT16: return "INT8";
        case FP16: return "FP16";
        case FP32: return "FP32";
    }

    return "FP32";
}

PRECISION get_precision(const char *s)
{
    PRECISION precision = FP32;
    if (strcmp(s, "uint8") == 0)
        precision = UINT8;
    else if (strcmp(s, "int8") == 0)
        precision = INT8;
    else if (strcmp(s, "int16") == 0)
        precision = INT16;
    else if (strcmp(s, "fp16") == 0)
        precision = FP16;
    else if (strcmp(s, "fp32") == 0)
        precision = FP32;
    else {
        fprintf(stderr, "INVALID PRECISION specified. "
                        "Defaulting to FP32\n");
    }

    return precision;
}

double int_to_fp_elmt(
    int8_t in_element,
    converter_params params,
    int channel
)
{
    // ignored post_scale and offset, because post_scale always 1 and offset always 0;
    
    // FIXME: use double just to apple-to-apple compare with AMOD; float should be fine
    // for demo which provide better perf;
    double scale = params.scale;
    double shifter = pow(2.0, params.shifter);

    double out_value;
    out_value = ((double)in_element) * scale/shifter;

    return out_value;
}




/**
 * Converts from in_element double to int in specified range
 * [min_value, max_value]
 **/
long long int fp_to_int_elmt(
    double in_element,
    long long int min_value,
    long long int max_value,
    converter_params params,
    int channel
)
{
    double scale = params.scale;
    double shifter = pow(2.0, params.shifter);
    double post_scale = params.post_scale;
    double offset = params.offset;
    double post_offset = params.post_offset;
    double scaled = (in_element - offset) * scale/shifter;

    long long int out_value;
    if(isnan(in_element)) {
        out_value = min_value;
    }
    else {
        if (scaled > max_value)
            out_value = max_value;
        else if(scaled < min_value)
            out_value = min_value;
        else
            out_value = round(scaled);
    }

    out_value = out_value * ((long long int)post_scale)
                    - (long long int)post_offset;
    if (out_value > max_value)
        out_value = max_value;
    else if (out_value < min_value)
        out_value = min_value;

    return out_value;
}

void fp32_to_uint8(fp32 *in, uint8_t *out, unsigned count,
                    converter_params params)
{
    unsigned i;
    uint8_t min_value = 0;
    uint8_t max_value = 255;

    for (i = 0; i < count; i++) {
        long long int out_value =
            fp_to_int_elmt(in[i], min_value, max_value, params, 0);
        out[i] = (uint8_t)out_value;
    }
}

void fp32_to_int8(fp32 *in, int8_t *out, unsigned count,
                    converter_params params)
{
    unsigned i;
    int8_t min_value = -128;
    int8_t max_value = 127;

    for (i = 0; i < count; i++) {
        long long int out_value =
            fp_to_int_elmt(in[i], min_value, max_value, params, 0);
        out[i] = (int8_t)out_value;
    }
}

void uint8_to_fp32(uint8_t *in, fp32 *out, unsigned count,
                    converter_params params)
{
    unsigned i = 0;

    for (i = 0; i < count; i++) {
        out[i] = (fp32)int_to_fp_elmt(in[i], params, 0);
        if (fpclassify(out[i]) == FP_SUBNORMAL) {
            out[i] = 0;
        }
    }
}

void int8_to_fp32(int8_t *in, fp32 *out, unsigned count,
                    converter_params params)
{
    unsigned i = 0;

    for (i = 0; i < count; i++) {
        out[i] = (fp32)int_to_fp_elmt(in[i], params, 0);
    }
}
