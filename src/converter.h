#ifndef CONVERTERS_H
#define CONVERTERS_H
#include "layer.h"

const char * get_precision_str(PRECISION precision);
PRECISION get_precision(const char *precision);

void fp32_to_uint8(fp32 *in, uint8_t *out, unsigned count,
                    converter_params params);
void fp32_to_int8(fp32 *in, int8_t *out, unsigned count,
                    converter_params params);
void int8_to_fp32(int8_t *in, fp32 *out, unsigned count,
                    converter_params params);
void uint8_to_fp32(uint8_t *in, fp32 *out, unsigned count,
                    converter_params params);

long long int fp_to_int_elmt(
    double in_element,
    long long int min_value,
    long long int max_value,
    converter_params params,
    int channel);

double int_to_fp_elmt(
    int8_t in_element,
    converter_params params,
    int channel);

#endif
