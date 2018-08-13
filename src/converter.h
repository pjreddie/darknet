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
#endif
