// @file low_precision.c
//
//  \date Created on: Oct 14, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//
#include "lowp_darknet.h"
#include "mem_manager.h"
#include "utils.h"
#include "extra_utils.h"

#include <stddef.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>

#ifndef OPENBLAS
#define OPENBLAS
#endif

#if defined(OPENBLAS) || defined(MKL) || defined(ATLAS)
#ifndef USE_BLAS
#define USE_BLAS
#endif
#endif

#if defined(OPENBLAS) || defined(ATLAS)
#include <cblas.h>
#elif defined(MKL)
#include <mkl.h>
#endif

#define ANALYZE_QUANT_EFFECTS
#define ROUND(a)  floor(a)

LowpTensor CreateLowpTensor(int bytes_per_ele, size_t count) {
  LowpTensor t;
  t.bytes_per_ele = bytes_per_ele;
  t.count = count;
  t.data = dn_malloc(t.count * t.bytes_per_ele);
  t.alloc = true;
  return t;
}

float ArrayMax(const float *data,  const int size) {
  int argmax = max_index(data, size);
  return data[argmax];
}

float ArrayMin(const float *data,  const int size) {
  float min = FLT_MAX;
  for (int i = 0; i < size; i++) {
    if (data[i] < min) {
      min = data[i];
    }
  }
  return min;
}

float QuantError(const float *org, const float *quant, const int n) {
  if (n == 0) {
    return 0;
  }
  float rmse = 0;
  for (int i = 0; i < n; i++) {
    float diff = org[i] - quant[i];
    rmse += (diff * diff);
  }
  rmse = sqrt(rmse) / n;
  return rmse;
}

void ConvertToLowp(const float *f32_data, int no_elements, float max,
                   float min, int bytes_per_element, void *lowp_data) {
  uint8_t *u8 = (uint8_t *)lowp_data;
  uint16_t *u16 = (uint16_t *)lowp_data;

  if (bytes_per_element == 1) {
    float scale = ((1 << 8) - 1) / (max - min);
    for (int i = 0; i < no_elements; i++) {
      *u8++ = ROUND((f32_data[i] - min) * scale);
    }
  } else if (bytes_per_element == 2) {
    float scale = ((1 << 16) - 1) / (max - min);
    //printf("Scale : %f\n", scale);
    for (int i = 0; i < no_elements; i++) {
      *u16++ = ROUND((f32_data[i] - min) * scale);
    }
  } else {
    printf("%d bytes per element is not supported low precision format."
        "Only 1 and 2 bytes per element are supported\n", bytes_per_element);
    exit(-1);
  }
}

void ConvertToF32(const void *lowp_data, int no_elements, float max, float min,
                  int bytes_per_element, float *f32_data) {
  uint8_t *u8 = (uint8_t *)lowp_data;
  uint16_t *u16 = (uint16_t *)lowp_data;

  if (bytes_per_element == 1) {
    float scale =  (max - min) / ((1 << 8) - 1);
    for (int i = 0; i < no_elements; i++) {
      *f32_data++ = (u8[i] * scale + min);
    }
  } else if (bytes_per_element == 2) {
    float scale = (max - min) / ((1 << 16) - 1);
    //printf("Inv Scale : %f\n", scale);
    for (int i = 0; i < no_elements; i++) {
      *f32_data++ = (u16[i] * scale + min);
    }
  } else {
    printf("%d bytes per element is not supported low precision format."
        "Only 1 and 2 bytes per element are supported\n", bytes_per_element);
    exit(-1);
  }
}

LowpTensor FloatToLowp(const float *f32_data, const int no_elements,
                       const int bytes_per_element, void *lowp_data) {
  LowpTensor sx_tensor;
  sx_tensor.bytes_per_ele = bytes_per_element;
  sx_tensor.count = no_elements;
  if (lowp_data == NULL) {
    sx_tensor.data = dn_malloc(sx_tensor.bytes_per_ele * sx_tensor.count);
    sx_tensor.alloc = true;
  } else {
    sx_tensor.data = lowp_data;
    sx_tensor.alloc = false;
  }
  sx_tensor.max = ArrayMax(f32_data, no_elements);
  sx_tensor.min = ArrayMin(f32_data, no_elements);

  ConvertToLowp(f32_data, no_elements, sx_tensor.max, sx_tensor.min,
                sx_tensor.bytes_per_ele, sx_tensor.data);

#ifdef ANALYZE_QUANT_EFFECTS
  float *quant_f32 = dn_malloc(no_elements * sizeof(float));
  ConvertToF32(sx_tensor.data, sx_tensor.count, sx_tensor.max,
               sx_tensor.min, sx_tensor.bytes_per_ele, quant_f32);
  float rmse = QuantError(f32_data, quant_f32, no_elements);
  printf("Max : %f\tMin : %f\t", sx_tensor.max, sx_tensor.min);
  printf("RMSE over %d element = %f\n", no_elements, rmse);
  if (0) {
    PrintArrayF32("Original", f32_data, no_elements);

    PrintArrayF32("Quantized", quant_f32, no_elements);
    PrintArrayU16("Lowp int", sx_tensor.data, no_elements);
  }
#endif

  return sx_tensor;
}

void *LowpToFloat(LowpTensor *lowp_tensor, float *f32_data) {
  float *data;
  if (f32_data == NULL) {
    data = dn_malloc(lowp_tensor->count * sizeof(float));
  } else {
    data = f32_data;
  }
  ConvertToF32(lowp_tensor->data, lowp_tensor->count, lowp_tensor->max,
               lowp_tensor->min, lowp_tensor->bytes_per_ele, data);

  // If the buffer is allocated by FloatTo8bit() then release it.
  if (lowp_tensor->alloc) {
    dn_free(lowp_tensor->data);
    lowp_tensor->alloc = false;
    lowp_tensor->data = NULL;
    lowp_tensor->count = 0;
  }
  return data;
}

void WriteLowpTensorToFile(LowpTensor t, FILE *fp) {
  fwrite(&t.max, sizeof(float), 1, fp);
  fwrite(&t.min, sizeof(float), 1, fp);
  fwrite(t.data, t.bytes_per_ele, t.count, fp);
}

LowpTensor ReadLowpTensorFromFile(FILE *fp, int bytes_per_ele, size_t count) {
  LowpTensor t = CreateLowpTensor(bytes_per_ele, count);
  fread(&t.max, sizeof(float), 1, fp);
  fread(&t.min, sizeof(float), 1, fp);
  fread(t.data, bytes_per_ele, count, fp);
  return t;
}

void LoadLowpTensorAsFloatFromFile(float *f32_data, FILE *fp,
                                   int bytes_per_ele, size_t count) {
  LowpTensor t = ReadLowpTensorFromFile(fp, bytes_per_ele, count);
  LowpToFloat(&t, f32_data);
  // buffer allocated for tensor t is free up by LowpToFloat()
}

void FreeLowpTensor(LowpTensor t) {
  if (t.alloc) {
    dn_free(t.data);
  }
}
