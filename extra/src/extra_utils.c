// @file extra_utils.c
//
//  \date Created on: Oct 14, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//
#include <stdio.h>
#include <stdint.h>

void PrintArrayF32(const char *name, const float *data, const int n) {
  printf("-------- %s ------------\n", name);
  for (int i = 0; i < n; i++) {
    printf("%.4f, ", data[i]);
    if (i > 0 && (i % 12 == 0)) {
      printf("\n");
    }
  }
  printf("\n------------------------\n");
}

void PrintArrayU16(const char *name, const uint16_t *data, const int n) {
  printf("-------- %s ------------\n", name);
  for (int i = 0; i < n; i++) {
    printf("%d, ", data[i]);
    if (i > 0 && (i % 12 == 0)) {
      printf("\n");
    }
  }
  printf("\n------------------------\n");
}

void PrintArrayU8(const char *name, const uint8_t *data, const int n) {
  printf("-------- %s ------------\n", name);
  for (int i = 0; i < n; i++) {
    printf("%d, ", data[i]);
    if (i > 0 && (i % 12 == 0)) {
      printf("\n");
    }
  }
  printf("\n------------------------\n");
}
