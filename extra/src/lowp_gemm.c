// @file lowp_gemm.c
//
//  \date Created on: Oct 15, 2017
//  \author Gopalakrishna Hegde
//
//   Description:
//
//
//
#include "blas.h"
#include <cblas.h>
#include <stdint.h>

// Unoptimized implementations for C = alpha A*B + beta * C where A is in low
// precision format (8/16b) and B and C are in single precision floating format.
void CblasSgemmLowpTT(const int layout, const int m, const int n, const int k,
                      const float alpha, const void *a, const int lda,
                      const float *b, const int ldb, float *c, const int ldc,
                      const float amax, const float amin, const int abits) {
  if (layout == CblasRowMajor) { // row major
    if (abits == 8) {
      uint8_t *mat_a = (uint8_t *)a;
      float scale =  (amax - amin) / ((1 << 8) - 1);
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          float sum = 0;
          for (int e = 0; e < k; e++ ) {
            int a_addr = e * lda + i;
            int b_addr = j * ldb + e;
            uint8_t aval_u8 = mat_a[a_addr];
            float bval = b[b_addr];
            float aval_f32 = aval_u8 * scale + amin;
            sum += (bval * aval_f32);
          }
          int c_addr = i * ldc + j;
          c[c_addr] = alpha * sum;
        }
      }
    } else if (abits == 16) {
      uint16_t *mat_a = (uint16_t *)a;
      float scale =  (amax - amin) / ((1 << 16) - 1);
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          float sum = 0;
          for (int e = 0; e < k; e++ ) {
            int a_addr = e * lda + i;
            int b_addr = j * ldb + e;
            uint16_t aval_u16 = mat_a[a_addr];
            float bval = b[b_addr];
            float aval_f32 = aval_u16 * scale + amin;
            sum += (bval * aval_f32);
          }
          int c_addr = i * ldc + j;
          c[c_addr] = alpha * sum;
        }
      }
    } else {
      printf("Unsupported.\n");
    }
  } else { // col major
    if (abits == 8) {
      uint8_t *mat_a = (uint8_t *)a;
      float scale =  (amax - amin) / ((1 << 8) - 1);
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          float sum = 0;
          for (int e = 0; e < k; e++ ) {
            int a_addr = i * lda + e;
            int b_addr = e * ldb + j;
            uint8_t aval_u8 = mat_a[a_addr];
            float bval = b[b_addr];
            float aval_f32 = aval_u8 * scale + amin;
            sum += (bval * aval_f32);
          }
          int c_addr = j * ldc + i;
          c[c_addr] = alpha * sum;
        }
      }
    } else if (abits == 16) {
      uint16_t *mat_a = (uint16_t *)a;
      float scale =  (amax - amin) / ((1 << 16) - 1);
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          float sum = 0;
          for (int e = 0; e < k; e++ ) {
            int a_addr = i * lda + e;
            int b_addr = e * ldb + j;
            uint16_t aval_u16 = mat_a[a_addr];
            float bval = b[b_addr];
            float aval_f32 = aval_u16 * scale + amin;
            sum += (bval * aval_f32);
          }
          int c_addr = j * ldc + i;
          c[c_addr] = alpha * sum;
        }
      }
    } else {
      printf("Unsupported.\n");
    }
  }
}

void CblasSgemmLowpTN(const int layout, const int m, const int n, const int k,
                      const float alpha, const void *a, const int lda,
                      const float *b, const int ldb, float *c, const int ldc,
                      const float amax, const float amin, const int abits) {
  if (layout == CblasRowMajor) { // row major
    if (abits == 8) {
      uint8_t *mat_a = (uint8_t *)a;
      float scale =  (amax - amin) / ((1 << 8) - 1);
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          float sum = 0;
          for (int e = 0; e < k; e++ ) {
            int a_addr = e * lda + i;
            int b_addr = e * ldb + j;
            uint8_t aval_u8 = mat_a[a_addr];
            float bval = b[b_addr];
            float aval_f32 = aval_u8 * scale + amin;
            sum += (bval * aval_f32);
          }
          int c_addr = i * ldc + j;
          c[c_addr] = alpha * sum;
        }
      }
    } else if (abits == 16) {
      uint16_t *mat_a = (uint16_t *)a;
      float scale =  (amax - amin) / ((1 << 16) - 1);
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          float sum = 0;
          for (int e = 0; e < k; e++ ) {
            int a_addr = e * lda + i;
            int b_addr = e * ldb + j;
            uint16_t aval_u16 = mat_a[a_addr];
            float bval = b[b_addr];
            float aval_f32 = aval_u16 * scale + amin;
            sum += (bval * aval_f32);
          }
          int c_addr = i * ldc + j;
          c[c_addr] = alpha * sum;
        }
      }
    } else {
      printf("Unsupported.\n");
    }
  } else {
    if (abits == 8) {
      uint8_t *mat_a = (uint8_t *)a;
      float scale =  (amax - amin) / ((1 << 8) - 1);
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          float sum = 0;
          for (int e = 0; e < k; e++ ) {
            int a_addr = i * lda + e;
            int b_addr = j * ldb + e;
            uint8_t aval_u8 = mat_a[a_addr];
            float bval = b[b_addr];
            float aval_f32 = aval_u8 * scale + amin;
            sum += (bval * aval_f32);
          }
          int c_addr = j * ldc + i;
          c[c_addr] = alpha * sum;
        }
      }
    } else if (abits == 16) {
      uint16_t *mat_a = (uint16_t *)a;
      float scale =  (amax - amin) / ((1 << 16) - 1);
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          float sum = 0;
          for (int e = 0; e < k; e++ ) {
            int a_addr = i * lda + e;
            int b_addr = j * ldb + e;
            uint16_t aval_u16 = mat_a[a_addr];
            float bval = b[b_addr];
            float aval_f32 = aval_u16 * scale + amin;
            sum += (bval * aval_f32);
          }
          int c_addr = j * ldc + i;
          c[c_addr] = alpha * sum;
        }
      }
    } else {
      printf("Unsupported.\n");
    }
  }
}

void CblasSgemmLowpNT(const int layout, const int m, const int n, const int k,
                      const float alpha, const void *a, const int lda,
                      const float *b, const int ldb, float *c, const int ldc,
                      const float amax, const float amin, const int abits) {
  if (layout == CblasRowMajor) { // row major
    if (abits == 8) {
      uint8_t *mat_a = (uint8_t *)a;
      float scale =  (amax - amin) / ((1 << 8) - 1);
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          float sum = 0;
          for (int e = 0; e < k; e++ ) {
            int a_addr = i * lda + e;
            int b_addr = j * ldb + e;
            uint8_t aval_u8 = mat_a[a_addr];
            float bval = b[b_addr];
            float aval_f32 = aval_u8 * scale + amin;
            sum += (bval * aval_f32);
          }
          int c_addr = i * ldc + j;
          c[c_addr] = alpha * sum;
        }
      }
    } else if (abits == 16) {
      uint16_t *mat_a = (uint16_t *)a;
      float scale =  (amax - amin) / ((1 << 16) - 1);
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          float sum = 0;
          for (int e = 0; e < k; e++ ) {
            int a_addr = i * lda + e;
            int b_addr = j * ldb + e;
            uint16_t aval_u16 = mat_a[a_addr];
            float bval = b[b_addr];
            float aval_f32 = aval_u16 * scale + amin;
            sum += (bval * aval_f32);
          }
          int c_addr = i * ldc + j;
          c[c_addr] = alpha * sum;
        }
      }
    } else {
      printf("Unsupported.\n");
    }
  } else {
    if (abits == 8) {
      uint8_t *mat_a = (uint8_t *)a;
      float scale =  (amax - amin) / ((1 << 8) - 1);
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          float sum = 0;
          for (int e = 0; e < k; e++ ) {
            int a_addr = e * lda + i;
            int b_addr = e * ldb + j;
            uint8_t aval_u8 = mat_a[a_addr];
            float bval = b[b_addr];
            float aval_f32 = aval_u8 * scale + amin;
            sum += (bval * aval_f32);
          }
          int c_addr = j * ldc + i;
          c[c_addr] = alpha * sum;
        }
      }
    } else if (abits == 16) {
      uint16_t *mat_a = (uint16_t *)a;
      float scale =  (amax - amin) / ((1 << 16) - 1);
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          float sum = 0;
          for (int e = 0; e < k; e++ ) {
            int a_addr = e * lda + i;
            int b_addr = e * ldb + j;
            uint16_t aval_u16 = mat_a[a_addr];
            float bval = b[b_addr];
            float aval_f32 = aval_u16 * scale + amin;
            sum += (bval * aval_f32);
          }
          int c_addr = j * ldc + i;
          c[c_addr] = alpha * sum;
        }
      }
    } else {
      printf("Unsupported.\n");
    }
  }
}

void CblasSgemmLowpNN(const int layout, const int m, const int n, const int k,
                      const float alpha, const void *a, const int lda,
                      const float *b, const int ldb, float *c, const int ldc,
                      const float amax, const float amin, const int abits) {
  if (layout == CblasRowMajor) { // row major
    if (abits == 8) {
      uint8_t *mat_a = (uint8_t *)a;
      float scale =  (amax - amin) / ((1 << 8) - 1);
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          float sum = 0;
          for (int e = 0; e < k; e++ ) {
            int a_addr = i * lda + e;
            int b_addr = e * ldb + j;
            uint8_t aval_u8 = mat_a[a_addr];
            float bval = b[b_addr];
            float aval_f32 = aval_u8 * scale + amin;
            sum += (bval * aval_f32);
          }
          int c_addr = i * ldc + j;
          c[c_addr] = alpha * sum;
        }
      }
    } else if (abits == 16) {
      uint16_t *mat_a = (uint16_t *)a;
      float scale =  (amax - amin) / ((1 << 16) - 1);
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          float sum = 0;
          for (int e = 0; e < k; e++ ) {
            int a_addr = i * lda + e;
            int b_addr = e * ldb + j;
            uint16_t aval_u16 = mat_a[a_addr];
            float bval = b[b_addr];
            float aval_f32 = aval_u16 * scale + amin;
            sum += (bval * aval_f32);
          }
          int c_addr = i * ldc + j;
          c[c_addr] = alpha * sum;
        }
      }
    } else {
      printf("Unsupported.\n");
    }
  } else {
    if (abits == 8) {
      uint8_t *mat_a = (uint8_t *)a;
      float scale =  (amax - amin) / ((1 << 8) - 1);
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          float sum = 0;
          for (int e = 0; e < k; e++ ) {
            int a_addr = e * lda + i;
            int b_addr = j * ldb + e;
            uint8_t aval_u8 = mat_a[a_addr];
            float bval = b[b_addr];
            float aval_f32 = aval_u8 * scale + amin;
            sum += (bval * aval_f32);
          }
          int c_addr = j * ldc + i;
          c[c_addr] = alpha * sum;
        }
      }
    } else if (abits == 16) {
      uint16_t *mat_a = (uint16_t *)a;
      float scale =  (amax - amin) / ((1 << 16) - 1);
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          float sum = 0;
          for (int e = 0; e < k; e++ ) {
            int a_addr = e * lda + i;
            int b_addr = j * ldb + e;
            uint16_t aval_u16 = mat_a[a_addr];
            float bval = b[b_addr];
            float aval_f32 = aval_u16 * scale + amin;
            sum += (bval * aval_f32);
          }
          int c_addr = j * ldc + i;
          c[c_addr] = alpha * sum;
        }
      }
    } else {
      printf("Unsupported.\n");
    }
  }
}

void cblas_sgemm_lowp(const int layout, const int transa, const int transb,
                     const int m, const int n, const int k, const float alpha,
                     const void *a, const int lda, const float *b, const int ldb,
                     const float beta, float *c, const int ldc,
                     const float amax, const float amin, const int abits) {

  scal_cpu(m * n, beta, c, 1);
  if (!transa && !transb) {
    CblasSgemmLowpNN(layout, m, n, k, alpha, a, lda, b, ldb, c, ldc, amax,
                            amin, abits);
  } else if (transa && !transb) {
    CblasSgemmLowpTN(layout, m, n, k, alpha, a, lda, b, ldb, c, ldc, amax,
                            amin, abits);
  } else if (!transa && transb) {
    CblasSgemmLowpNT(layout, m, n, k, alpha, a, lda, b, ldb, c, ldc, amax,
                            amin, abits);
  } else {
    CblasSgemmLowpTT(layout, m, n, k, alpha, a, lda, b, ldb, c, ldc, amax,
                        amin, abits);
  }
}


