#ifndef MATRIX_H
#define MATRIX_H
#include "darknet.h"

dn_matrix copy_matrix(dn_matrix m);
void print_matrix(dn_matrix m);

dn_matrix hold_out_matrix(dn_matrix *m, int n);
dn_matrix resize_matrix(dn_matrix m, int size);

float *pop_column(dn_matrix *m, int c);

#endif
