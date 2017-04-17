#ifndef MATRIX_H
#define MATRIX_H
typedef struct matrix{
    int rows, cols;
    float **vals;
} matrix;

matrix make_matrix(int rows, int cols);
matrix copy_matrix(matrix m);
void free_matrix(matrix m);
void print_matrix(matrix m);

matrix csv_to_matrix(char *filename);
void matrix_to_csv(matrix m);
matrix hold_out_matrix(matrix *m, int n);
float matrix_topk_accuracy(matrix truth, matrix guess, int k);
void matrix_add_matrix(matrix from, matrix to);
void scale_matrix(matrix m, float scale);
matrix resize_matrix(matrix m, int size);

float *pop_column(matrix *m, int c);

#endif
